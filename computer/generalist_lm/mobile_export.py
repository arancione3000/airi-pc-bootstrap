from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .runtime import GeneralistRuntime
from .tokenizer import BOS, USER, ASSISTANT


MOBILE_SCHEMA = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric(source: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    value = source.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _export_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    slot_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    runtime = GeneralistRuntime.from_checkpoint(checkpoint, device="cpu")
    runtime.model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    class NextTokenLogits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            result = self.model(input_ids, use_cache=False)
            return result["logits"][:, -1, :]

    wrapper = NextTokenLogits(runtime.model).eval()
    sample = torch.tensor(
        [[BOS, USER, ASSISTANT]],
        dtype=torch.long,
    )
    model_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (sample,),
        model_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "sequence"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    with torch.no_grad():
        torch_logits = wrapper(sample).detach().cpu().numpy()
    ort_logits = session.run(
        ["logits"],
        {"input_ids": sample.cpu().numpy().astype(np.int64)},
    )[0]
    max_abs_error = float(np.max(np.abs(torch_logits - ort_logits)))
    if not np.isfinite(max_abs_error) or max_abs_error > 5e-4:
        raise RuntimeError(
            f"ONNX validation mismatch for {slot_name}: max_abs_error={max_abs_error}"
        )

    config_path = output_dir / "config.json"
    _write_json(config_path, runtime.config.to_dict())

    tokenizer_path = output_dir / "tokenizer.json"
    if runtime.tokenizer.version == "bpe-v1":
        runtime.tokenizer.save(tokenizer_path)
    else:
        _write_json(
            tokenizer_path,
            {
                "version": "byte-v1",
                "vocab_size": int(runtime.tokenizer.vocab_size),
            },
        )

    source_metadata = checkpoint / "metadata.json"
    metadata_path = output_dir / "metadata.json"
    if source_metadata.is_file():
        shutil.copy2(source_metadata, metadata_path)
    else:
        _write_json(metadata_path, {})

    files = {}
    for name in ("model.onnx", "config.json", "tokenizer.json", "metadata.json"):
        file_path = output_dir / name
        files[name] = {
            "sha256": _sha256(file_path),
            "bytes": int(file_path.stat().st_size),
        }

    return {
        "name": slot_name,
        "id": str(
            summary.get("candidate_id")
            or summary.get("genome_id")
            or summary.get("id")
            or slot_name
        ),
        "path": slot_name,
        "cycle": int(summary.get("cycle", 0) or 0),
        "parameters": int(
            summary.get("parameters")
            or summary.get("parameter_count")
            or 0
        ),
        "score": _metric(summary, "score"),
        "nll_per_byte": _metric(
            summary,
            "nll_per_byte",
            _metric(summary, "mean_nll_per_byte"),
        ),
        "generation_similarity": _metric(
            summary,
            "generation_similarity",
            _metric(summary, "mean_generation_similarity"),
        ),
        "generation_exact_accuracy": _metric(
            summary,
            "generation_exact_accuracy",
            _metric(summary, "mean_generation_accuracy"),
        ),
        "generation_nonempty_rate": _metric(
            summary,
            "generation_nonempty_rate",
            _metric(summary, "mean_generation_nonempty_rate"),
        ),
        "tokenizer_version": str(runtime.tokenizer.version),
        "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
        "context_length": int(runtime.config.context_length),
        "research_only": bool(summary.get("research_only", slot_name != "champion")),
        "all_seed_eligible": bool(summary.get("all_seed_eligible", slot_name == "champion")),
        "onnx_max_abs_error": max_abs_error,
        "files": files,
    }


def export_mobile_bundle(
    state_dir: str | Path,
    output_dir: str | Path,
    *,
    state_sha: str = "",
) -> dict[str, Any]:
    state = Path(state_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    status_path = state / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError("generalist state is missing status.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))

    champion_report = dict(status.get("champion_report") or {})
    champion_genome = dict(status.get("champion") or {})
    champion_summary = {
        **champion_report,
        "candidate_id": champion_genome.get("genome_id", "champion"),
        "cycle": int(status.get("cycle", 0) or 0),
        "research_only": False,
        "all_seed_eligible": True,
    }
    slots: dict[str, Any] = {
        "champion": _export_checkpoint(
            state / "champion",
            output / "champion",
            slot_name="champion",
            summary=champion_summary,
        )
    }

    research_checkpoint = state / "latest-research"
    research_summary_path = research_checkpoint / "research-summary.json"
    if research_checkpoint.is_dir() and research_summary_path.is_file():
        research_summary = json.loads(
            research_summary_path.read_text(encoding="utf-8")
        )
        slots["research"] = _export_checkpoint(
            research_checkpoint,
            output / "research",
            slot_name="research",
            summary=research_summary,
        )

    manifest = {
        "schema": MOBILE_SCHEMA,
        "state_sha": str(state_sha),
        "cycle": int(status.get("cycle", 0) or 0),
        "generated_at": int(time.time()),
        "generalist_version": str(status.get("version", "")),
        "promoted_this_cycle": bool(status.get("promoted")),
        "promotion_reason": str(status.get("promotion_reason", "")),
        "slots": slots,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export AIRI Generalist checkpoints for Android ONNX inference"
    )
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-sha", default="")
    args = parser.parse_args(argv)
    manifest = export_mobile_bundle(
        args.state_dir,
        args.output,
        state_sha=args.state_sha,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
