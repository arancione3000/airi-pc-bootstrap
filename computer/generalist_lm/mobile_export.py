from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .airi_pc_lab import (
    build_airi_pc_lab_rows,
    run_airi_pc_lab_probe,
    snapshot_airi_pc_lab,
    summarize_lab_learning,
)
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


def _safe_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _research_profile(
    checkpoint: Path,
    summary: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    metrics = _safe_json(checkpoint / "research-metrics.json", {})
    if not isinstance(metrics, dict):
        metrics = {}
    merged = {**metrics, **dict(summary)}
    merged["research_source"] = source
    merged["generation_pathological_repetition"] = bool(
        metrics.get(
            "generation_pathological_repetition",
            summary.get(
                "pathological_repetition",
                summary.get("any_generation_pathological_repetition", True),
            ),
        )
    )
    merged["generation_repetition_rate"] = _metric(
        metrics,
        "generation_repetition_rate",
        _metric(
            summary,
            "mean_generation_repetition_rate",
            _metric(summary, "generation_repetition_rate", 1.0),
        ),
    )
    merged["generation_similarity"] = _metric(
        metrics,
        "generation_similarity",
        _metric(
            summary,
            "mean_generation_similarity",
            _metric(summary, "generation_similarity", 0.0),
        ),
    )
    merged["nll_per_byte"] = _metric(
        metrics,
        "nll_per_byte",
        _metric(
            summary,
            "mean_nll_per_byte",
            _metric(summary, "nll_per_byte", float("inf")),
        ),
    )
    merged["worst_domain_regression"] = _metric(
        summary,
        "worst_domain_regression",
        _metric(metrics, "worst_domain_regression", float("inf")),
    )
    merged["score"] = _metric(
        summary,
        "score",
        _metric(metrics, "score", 0.0),
    )
    return merged


def _research_mobile_key(profile: dict[str, Any]) -> tuple:
    return (
        0 if bool(profile.get("all_seed_eligible")) else 1,
        bool(profile.get("generation_pathological_repetition", True)),
        float(profile.get("worst_domain_regression", float("inf"))),
        float(profile.get("generation_repetition_rate", 1.0)),
        -float(profile.get("generation_similarity", 0.0)),
        float(profile.get("nll_per_byte", float("inf"))),
        -float(profile.get("score", 0.0)),
    )


def _select_mobile_research(
    state: Path,
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []

    latest = state / "latest-research"
    latest_summary = _safe_json(latest / "research-summary.json", {})
    if latest.is_dir() and isinstance(latest_summary, dict) and latest_summary:
        candidates.append((
            latest,
            _research_profile(
                latest,
                latest_summary,
                source="latest-research",
            ),
        ))

    incumbent = state / "architecture-research" / "incumbent"
    incumbent_checkpoint = incumbent / "checkpoint"
    incumbent_summary = _safe_json(incumbent / "summary.json", {})
    if (
        incumbent_checkpoint.is_dir()
        and isinstance(incumbent_summary, dict)
        and incumbent_summary
        and not bool(incumbent_summary.get("external_pretrained"))
    ):
        candidates.append((
            incumbent_checkpoint,
            _research_profile(
                incumbent_checkpoint,
                incumbent_summary,
                source="architecture-incumbent",
            ),
        ))

    if not candidates:
        return None
    return min(candidates, key=lambda row: _research_mobile_key(row[1]))


def _neural_diagnostics(runtime) -> dict[str, Any]:
    import torch

    cfg = runtime.config
    groups: dict[str, dict[str, float]] = {}

    def group_name(name: str) -> str:
        if name.startswith("token_embedding"):
            return "embedding"
        if name.startswith("blocks."):
            parts = name.split(".")
            layer = parts[1] if len(parts) > 1 else "?"
            if ".attn." in name:
                return f"layer_{layer}_attention"
            if ".ff." in name:
                return f"layer_{layer}_ffn"
            return f"layer_{layer}_norm"
        if name.startswith("final_norm"):
            return "final_norm"
        if name.startswith("position_embedding"):
            return "position_embedding"
        return "other"

    with torch.no_grad():
        for name, parameter in runtime.model.named_parameters():
            tensor = parameter.detach().float().cpu()
            count = int(tensor.numel())
            if not count:
                continue
            key = group_name(name)
            row = groups.setdefault(
                key,
                {
                    "parameters": 0.0,
                    "abs_sum": 0.0,
                    "sq_sum": 0.0,
                    "max_abs": 0.0,
                },
            )
            row["parameters"] += count
            row["abs_sum"] += float(tensor.abs().sum().item())
            row["sq_sum"] += float(tensor.pow(2).sum().item())
            row["max_abs"] = max(
                float(row["max_abs"]),
                float(tensor.abs().max().item()),
            )

    components: list[dict[str, Any]] = []
    order = ["embedding", "position_embedding"]
    for layer in range(int(cfg.n_layers)):
        order.extend(
            [
                f"layer_{layer}_attention",
                f"layer_{layer}_norm",
                f"layer_{layer}_ffn",
            ]
        )
    order.extend(["final_norm", "other"])
    for key in order:
        row = groups.get(key)
        if not row:
            continue
        count = max(1.0, float(row["parameters"]))
        components.append(
            {
                "id": key,
                "label": key.replace("_", " ").title(),
                "parameters": int(row["parameters"]),
                "mean_abs_weight": float(row["abs_sum"]) / count,
                "rms_weight": (float(row["sq_sum"]) / count) ** 0.5,
                "max_abs_weight": float(row["max_abs"]),
            }
        )

    heads: list[dict[str, Any]] = []
    for layer in range(int(cfg.n_layers)):
        module = runtime.model.blocks[layer].attn
        head_dim = int(cfg.d_model) // int(cfg.n_heads)
        if getattr(module, "qkv", None) is not None:
            weight = module.qkv.weight.detach().float().cpu()
            reshaped = weight.view(
                3,
                int(cfg.n_heads),
                head_dim,
                int(cfg.d_model),
            )
            per_head = [reshaped[:, head] for head in range(int(cfg.n_heads))]
            head_kind = "mha_qkv"
        else:
            # GQA has fewer KV heads, so a single QKV tensor per query head
            # does not exist.  Visualize the query-head weights truthfully
            # rather than fabricating duplicated KV parameters.
            weight = module.q_proj.weight.detach().float().cpu()
            reshaped = weight.view(
                int(cfg.n_heads),
                head_dim,
                int(cfg.d_model),
            )
            per_head = [reshaped[head] for head in range(int(cfg.n_heads))]
            head_kind = "gqa_query"
        for head, tensor in enumerate(per_head):
            heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "kind": head_kind,
                    "parameters": int(tensor.numel()),
                    "mean_abs_weight": float(tensor.abs().mean().item()),
                    "rms_weight": float(tensor.pow(2).mean().sqrt().item()),
                    "max_abs_weight": float(tensor.abs().max().item()),
                }
            )

    node_ids = [row["id"] for row in components]
    connections = [
        {
            "from": node_ids[index],
            "to": node_ids[index + 1],
            "kind": "residual_flow",
        }
        for index in range(max(0, len(node_ids) - 1))
    ]
    return {
        "kind": "aggregated_checkpoint_weights",
        "note": (
            "This is a truthful aggregate view of checkpoint weights and model "
            "topology, not a claim to display every biological-style neuron."
        ),
        "architecture": {
            "type": "decoder_only_causal_transformer",
            "vocab_size": int(cfg.vocab_size),
            "context_length": int(cfg.context_length),
            "d_model": int(cfg.d_model),
            "n_layers": int(cfg.n_layers),
            "n_heads": int(cfg.n_heads),
            "head_dim": int(cfg.d_model) // int(cfg.n_heads),
            "d_ff": int(cfg.d_ff),
            "norm_type": str(cfg.norm_type),
            "position_encoding": str(cfg.position_encoding),
            "ff_variant": str(cfg.ff_variant),
            "attention_type": str(cfg.attention_type),
            "n_kv_heads": int(cfg.n_kv_heads or cfg.n_heads),
            "local_attention_window": int(cfg.local_attention_window),
            "local_attention_every": int(cfg.local_attention_every),
            "norm_placement": str(cfg.norm_placement),
            "tied_embedding_lm_head": bool(cfg.tie_embeddings),
        },
        "components": components,
        "heads": heads,
        "connections": connections,
    }


def _mobile_airi_pc_lab(status: dict[str, Any], state: Path) -> dict[str, Any]:
    existing = status.get("airi_pc_lab")
    if isinstance(existing, dict) and existing.get("version"):
        return existing
    persisted = _safe_json(state / "airi-pc-lab-report.json", {})
    if isinstance(persisted, dict) and persisted.get("version"):
        return persisted

    snapshot = snapshot_airi_pc_lab(Path.cwd())
    rows = build_airi_pc_lab_rows(snapshot, max_rows=18)
    report: dict[str, Any] = {
        "version": str(snapshot.get("version") or "airi-pc-lab-v1"),
        "mode": str(snapshot.get("mode") or "read_only_sandbox"),
        "snapshot": snapshot,
        "learning": summarize_lab_learning(rows),
    }
    try:
        champion = GeneralistRuntime.from_checkpoint(state / "champion", device="cpu")
        report["champion"] = run_airi_pc_lab_probe(champion, snapshot)
    except Exception as exc:
        report["champion"] = {
            "ok": False,
            "tool_call_valid": False,
            "error": f"{type(exc).__name__}:{exc}",
        }

    selected_research = _select_mobile_research(state)
    if selected_research is not None:
        research_path, research_summary = selected_research
        try:
            research = GeneralistRuntime.from_checkpoint(research_path, device="cpu")
            report["research"] = {
                **run_airi_pc_lab_probe(research, snapshot),
                "research_source": research_summary.get("research_source"),
                "candidate_id": research_summary.get("candidate_id"),
            }
        except Exception as exc:
            report["research"] = {
                "ok": False,
                "tool_call_valid": False,
                "research_source": research_summary.get("research_source"),
                "candidate_id": research_summary.get("candidate_id"),
                "error": f"{type(exc).__name__}:{exc}",
            }
    return report


def _evolution_summary(status: dict[str, Any], state: Path) -> dict[str, Any]:
    growth = dict(status.get("automatic_data_growth") or {})
    manifest = _safe_json(state / "autodata" / "manifest.json", {"files": []})
    repo_stats: dict[str, dict[str, Any]] = {}
    for row in manifest.get("files") or []:
        if not isinstance(row, dict):
            continue
        repo = str(row.get("repo") or "")
        if not repo:
            continue
        stat = repo_stats.setdefault(
            repo,
            {"repo": repo, "files": 0, "bytes": 0, "domains": {}},
        )
        stat["files"] += 1
        stat["bytes"] += int(row.get("bytes", 0) or 0)
        domain = str(row.get("domain") or "general")
        stat["domains"][domain] = stat["domains"].get(domain, 0) + 1

    top_repositories = sorted(
        repo_stats.values(),
        key=lambda row: (-int(row["bytes"]), -int(row["files"]), str(row["repo"])),
    )[:12]
    rejection_counts: dict[str, int] = {}
    for row in growth.get("rejected") or []:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "unknown")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    return {
        "signals": list(status.get("signals") or []),
        "domain_weights": dict(
            (status.get("adaptive_curriculum") or {}).get("domain_weights") or {}
        ),
        "curriculum_memory": status.get("curriculum_memory") or {},
        "data_growth": {
            "version": growth.get("version"),
            "queries": list(growth.get("queries") or []),
            "desired_domains": list(growth.get("desired_domains") or []),
            "added_files": int(growth.get("added_files", 0) or 0),
            "added_bytes": int(growth.get("added_bytes", 0) or 0),
            "total_files": int(growth.get("total_files", 0) or 0),
            "total_bytes": int(growth.get("total_bytes", 0) or 0),
            "domain_files": dict(growth.get("domain_files") or {}),
            "domain_bytes": dict(growth.get("domain_bytes") or {}),
            "repositories_used": int(growth.get("repositories_used", 0) or 0),
            "top_repositories": top_repositories,
            "rejection_counts": rejection_counts,
        },
        "progressive_scaling": status.get("progressive_scaling") or {},
        "progressive_tokenizer": status.get("progressive_tokenizer") or {},
        "plateau": status.get("plateau") or {},
        "latest_research": status.get("latest_research"),
        "policy": status.get("policy") or {},
    }


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
        dynamo=True,
        external_data=False,
    )
    onnx_model = onnx.load(str(model_path), load_external_data=False)
    external_initializers = [
        initializer.name
        for initializer in onnx_model.graph.initializer
        if int(initializer.data_location) == int(onnx.TensorProto.EXTERNAL)
        or bool(initializer.external_data)
    ]
    if external_initializers:
        raise RuntimeError(
            f"mobile ONNX for {slot_name} is not self-contained: "
            f"external initializers={external_initializers[:8]}"
        )
    onnx.checker.check_model(onnx_model)

    # No sidecar is permitted in the Android bundle. Catch exporter behavior
    # changes even if ONNX metadata is malformed or incomplete.
    sidecars = sorted(
        path.name
        for path in output_dir.glob("model.onnx*")
        if path.name != "model.onnx"
    )
    if sidecars:
        raise RuntimeError(
            f"mobile ONNX for {slot_name} emitted forbidden sidecars: {sidecars}"
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    dynamic_probe = torch.tensor(
        [[BOS, USER, 10, 11, ASSISTANT]],
        dtype=torch.long,
    )
    max_abs_error = 0.0
    for probe in (sample, dynamic_probe):
        with torch.no_grad():
            torch_logits = wrapper(probe).detach().cpu().numpy()
        ort_logits = session.run(
            ["logits"],
            {"input_ids": probe.cpu().numpy().astype(np.int64)},
        )[0]
        error = float(np.max(np.abs(torch_logits - ort_logits)))
        max_abs_error = max(max_abs_error, error)
        if not np.isfinite(error) or error > 5e-4:
            raise RuntimeError(
                f"ONNX validation mismatch for {slot_name}: "
                f"sequence={int(probe.shape[1])} max_abs_error={error}"
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
        "neural": _neural_diagnostics(runtime),
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

    selected_research = _select_mobile_research(state)
    if selected_research is not None:
        research_checkpoint, research_summary = selected_research
        slots["research"] = _export_checkpoint(
            research_checkpoint,
            output / "research",
            slot_name="research",
            summary=research_summary,
        )
        slots["research"]["research_source"] = str(
            research_summary.get("research_source") or "unknown"
        )

    manifest = {
        "schema": MOBILE_SCHEMA,
        "state_sha": str(state_sha),
        "cycle": int(status.get("cycle", 0) or 0),
        "generated_at": int(time.time()),
        "generalist_version": str(status.get("version", "")),
        "promoted_this_cycle": bool(status.get("promoted")),
        "promotion_reason": str(status.get("promotion_reason", "")),
        "evolution": _evolution_summary(status, state),
        "airi_pc_lab": _mobile_airi_pc_lab(status, state),
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
