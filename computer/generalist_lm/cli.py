from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .distillation import DistillationPrompt, distill_prompts, save_distilled_jsonl
from .hf_backend import LocalTransformersBackend
from .foundation import FoundationManifest, foundation_identity, write_foundation_manifest
from .pretraining import load_local_corpus, pretrain_causal
from .qualification import (
    qualify_checkpoint,
    qualify_transformers_model,
    qualification_status,
    transformers_qualification_status,
    qualify_foundation_model,
    foundation_qualification_status,
)
from .research_cycle import run_research_cycle
from .research_health import research_health
from .runtime import GeneralistRuntime
from .training import load_sft_jsonl
from .transformers_lora import LoRATrainConfig, train_local_lora


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIRI Generalist LM research, learning and qualification CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status")
    status.add_argument("--state", default=os.environ.get("AIRI_GENERALIST_STATE", ".ai/generalist-lm/champion"))

    qn = sub.add_parser("qualify-native")
    qn.add_argument("state")
    qn.add_argument("--minimum-score", type=float, default=85.0)

    qt = sub.add_parser("qualify-transformers")
    qt.add_argument("model_dir")
    qt.add_argument("--attestation")
    qt.add_argument("--minimum-score", type=float, default=85.0)

    ts = sub.add_parser("transformers-status")
    ts.add_argument("model_dir")
    ts.add_argument("--attestation")

    fi = sub.add_parser("foundation-init", help="write a reviewed local foundation-model manifest")
    fi.add_argument("model_dir")
    fi.add_argument("--model-id", required=True)
    fi.add_argument("--revision", required=True)
    fi.add_argument("--license", required=True)
    fi.add_argument("--architecture", required=True)
    fi.add_argument("--context-length", type=int, required=True)
    fi.add_argument("--parameter-count", type=int, default=0)
    fi.add_argument("--dtype", default="unknown")
    fi.add_argument("--quantization", default="none")

    fq = sub.add_parser("qualify-foundation", help="run the harder protected foundation suite")
    fq.add_argument("model_dir")
    fq.add_argument("--attestation")
    fq.add_argument("--minimum-score", type=float, default=90.0)
    fq.add_argument("--device", default="cpu")
    fq.add_argument(
        "--device-map",
        choices=("none", "auto", "balanced", "balanced_low_0", "sequential"),
        default="auto",
    )
    fq.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    fq.add_argument("--max-memory-json")
    fq.add_argument("--offload-folder")

    fs = sub.add_parser("foundation-status")
    fs.add_argument("model_dir")
    fs.add_argument("--attestation")

    rc = sub.add_parser("research-cycle")
    rc.add_argument("--state", default=os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research"))

    rh = sub.add_parser("research-health")
    rh.add_argument("--state", default=os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research"))

    pre = sub.add_parser("pretrain-native", help="causal pretraining on reviewed local text")
    pre.add_argument("state", help="existing native checkpoint directory")
    pre.add_argument("corpus", nargs="+", help="local file/directory corpus paths")
    pre.add_argument("--allowed-root", action="append", required=True)
    pre.add_argument("--output", required=True)
    pre.add_argument("--steps", type=int, default=100)
    pre.add_argument("--batch-size", type=int, default=4)
    pre.add_argument("--learning-rate", type=float, default=3e-4)
    pre.add_argument("--max-total-bytes", type=int, default=100_000_000)

    dist = sub.add_parser("distill-transformers", help="distill a local Transformers teacher into SFT JSONL")
    dist.add_argument("teacher_model")
    dist.add_argument("prompts_jsonl")
    dist.add_argument("output_jsonl")
    dist.add_argument("--max-new-tokens", type=int, default=256)

    lora = sub.add_parser("lora-transformers", help="LoRA fine-tune an already-downloaded local causal LM")
    lora.add_argument("base_model")
    lora.add_argument("sft_jsonl")
    lora.add_argument("output_dir")
    lora.add_argument("--steps", type=int, default=100)
    lora.add_argument("--batch-size", type=int, default=1)
    lora.add_argument("--learning-rate", type=float, default=2e-4)
    lora.add_argument("--rank", type=int, default=8)
    lora.add_argument("--alpha", type=int, default=16)
    lora.add_argument("--max-length", type=int, default=1024)

    return p


def _load_distillation_prompts(path: str | Path) -> list[DistillationPrompt]:
    rows: list[DistillationPrompt] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("row is not an object")
                rows.append(DistillationPrompt(
                    domain=str(raw.get("domain", "")),
                    prompt=str(raw.get("prompt", "")),
                    system=str(raw.get("system", "")),
                ))
            except Exception as exc:
                raise ValueError(f"invalid prompt row at line {line_no}: {exc}") from exc
    return rows


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.cmd == "status":
        result = qualification_status(args.state)

    elif args.cmd == "qualify-native":
        result = qualify_checkpoint(args.state, minimum_score=args.minimum_score)

    elif args.cmd == "qualify-transformers":
        result = qualify_transformers_model(
            args.model_dir,
            attestation_path=args.attestation,
            minimum_score=args.minimum_score,
        )

    elif args.cmd == "transformers-status":
        result = transformers_qualification_status(
            args.model_dir,
            attestation_path=args.attestation,
        )

    elif args.cmd == "foundation-init":
        result = write_foundation_manifest(
            args.model_dir,
            FoundationManifest(
                model_id=args.model_id,
                source_revision=args.revision,
                license=args.license,
                architecture=args.architecture,
                context_length=args.context_length,
                parameter_count=args.parameter_count,
                dtype=args.dtype,
                quantization=args.quantization,
            ),
        )

    elif args.cmd == "qualify-foundation":
        max_memory = None
        if args.max_memory_json:
            max_memory = json.loads(args.max_memory_json)
            if not isinstance(max_memory, dict) or not max_memory:
                raise ValueError("--max-memory-json must be a non-empty JSON object")
        result = qualify_foundation_model(
            args.model_dir,
            attestation_path=args.attestation,
            minimum_score=args.minimum_score,
            device=args.device,
            device_map=None if args.device_map == "none" else args.device_map,
            torch_dtype=args.torch_dtype,
            max_memory=max_memory,
            offload_folder=args.offload_folder,
        )

    elif args.cmd == "foundation-status":
        status = foundation_qualification_status(
            args.model_dir,
            attestation_path=args.attestation,
        )
        result = {
            **status,
            "identity": foundation_identity(args.model_dir),
        }

    elif args.cmd == "research-cycle":
        result = run_research_cycle(args.state)

    elif args.cmd == "research-health":
        result = research_health(args.state)

    elif args.cmd == "pretrain-native":
        runtime = GeneralistRuntime.from_checkpoint(args.state)
        corpus = load_local_corpus(
            args.corpus,
            allowed_roots=args.allowed_root,
            max_total_bytes=args.max_total_bytes,
        )
        training = pretrain_causal(
            runtime.model,
            runtime.tokenizer,
            corpus.documents,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=str(runtime.device),
        )
        metadata = runtime.save_checkpoint(
            args.output,
            metadata={
                "learning_stage": "causal-pretraining",
                "source_checkpoint": str(Path(args.state).resolve()),
                "corpus_sha256": training["corpus_sha256"],
            },
        )
        result = {
            "ok": training["ok"],
            "training": training,
            "checkpoint": metadata,
            "output": str(Path(args.output).resolve()),
            "qualification_required": True,
        }

    elif args.cmd == "distill-transformers":
        teacher = LocalTransformersBackend(args.teacher_model, local_files_only=True)
        prompts = _load_distillation_prompts(args.prompts_jsonl)
        examples, report = distill_prompts(
            teacher,
            prompts,
            max_new_tokens=args.max_new_tokens,
        )
        artifact = save_distilled_jsonl(
            args.output_jsonl,
            examples,
            provenance={
                "teacher_model": str(Path(args.teacher_model).resolve()),
                "policy": "local-files-only teacher; outputs require held-out qualification",
            },
        )
        result = {"ok": bool(examples), "distillation": report, "artifact": artifact}

    elif args.cmd == "lora-transformers":
        examples = load_sft_jsonl(args.sft_jsonl)
        cfg = LoRATrainConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            rank=args.rank,
            alpha=args.alpha,
            max_length=args.max_length,
        ).validate()
        result = train_local_lora(
            args.base_model,
            examples,
            args.output_dir,
            config=cfg,
        )
        result["qualification_required"] = True

    else:  # pragma: no cover
        raise AssertionError(args.cmd)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok", result.get("qualified", True)) is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
