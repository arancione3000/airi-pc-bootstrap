from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .distillation import DistillationPrompt, distill_prompts, save_distilled_jsonl
from .hf_backend import LocalTransformersBackend
from .foundation import FoundationManifest, foundation_identity, write_foundation_manifest
from .foundation_probe import foundation_preflight
from .native_foundation import (
    NativeFoundationConfig,
    create_native_root_checkpoint,
    native_checkpoint_status,
    native_parameter_count,
    native_scale_profile,
)
from .native_acquisition import acquire_native_corpus
from .native_evaluation import evaluate_native_checkpoint
from .native_evolution_cycle import run_native_evolution_cycle
from .native_online_research import discover_native_research
from .native_data import (
    NATIVE_CORPUS_DOMAINS,
    audit_native_corpus,
    load_native_corpus,
    train_native_bpe,
)
from .native_training import (
    NativeTrainConfig,
    native_training_status,
    train_native_foundation,
)
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

    fp = sub.add_parser(
        "foundation-preflight",
        help="inspect a local Foundation candidate without loading model tensors",
    )
    fp.add_argument("model_dir")
    fp.add_argument("--max-memory-json")
    fp.add_argument("--no-hardware", action="store_true")

    nfp = sub.add_parser(
        "native-foundation-plan",
        help="plan an AIRI scratch Foundation scale without allocating model weights",
    )
    nfp.add_argument("profile", choices=("micro", "1b", "3b", "7b"))
    nfp.add_argument("--vocab-size", type=int, default=32768)

    nfi = sub.add_parser(
        "native-foundation-init",
        help="create an AIRI Native Foundation root checkpoint from random init only",
    )
    nfi.add_argument("output")
    nfi.add_argument("--profile", choices=("micro", "1b", "3b", "7b"), default="micro")
    nfi.add_argument("--vocab-size", type=int, default=32768)
    nfi.add_argument("--config-json")
    nfi.add_argument("--tokenizer-json")
    nfi.add_argument("--seed", type=int, default=17)
    nfi.add_argument("--max-init-parameters", type=int, default=100_000_000)
    nfi.add_argument("--allow-large-init", action="store_true")

    nfs = sub.add_parser("native-foundation-status")
    nfs.add_argument("state")

    nac = sub.add_parser(
        "native-corpus-acquire",
        help="download a pinned approved source catalog into an AIRI Native corpus",
    )
    nac.add_argument("catalog")
    nac.add_argument("output")
    nac.add_argument("--allowed-host", action="append")
    nac.add_argument("--max-file-bytes", type=int, default=250_000_000)
    nac.add_argument("--max-total-bytes", type=int, default=5_000_000_000)
    nac.add_argument("--timeout-seconds", type=float, default=60.0)

    nca = sub.add_parser(
        "native-corpus-audit",
        help="audit a local provenance-governed AIRI Native corpus manifest",
    )
    nca.add_argument("manifest")
    nca.add_argument("--allowed-root", action="append", required=True)
    nca.add_argument("--require-domain", action="append")
    nca.add_argument("--max-total-bytes", type=int, default=2_000_000_000)

    ntt = sub.add_parser(
        "native-tokenizer-train",
        help="train AIRI's own BPE tokenizer from the approved Native corpus",
    )
    ntt.add_argument("manifest")
    ntt.add_argument("output")
    ntt.add_argument("--allowed-root", action="append", required=True)
    ntt.add_argument("--vocab-size", type=int, default=32768)
    ntt.add_argument("--min-frequency", type=int, default=2)
    ntt.add_argument("--max-bytes", type=int, default=500_000_000)
    ntt.add_argument("--max-total-bytes", type=int, default=2_000_000_000)

    nt = sub.add_parser(
        "native-train",
        help="pretrain AIRI Native Foundation from its own checkpoint and corpus",
    )
    nt.add_argument("state")
    nt.add_argument("manifest")
    nt.add_argument("--allowed-root", action="append", required=True)
    nt.add_argument("--output", required=True)
    nt.add_argument("--max-steps", type=int, default=1000)
    nt.add_argument("--micro-batch-size", type=int, default=2)
    nt.add_argument("--gradient-accumulation-steps", type=int, default=8)
    nt.add_argument("--learning-rate", type=float, default=3e-4)
    nt.add_argument("--min-learning-rate", type=float, default=3e-5)
    nt.add_argument("--warmup-steps", type=int, default=100)
    nt.add_argument("--weight-decay", type=float, default=0.1)
    nt.add_argument("--grad-clip", type=float, default=1.0)
    nt.add_argument("--validation-fraction", type=float, default=0.05)
    nt.add_argument("--max-eval-blocks", type=int, default=128)
    nt.add_argument("--seed", type=int, default=17)
    nt.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    nt.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    nt.add_argument("--domain-weights-json")
    nt.add_argument("--max-total-bytes", type=int, default=2_000_000_000)
    nt.add_argument("--no-optimizer-state", action="store_true")

    nts = sub.add_parser("native-training-status")
    nts.add_argument("state")

    nro = sub.add_parser(
        "native-research-online",
        help="search bounded public research metadata for Native evolution signals",
    )
    nro.add_argument("--signal", action="append")
    nro.add_argument("--max-evidence", type=int, default=24)

    nev = sub.add_parser(
        "native-evaluate",
        help="independently evaluate a Native checkpoint on deterministic held-out corpus data",
    )
    nev.add_argument("state")
    nev.add_argument("manifest")
    nev.add_argument("--allowed-root", action="append", required=True)
    nev.add_argument("--validation-fraction", type=float, default=0.2)
    nev.add_argument("--seed", type=int, default=431)
    nev.add_argument("--max-eval-blocks", type=int, default=64)
    nev.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    nevo = sub.add_parser(
        "native-evolve",
        help="run one proof-gated AIRI Native evolution cycle",
    )
    nevo.add_argument("evolution_state")
    nevo.add_argument("seed_checkpoint")
    nevo.add_argument("manifest")
    nevo.add_argument("--allowed-root", action="append", required=True)
    nevo.add_argument("--mathesis-state")
    nevo.add_argument("--offline", action="store_true")
    nevo.add_argument("--challengers", type=int, default=3)
    nevo.add_argument("--trial-steps", type=int, default=4)
    nevo.add_argument("--reinit-steps", type=int, default=0)
    nevo.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    nevo.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    nevo.add_argument("--minimum-gain", type=float, default=0.002)
    nevo.add_argument("--max-domain-regression", type=float, default=0.05)
    nevo.add_argument("--max-parameter-ratio", type=float, default=1.5)

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

    elif args.cmd == "foundation-preflight":
        max_memory = None
        if args.max_memory_json:
            max_memory = json.loads(args.max_memory_json)
            if not isinstance(max_memory, dict) or not max_memory:
                raise ValueError("--max-memory-json must be a non-empty JSON object")
        result = foundation_preflight(
            args.model_dir,
            max_memory=max_memory,
            probe_hardware=not args.no_hardware,
        )

    elif args.cmd == "native-foundation-plan":
        cfg = native_scale_profile(args.profile, vocab_size=args.vocab_size)
        result = {
            "ok": True,
            "family": "airi-native-foundation",
            "profile": args.profile,
            "config": cfg.to_dict(),
            "parameters": native_parameter_count(cfg),
            "weights_origin": "random-init",
            "external_pretrained": False,
            "allocation_performed": False,
        }

    elif args.cmd == "native-foundation-init":
        if args.config_json:
            raw = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
            cfg = NativeFoundationConfig.from_dict(raw)
        else:
            cfg = native_scale_profile(args.profile, vocab_size=args.vocab_size)
        parameters = native_parameter_count(cfg)
        if parameters > int(args.max_init_parameters) and not args.allow_large_init:
            raise ValueError(
                f"native init would allocate {parameters} parameters; "
                "raise --max-init-parameters or pass --allow-large-init explicitly"
            )
        result = create_native_root_checkpoint(
            args.output,
            cfg,
            root_seed=args.seed,
            tokenizer_path=args.tokenizer_json,
        )

    elif args.cmd == "native-foundation-status":
        result = native_checkpoint_status(args.state)

    elif args.cmd == "native-corpus-acquire":
        result = acquire_native_corpus(
            args.catalog,
            args.output,
            allowed_hosts=args.allowed_host,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
            timeout_seconds=args.timeout_seconds,
        )

    elif args.cmd == "native-corpus-audit":
        corpus = load_native_corpus(
            args.manifest,
            allowed_roots=args.allowed_root,
            max_total_bytes=args.max_total_bytes,
        )
        result = audit_native_corpus(
            corpus,
            required_domains=args.require_domain or NATIVE_CORPUS_DOMAINS,
        )

    elif args.cmd == "native-tokenizer-train":
        corpus = load_native_corpus(
            args.manifest,
            allowed_roots=args.allowed_root,
            max_total_bytes=args.max_total_bytes,
        )
        result = train_native_bpe(
            corpus,
            args.output,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            max_bytes=args.max_bytes,
        )

    elif args.cmd == "native-train":
        domain_weights = None
        if args.domain_weights_json:
            domain_weights = json.loads(args.domain_weights_json)
            if not isinstance(domain_weights, dict):
                raise ValueError("--domain-weights-json must be a JSON object")
        train_config = NativeTrainConfig(
            max_steps=args.max_steps,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            validation_fraction=args.validation_fraction,
            max_eval_blocks=args.max_eval_blocks,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            save_optimizer_state=not args.no_optimizer_state,
            domain_weights=domain_weights,
        ).validate()
        result = train_native_foundation(
            args.state,
            args.manifest,
            allowed_roots=args.allowed_root,
            output_dir=args.output,
            config=train_config,
            max_total_bytes=args.max_total_bytes,
        )

    elif args.cmd == "native-training-status":
        result = native_training_status(args.state)

    elif args.cmd == "native-research-online":
        result = discover_native_research(
            signals=args.signal,
            max_evidence=max(1, int(args.max_evidence)),
        )

    elif args.cmd == "native-evaluate":
        result = evaluate_native_checkpoint(
            args.state,
            args.manifest,
            allowed_roots=args.allowed_root,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            max_eval_blocks=args.max_eval_blocks,
            device=args.device,
        )

    elif args.cmd == "native-evolve":
        result = run_native_evolution_cycle(
            args.evolution_state,
            args.seed_checkpoint,
            args.manifest,
            allowed_roots=args.allowed_root,
            mathesis_state_dir=args.mathesis_state,
            online_research=not args.offline,
            challenger_count=max(1, int(args.challengers)),
            steps_per_trial=max(1, int(args.trial_steps)),
            reinit_training_steps=max(0, int(args.reinit_steps)),
            device=args.device,
            precision=args.precision,
            minimum_gain=max(0.0, float(args.minimum_gain)),
            max_domain_regression=max(0.0, float(args.max_domain_regression)),
            max_parameter_ratio=max(1.0, float(args.max_parameter_ratio)),
        )

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
