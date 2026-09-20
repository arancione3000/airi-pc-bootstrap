from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .curriculum import DOMAINS, curriculum_manifest
from .curriculum_memory import CurriculumMemory
from .evolution import GeneralistGenome
from .model import parameter_count
from .qualification import qualification_status
from .runtime import GeneralistRuntime


def research_health(state_dir: str | Path) -> dict[str, Any]:
    root = Path(state_dir)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    status = {}
    genome = None
    runtime = None

    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        check("status:loadable", isinstance(status, dict), type(status).__name__)
    except Exception as exc:
        check("status:loadable", False, repr(exc))

    try:
        raw = json.loads((root / "champion-genome.json").read_text(encoding="utf-8"))
        genome = GeneralistGenome(**raw).validate()
        check("genome:loadable", True, genome.genome_id)
    except Exception as exc:
        check("genome:loadable", False, repr(exc))

    try:
        runtime = GeneralistRuntime.from_checkpoint(root / "champion")
        check("checkpoint:loadable", True, parameter_count(runtime.model))
    except Exception as exc:
        check("checkpoint:loadable", False, repr(exc))

    model_path = root / "champion" / "model.pt"
    max_persisted_bytes = max(
        65_536,
        int(os.environ.get("AIRI_GENERALIST_MAX_PERSISTED_CHECKPOINT_BYTES", str(32 * 1024 * 1024))),
    )
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    check(
        "checkpoint:persistence_size",
        bool(model_bytes and model_bytes <= max_persisted_bytes),
        {"bytes": model_bytes, "max_bytes": max_persisted_bytes},
    )

    if genome is not None and runtime is not None:
        cfg = runtime.config
        check("checkpoint:context_match", cfg.context_length == genome.context_length, [cfg.context_length, genome.context_length])
        check("checkpoint:width_match", cfg.d_model == genome.d_model, [cfg.d_model, genome.d_model])
        check("checkpoint:layers_match", cfg.n_layers == genome.n_layers, [cfg.n_layers, genome.n_layers])
        check("checkpoint:heads_match", cfg.n_heads == genome.n_heads, [cfg.n_heads, genome.n_heads])
        check("checkpoint:ff_match", cfg.d_ff == genome.d_ff, [cfg.d_ff, genome.d_ff])
        check("checkpoint:norm_match", cfg.norm_type == genome.norm_type, [cfg.norm_type, genome.norm_type])
        check(
            "checkpoint:position_match",
            cfg.position_encoding == genome.position_encoding,
            [cfg.position_encoding, genome.position_encoding],
        )
        check("checkpoint:ff_variant_match", cfg.ff_variant == genome.ff_variant, [cfg.ff_variant, genome.ff_variant])
        check(
            "checkpoint:tokenizer_match",
            cfg.tokenizer_version == genome.tokenizer_version == runtime.tokenizer.version,
            [cfg.tokenizer_version, genome.tokenizer_version, runtime.tokenizer.version],
        )

    production = root / "production"
    if production.exists():
        production_status = qualification_status(production)
        check(
            "production:qualified_integrity",
            bool(production_status.get("qualified") and production_status.get("integrity_ok")),
            production_status,
        )

    curriculum = curriculum_manifest()
    check("curriculum:no_prompt_overlap", curriculum.get("prompt_overlap") == [], curriculum.get("prompt_overlap"))
    check(
        "curriculum:all_domains_train",
        set((curriculum.get("train_domains") or {})) == set(DOMAINS)
        and all(int(value) >= 1 for value in (curriculum.get("train_domains") or {}).values()),
        curriculum.get("train_domains"),
    )
    check(
        "curriculum:all_domains_validation",
        set((curriculum.get("validation_domains") or {})) == set(DOMAINS)
        and all(int(value) >= 1 for value in (curriculum.get("validation_domains") or {}).values()),
        curriculum.get("validation_domains"),
    )

    try:
        memory = CurriculumMemory(root)
        memory_manifest = memory.manifest()
        check(
            "curriculum_memory:bounded",
            int(memory_manifest.get("stored", 0)) <= int(memory_manifest.get("max_rows", 0)),
            memory_manifest,
        )
        check(
            "curriculum_memory:no_validation_overlap",
            memory_manifest.get("validation_overlap") == [],
            memory_manifest.get("validation_overlap"),
        )
        memory_domains = memory_manifest.get("domains") or {}
        if int(memory_manifest.get("stored", 0)) > 0:
            check(
                "curriculum_memory:all_domains",
                set(memory_domains) == set(DOMAINS)
                and all(int(memory_domains.get(domain, 0)) >= 1 for domain in DOMAINS),
                memory_domains,
            )
    except Exception as exc:
        check("curriculum_memory:loadable", False, repr(exc))

    if isinstance(status, dict) and status:
        report = status.get("champion_report") or {}
        domain_loss = report.get("domain_loss") or {}
        domain_accuracy = report.get("domain_token_accuracy") or {}
        check("report:finite", bool(report.get("finite")), report.get("loss"))
        check("report:domains", bool(domain_loss) and all(float(v) >= 0 for v in domain_loss.values()), domain_loss)
        nll_per_byte = report.get("nll_per_byte")
        domain_nll = report.get("domain_nll_per_byte") or {}
        check(
            "report:nll_per_byte",
            isinstance(nll_per_byte, (int, float)) and float(nll_per_byte) >= 0.0,
            nll_per_byte,
        )
        check(
            "report:domain_nll_per_byte",
            set(domain_nll) == set(DOMAINS)
            and all(isinstance(v, (int, float)) and float(v) >= 0.0 for v in domain_nll.values()),
            domain_nll,
        )
        check(
            "report:tokenizer_version",
            report.get("tokenizer_version") in {"byte-v1", "bpe-v1"},
            report.get("tokenizer_version"),
        )
        accuracy = report.get("target_token_accuracy")
        check(
            "report:target_token_accuracy",
            isinstance(accuracy, (int, float)) and 0.0 <= float(accuracy) <= 1.0,
            accuracy,
        )
        check(
            "report:domain_token_accuracy",
            bool(domain_accuracy)
            and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in domain_accuracy.values()),
            domain_accuracy,
        )
        solved = report.get("solved_items")
        check("report:solved_items", isinstance(solved, list), len(solved) if isinstance(solved, list) else type(solved).__name__)

        generation_accuracy = report.get("generation_exact_accuracy")
        check(
            "report:generation_exact_accuracy",
            isinstance(generation_accuracy, (int, float))
            and 0.0 <= float(generation_accuracy) <= 1.0,
            generation_accuracy,
        )
        generation_domains = report.get("domain_generation_accuracy")
        check(
            "report:domain_generation_accuracy",
            isinstance(generation_domains, dict)
            and set(generation_domains) == set(DOMAINS)
            and all(
                isinstance(value, (int, float))
                and 0.0 <= float(value) <= 1.0
                for value in generation_domains.values()
            ),
            generation_domains,
        )
        generated_solved = report.get("generated_solved_items")
        check(
            "report:generated_solved_items",
            isinstance(generated_solved, list),
            len(generated_solved) if isinstance(generated_solved, list) else type(generated_solved).__name__,
        )
        generation_probe = report.get("generation_probe")
        check(
            "report:generation_probe",
            isinstance(generation_probe, list)
            and len(generation_probe) == len(DOMAINS)
            and {row.get("domain") for row in generation_probe if isinstance(row, dict)} == set(DOMAINS),
            generation_probe,
        )

        canary_cycle = report.get("canary_cycle")
        canary = report.get("canary")
        check(
            "report:canary_cycle",
            isinstance(canary_cycle, int) and canary_cycle >= 1,
            canary_cycle,
        )
        check("report:canary_object", isinstance(canary, dict), type(canary).__name__)
        if isinstance(canary, dict):
            check("report:canary_finite", canary.get("finite") is True, canary.get("loss"))
            check(
                "report:canary_nll_per_byte",
                isinstance(canary.get("nll_per_byte"), (int, float))
                and float(canary.get("nll_per_byte")) >= 0.0,
                canary.get("nll_per_byte"),
            )
            canary_domain_nll = canary.get("domain_nll_per_byte") or {}
            check(
                "report:canary_domain_nll_per_byte",
                set(canary_domain_nll) == set(DOMAINS)
                and all(
                    isinstance(value, (int, float)) and float(value) >= 0.0
                    for value in canary_domain_nll.values()
                ),
                canary_domain_nll,
            )
            canary_domains = canary.get("domain_loss") or {}
            check(
                "report:canary_domains",
                set(canary_domains) == set(DOMAINS)
                and all(isinstance(value, (int, float)) and float(value) >= 0.0 for value in canary_domains.values()),
                canary_domains,
            )
            canary_generation = canary.get("domain_generation_accuracy") or {}
            check(
                "report:canary_generation_domains",
                set(canary_generation) == set(DOMAINS)
                and all(
                    isinstance(value, (int, float))
                    and 0.0 <= float(value) <= 1.0
                    for value in canary_generation.values()
                ),
                canary_generation,
            )

        rotating = status.get("rotating_canary") or {}
        check(
            "canary:training_disjoint",
            rotating.get("training_overlap") == [],
            rotating.get("training_overlap"),
        )
        check(
            "canary:domains",
            set(rotating.get("domains") or []) == set(DOMAINS),
            rotating.get("domains"),
        )

        policy = status.get("policy") or {}
        check("policy:research_only", policy.get("research_only") is True, policy)
        check("policy:production_separate", policy.get("production_qualification_separate") is True, policy)
        continual = policy.get("continual_learning") or {}
        check(
            "policy:continual_full_replay",
            continual.get("enabled") is True and continual.get("full_replay") is True,
            continual,
        )
        tokenizer_policy = policy.get("tokenizer_research") or {}
        check(
            "policy:tokenizer_research",
            tokenizer_policy.get("comparison_metric") == "nll_per_byte"
            and tokenizer_policy.get("protected_eval_rows_excluded_from_tokenizer_training") is True
            and set(tokenizer_policy.get("allowed") or []) == {"byte-v1", "bpe-v1"},
            tokenizer_policy,
        )
        rotating_policy = policy.get("rotating_canary") or {}
        check(
            "policy:rotating_canary",
            rotating_policy.get("enabled") is True
            and rotating_policy.get("training_excluded") is True
            and int(rotating_policy.get("domains", 0) or 0) == len(DOMAINS),
            rotating_policy,
        )

    failed = [row for row in checks if not row["ok"]]
    return {"ok": not failed, "checks": checks, "failed": failed}


def main() -> int:
    import os
    root = os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research")
    report = research_health(root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
