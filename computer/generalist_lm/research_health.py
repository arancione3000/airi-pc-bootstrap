from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evolution import GeneralistGenome
from .model import parameter_count
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

    if genome is not None and runtime is not None:
        cfg = runtime.config
        check("checkpoint:context_match", cfg.context_length == genome.context_length, [cfg.context_length, genome.context_length])
        check("checkpoint:width_match", cfg.d_model == genome.d_model, [cfg.d_model, genome.d_model])
        check("checkpoint:layers_match", cfg.n_layers == genome.n_layers, [cfg.n_layers, genome.n_layers])
        check("checkpoint:heads_match", cfg.n_heads == genome.n_heads, [cfg.n_heads, genome.n_heads])
        check("checkpoint:ff_match", cfg.d_ff == genome.d_ff, [cfg.d_ff, genome.d_ff])

    if isinstance(status, dict) and status:
        report = status.get("champion_report") or {}
        domain_loss = report.get("domain_loss") or {}
        domain_accuracy = report.get("domain_token_accuracy") or {}
        check("report:finite", bool(report.get("finite")), report.get("loss"))
        check("report:domains", bool(domain_loss) and all(float(v) >= 0 for v in domain_loss.values()), domain_loss)
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
        policy = status.get("policy") or {}
        check("policy:research_only", policy.get("research_only") is True, policy)
        check("policy:production_separate", policy.get("production_qualification_separate") is True, policy)

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
