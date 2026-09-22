from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .architecture_mutations import next_parameter_tier


META_CONTROLLER_VERSION = "airi-generalist-meta-controller-v2"


def _read(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def decide_next_action(
    state_dir: str | Path,
    *,
    parameter_cap: int = 7_000_000,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    status = _read(root / "status.json", {})
    champion_metrics = _read(root / "champion" / "research-metrics.json", {})
    phase5 = status.get("phase5_bootstrap") or {}
    latest = status.get("latest_research") or {}
    status_champion = status.get("champion_report") or {}
    phase5_after = phase5.get("after") or {}
    canary = (
        status_champion.get("canary")
        or champion_metrics.get("canary")
        or {}
    )

    # status.json is regenerated from the live verifier and can contain newer
    # generation diagnostics than the checkpoint-local research-metrics file.
    # Merge the evidence instead of allowing an older file to hide a collapse.
    parameters = int(
        status_champion.get("parameters")
        or champion_metrics.get("parameters")
        or 0
    )
    pathological = any(bool(value) for value in (
        status_champion.get("generation_pathological_repetition"),
        canary.get("generation_pathological_repetition"),
        champion_metrics.get("generation_pathological_repetition"),
        phase5_after.get("pathological_repetition"),
    ))
    repetition_values = [
        float(value)
        for value in (
            status_champion.get("generation_repetition_rate"),
            canary.get("generation_repetition_rate"),
            champion_metrics.get("generation_repetition_rate"),
            phase5_after.get("repetition_rate"),
        )
        if isinstance(value, (int, float))
    ]
    repetition = max(repetition_values, default=0.0)
    unique_values = [
        float(value)
        for value in (
            status_champion.get("generation_unique_token_ratio"),
            canary.get("generation_unique_token_ratio"),
            champion_metrics.get("generation_unique_token_ratio"),
            phase5_after.get("unique_token_ratio"),
        )
        if isinstance(value, (int, float))
    ]
    unique_token_ratio = min(unique_values, default=1.0)
    repeated_runs = [
        int(value)
        for value in (
            status_champion.get("generation_longest_repeated_token_run"),
            canary.get("generation_longest_repeated_token_run"),
            champion_metrics.get("generation_longest_repeated_token_run"),
            phase5_after.get("longest_repeated_token_run"),
        )
        if isinstance(value, (int, float))
    ]
    longest_repeated_token_run = max(repeated_runs, default=0)

    language_nll = phase5_after.get("language_nll")
    if language_nll is None:
        domains = status_champion.get("domain_nll_per_byte") or {}
        language_nll = domains.get("language")
    minimum_language_success = bool(phase5.get("minimum_success"))
    next_tier = next_parameter_tier(parameters, cap=int(parameter_cap))

    reasons: list[str] = []
    action = "continue_training"

    if (
        pathological
        or repetition >= 0.65
        or longest_repeated_token_run >= 8
        or unique_token_ratio <= 0.20
    ):
        action = "architecture_search"
        reasons.append("autoregressive language collapse remains")
    elif not minimum_language_success and phase5:
        action = "architecture_search"
        reasons.append("language bootstrap minimum success gate not met")
    elif next_tier is not None and parameters < 1_250_000:
        action = "architecture_search"
        reasons.append("champion remains below first serious capacity tier")
    elif latest and not bool(latest.get("all_seed_eligible", False)):
        action = "architecture_search"
        reasons.append("latest research has not passed independent promotion gates")
    else:
        reasons.append("no architecture emergency; continue evidence-driven training")

    if parameters >= int(parameter_cap) and action == "architecture_search":
        reasons.append("parameter cap reached: structural mutations only")

    return {
        "schema": 1,
        "version": META_CONTROLLER_VERSION,
        "action": action,
        "reasons": reasons,
        "observations": {
            "parameters": parameters,
            "parameter_cap": int(parameter_cap),
            "next_parameter_tier": next_tier,
            "pathological_repetition": pathological,
            "repetition_rate": repetition,
            "unique_token_ratio": unique_token_ratio,
            "longest_repeated_token_run": longest_repeated_token_run,
            "language_nll": language_nll,
            "degeneration_evidence": {
                "status_champion": bool(status_champion),
                "checkpoint_metrics": bool(champion_metrics),
                "canary": bool(canary),
                "phase5": bool(phase5_after),
            },
            "minimum_language_success": minimum_language_success,
            "cycle": int(status.get("cycle", 0) or 0),
        },
        "permissions": {
            "may_train_candidates": True,
            "may_mutate_architecture_ir": True,
            "may_scale_parameters": True,
            "may_ingest_unverified_internet": False,
            "may_rewrite_production_code": False,
            "may_weaken_promotion_gates": False,
            "may_access_secrets": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dir")
    parser.add_argument("--parameter-cap", type=int, default=7_000_000)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = decide_next_action(
        args.state_dir,
        parameter_cap=args.parameter_cap,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
