from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .architecture_mutations import next_parameter_tier


META_CONTROLLER_VERSION = "airi-generalist-meta-controller-v1"


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

    parameters = int(
        champion_metrics.get("parameters")
        or (status.get("champion_report") or {}).get("parameters")
        or 0
    )
    pathological = bool(
        ((phase5.get("after") or {}).get("pathological_repetition"))
        or champion_metrics.get("generation_pathological_repetition")
    )
    repetition = float(
        ((phase5.get("after") or {}).get("repetition_rate"))
        or champion_metrics.get("generation_repetition_rate")
        or 0.0
    )
    language_nll = (
        (phase5.get("after") or {}).get("language_nll")
        if isinstance(phase5, dict)
        else None
    )
    minimum_language_success = bool(phase5.get("minimum_success"))
    next_tier = next_parameter_tier(parameters, cap=int(parameter_cap))

    reasons: list[str] = []
    action = "continue_training"

    if pathological or repetition >= 0.65:
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
            "language_nll": language_nll,
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
