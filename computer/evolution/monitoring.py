from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import class_counts, load_records


def detect_drift(
    baseline: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    max_f1_drop: float = 0.08,
    max_brier_increase: float = 0.08,
    max_class_shift: float = 0.20,
    baseline_real_fraction: float | None = None,
    current_real_fraction: float | None = None,
) -> dict[str, Any]:
    reasons = []
    baseline = baseline or {}
    base_f1 = baseline.get("f1")
    cur_f1 = current.get("f1")
    if base_f1 is not None and cur_f1 is not None and float(cur_f1) < float(base_f1) - float(max_f1_drop):
        reasons.append("macro_f1_drop")
    base_brier = baseline.get("brier")
    cur_brier = current.get("brier")
    if base_brier is not None and cur_brier is not None and float(cur_brier) > float(base_brier) + float(max_brier_increase):
        reasons.append("calibration_degradation")
    if baseline_real_fraction is not None and current_real_fraction is not None:
        if abs(float(current_real_fraction) - float(baseline_real_fraction)) > float(max_class_shift):
            reasons.append("class_distribution_shift")
    return {
        "drift": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "max_f1_drop": max_f1_drop,
            "max_brier_increase": max_brier_increase,
            "max_class_shift": max_class_shift,
        },
    }


def drift_report(
    state_dir: Path,
    *,
    window: int = 100,
    min_window: int = 20,
    max_f1_drop: float = 0.08,
    max_brier_increase: float = 0.08,
    max_class_shift: float = 0.20,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    records = load_records(state_dir / "data" / "verified.jsonl")
    if len(records) < max(1, int(min_window)):
        return {
            "ok": True,
            "ready": False,
            "drift": False,
            "reason": "insufficient_recent_verified_samples",
            "records": len(records),
            "required": max(1, int(min_window)),
        }

    try:
        from .engine import evaluate_model, load_champion_for_prediction
        genome, model = load_champion_for_prediction(state_dir)
    except Exception as exc:
        return {"ok": False, "ready": False, "drift": False, "error": repr(exc)}

    recent = sorted(records, key=lambda r: float(r.get("added_at", 0.0)))[-max(1, int(window)):]
    current = evaluate_model(model, genome, recent, 8192, latency_repeats=3)

    baseline = None
    baseline_counts = None
    metrics_path = state_dir / "champion" / "metrics.json"
    prov_path = state_dir / "champion" / "provenance.json"
    try:
        baseline = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        baseline = None
    try:
        baseline_counts = json.loads(prov_path.read_text(encoding="utf-8")).get("class_counts")
    except Exception:
        baseline_counts = None

    recent_counts = class_counts(recent)
    current_real_fraction = recent_counts["real"] / max(1, sum(recent_counts.values()))
    baseline_real_fraction = None
    if isinstance(baseline_counts, dict):
        total = int(baseline_counts.get("real", 0)) + int(baseline_counts.get("fake", 0))
        if total:
            baseline_real_fraction = int(baseline_counts.get("real", 0)) / total

    drift = detect_drift(
        baseline,
        current,
        max_f1_drop=max_f1_drop,
        max_brier_increase=max_brier_increase,
        max_class_shift=max_class_shift,
        baseline_real_fraction=baseline_real_fraction,
        current_real_fraction=current_real_fraction,
    )
    return {
        "ok": True,
        "ready": True,
        **drift,
        "window_records": len(recent),
        "current": current,
        "baseline": baseline,
        "recent_class_counts": recent_counts,
        "baseline_class_counts": baseline_counts,
    }
