from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import lab, state_sync
from .data import class_counts, load_records
from .engine import EvolutionConfig, run_evolution

CLOUD_META = lab.LAB_STATE / "cloud-meta.json"
MAX_CYCLES_PER_DATASET = max(0, int(os.environ.get("AIRI_CLOUD_MAX_CYCLES_PER_DATASET", "0")))


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_cloud_cycle() -> dict[str, Any]:
    lab.ensure_state_boundary()
    audit = state_sync.privacy_audit_dataset(lab.DATA)
    if not audit["ok"]:
        return {"ok": False, "trained": False, "reason": "privacy_audit_failed", "audit": audit}

    records = load_records(lab.DATA)
    counts = class_counts(records)
    digest = state_sync.dataset_digest()
    meta = _read_json(CLOUD_META, {})
    same_dataset = bool(digest and digest == str(meta.get("dataset_digest") or ""))
    attempts = int(meta.get("cycles_on_dataset", 0) or 0) if same_dataset else 0

    base = {
        "feature_schema": lab.FEATURE_SCHEMA,
        "dataset_digest": digest,
        "records": len(records),
        "class_counts": counts,
        "cycles_on_dataset": attempts,
        "max_cycles_per_dataset": MAX_CYCLES_PER_DATASET,
        "continuous_search": MAX_CYCLES_PER_DATASET == 0,
    }

    if len(records) < 40 or min(counts.values()) < 4:
        result = {**base, "ok": True, "trained": False, "reason": "insufficient_balanced_records"}
        _write_json(CLOUD_META, {**result, "updated_at": time.time()})
        return result

    if MAX_CYCLES_PER_DATASET > 0 and attempts >= MAX_CYCLES_PER_DATASET:
        result = {**base, "ok": True, "trained": False, "reason": "dataset_search_budget_exhausted"}
        _write_json(CLOUD_META, {**result, "updated_at": time.time()})
        return result

    cfg = EvolutionConfig.for_mode(
        "safe",
        population=max(4, min(8, int(os.environ.get("AIRI_CLOUD_POPULATION", "4")))),
        generations=max(1, min(3, int(os.environ.get("AIRI_CLOUD_GENERATIONS", "1")))),
        candidate_epochs=1,
        finalist_epochs=max(1, min(3, int(os.environ.get("AIRI_CLOUD_FINALIST_EPOCHS", "2")))),
        vocab_size=4096,
        max_params=500_000,
        max_latency_ms=25.0,
        promotion_repeats=2,
        min_promotion_votes=1,
        min_f1_first_champion=0.55,
        abstain_threshold=0.65,
    )
    evolution = run_evolution(lab.LAB_STATE, cfg)
    attempts += 1

    lab_meta = _read_json(lab.META, {})
    lab_meta.update({
        "feature_schema": lab.FEATURE_SCHEMA,
        "cloud_last_cycle": {
            "run_id": evolution.get("run_id"),
            "promoted": evolution.get("promoted"),
            "promotion_reason": evolution.get("promotion_reason"),
            "candidate": state_sync._clean_public(evolution.get("candidate")),
            "completed_at": time.time(),
        },
        "updated_at": time.time(),
    })
    _write_json(lab.META, lab_meta)

    result = {
        **base,
        "ok": True,
        "trained": True,
        "cycles_on_dataset": attempts,
        "run_id": evolution.get("run_id"),
        "promoted": bool(evolution.get("promoted")),
        "promotion_reason": evolution.get("promotion_reason"),
        "candidate": state_sync._clean_public(evolution.get("candidate")),
        "completed_at": time.time(),
    }
    _write_json(CLOUD_META, {**result, "updated_at": time.time()})
    lab.prune_lab_storage()
    return result


def main() -> int:
    result = run_cloud_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
