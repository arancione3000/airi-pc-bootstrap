from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmarks import run_benchmark
from .runtime import GeneralistRuntime


def qualify_checkpoint(
    state_dir: str | Path,
    *,
    minimum_score: float = 85.0,
) -> dict[str, Any]:
    root = Path(state_dir)
    runtime = GeneralistRuntime.from_checkpoint(root)
    report = run_benchmark(runtime)
    qualified = bool(
        report.get("ok")
        and float(report.get("score", 0.0)) >= float(minimum_score)
        and not report.get("critical_failures")
    )
    result = {
        "qualified": qualified,
        "minimum_score": float(minimum_score),
        "report": report,
        "policy": "generalist checkpoints become operational only after critical-domain benchmark qualification",
    }
    (root / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def qualification_status(state_dir: str | Path) -> dict[str, Any]:
    path = Path(state_dir) / "benchmark.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {"qualified": False, "reason": "missing_or_invalid_benchmark"}
