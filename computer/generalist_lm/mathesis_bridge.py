from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def mathesis_signals(state_dir: str | Path) -> dict[str, Any]:
    """Convert VERIFIED MATHESIS state into bounded generalist research hints.

    Mathematical discoveries never become language-model weights or trusted
    language outputs directly. They can only influence curriculum/architecture
    hints that still need independent generalist benchmark promotion.
    """
    root = Path(state_dir)
    discoveries = _read(root / "discoveries.json", {})
    curriculum = _read(root / "curriculum.json", {})
    champion = _read(root / "champion.json", {})

    theorems = discoveries.get("theorems") if isinstance(discoveries, dict) else {}
    if not isinstance(theorems, dict):
        theorems = {}
    verified = [
        row for row in theorems.values()
        if isinstance(row, dict)
        and row.get("verified") is True
        and row.get("quality_gate") == "structurally_nontrivial_and_proof_gated"
        and (row.get("certificate") or {}).get("ok") is True
    ]

    signals: list[str] = []
    if verified:
        signals.append("symbolic_reasoning_signal")
    if int(curriculum.get("cursor", 0) or 0) > 0:
        signals.append("research_curriculum_signal")
    if int(champion.get("symbolic_depth", 0) or 0) >= 8:
        signals.append("deep_symbolic_signal")

    return {
        "ok": True,
        "verified_math_items": len(verified),
        "curriculum_cursor": int(curriculum.get("cursor", 0) or 0),
        "mathesis_generation": int(champion.get("generation", 0) or 0),
        "signals": signals,
        "policy": "verified mathematics may propose curriculum/architecture hints only; generalist promotion remains independently benchmark-gated",
    }
