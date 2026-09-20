from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _current_verified_math_items(theorems: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], bool]:
    """Re-prove candidate MATHESIS rows with the current verifier.

    The bridge treats persisted theorem metadata as untrusted input. If the
    MATHESIS verifier stack is unavailable, it fails closed and emits no
    mathematical capability signal.
    """
    try:
        from mathesis.discovery import validate_verified_discovery
        from mathesis.verifiers import CompositeVerifier
    except Exception:
        return [], {}, False

    verifier = CompositeVerifier()
    verified: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for theorem_id, row in theorems.items():
        if not isinstance(row, dict):
            rejected[str(theorem_id)] = "row_not_object"
            continue
        if (
            row.get("verified") is not True
            or row.get("quality_gate") != "structurally_nontrivial_and_proof_gated"
            or (row.get("certificate") or {}).get("ok") is not True
        ):
            continue
        try:
            ok, reason = validate_verified_discovery(row, verifier)
        except Exception as exc:
            ok, reason = False, f"reverification_error:{type(exc).__name__}"
        if ok:
            verified.append(row)
        else:
            rejected[str(theorem_id)] = str(reason)
    return verified, rejected, True


def mathesis_signals(state_dir: str | Path) -> dict[str, Any]:
    """Convert currently re-verified MATHESIS state into bounded research hints.

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

    verified, rejected, verifier_available = _current_verified_math_items(theorems)

    def safe_nonnegative_int(value: Any) -> int:
        try:
            parsed = int(value or 0)
        except Exception:
            return 0
        return max(0, parsed)

    curriculum_cursor = safe_nonnegative_int(
        curriculum.get("cursor", 0) if isinstance(curriculum, dict) else 0
    )
    mathesis_generation = safe_nonnegative_int(
        champion.get("generation", 0) if isinstance(champion, dict) else 0
    )
    symbolic_depth = safe_nonnegative_int(
        champion.get("symbolic_depth", 0) if isinstance(champion, dict) else 0
    )

    signals: list[str] = []
    if verified:
        signals.append("symbolic_reasoning_signal")
    if curriculum_cursor > 0:
        signals.append("research_curriculum_signal")
    if symbolic_depth >= 8 and verified:
        signals.append("deep_symbolic_signal")

    return {
        "ok": bool(verifier_available),
        "verifier_available": bool(verifier_available),
        "verified_math_items": len(verified),
        "rejected_math_items": rejected,
        "curriculum_cursor": curriculum_cursor,
        "mathesis_generation": mathesis_generation,
        "signals": signals,
        "policy": (
            "persisted MATHESIS metadata is untrusted; current verifier re-proof is required "
            "before mathematics may influence generalist curriculum/architecture, and "
            "generalist promotion remains independently benchmark-gated"
        ),
    }
