from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .runtime import GeneralistRuntime, checkpoint_has_model

_ALLOWED = {"language", "coding", "reasoning", "tools", "efficiency"}


def classify_specialist(text: str, *, capability: str | None = None) -> str:
    explicit = str(capability or "").strip().lower()
    if explicit in _ALLOWED:
        return explicit
    value = str(text or "").casefold()
    if re.search(r"\b(code|python|javascript|typescript|bug|test|repository|function|class)\b", value):
        return "coding"
    if re.search(r"\b(search|web|source|browser|tool|file|project|research|latest)\b", value):
        return "tools"
    if re.search(r"\b(reason|prove|logic|math|calculate|analyse|analyze|data)\b", value):
        return "reasoning"
    if re.search(r"\b(write|translate|conversation|chat|hello|ciao|language|grammar)\b", value):
        return "language"
    return "efficiency"


def specialist_manifest(state_dir: str | Path) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve() / "specialists"
    specialists: dict[str, Any] = {}
    for island in sorted(_ALLOWED):
        summary_path = root / island / "summary.json"
        checkpoint = root / island
        if not summary_path.is_file() or not checkpoint_has_model(checkpoint):
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        if not bool(summary.get("research_only", True)):
            continue
        specialists[island] = {
            **summary,
            "checkpoint": str(checkpoint),
        }
    return {
        "mode": "system_sparse_experts",
        "max_active_experts": 1,
        "specialists": specialists,
    }


def load_research_specialist(
    state_dir: str | Path,
    *,
    prompt: str,
    capability: str | None = None,
    device: str = "cpu",
    research_only: bool = True,
) -> tuple[str, GeneralistRuntime, dict[str, Any]]:
    """Load exactly one persisted research specialist.

    This deliberately refuses production mode. Specialist checkpoints are
    research artifacts and may not bypass the normal qualification path.
    """
    if not research_only:
        raise PermissionError("research specialists cannot bypass production qualification")
    island = classify_specialist(prompt, capability=capability)
    manifest = specialist_manifest(state_dir)
    row = (manifest.get("specialists") or {}).get(island)
    if not isinstance(row, dict):
        raise FileNotFoundError(f"no safe persisted specialist for island {island!r}")
    checkpoint = Path(str(row["checkpoint"])).resolve()
    runtime = GeneralistRuntime.from_checkpoint(checkpoint, device=device)
    return island, runtime, row


def research_chat(
    state_dir: str | Path,
    messages: list[dict[str, str]],
    *,
    capability: str | None = None,
    max_new_tokens: int = 192,
    device: str = "cpu",
) -> dict[str, Any]:
    prompt = " ".join(
        str(row.get("content", ""))
        for row in messages
        if isinstance(row, dict)
    )
    island, runtime, summary = load_research_specialist(
        state_dir,
        prompt=prompt,
        capability=capability,
        device=device,
        research_only=True,
    )
    output = runtime.chat(messages, max_new_tokens=max_new_tokens)
    return {
        "island": island,
        "output": output,
        "candidate_id": summary.get("candidate_id"),
        "research_only": True,
        "max_active_experts": 1,
    }
