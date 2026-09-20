from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from generalist_lm.qualification import qualification_status
from generalist_lm.runtime import GeneralistRuntime


def state_dir() -> Path:
    default = Path(os.environ.get("AIRI_ROOT", "/home/user/airi")) / ".ai" / "generalist-lm" / "champion"
    return Path(os.environ.get("AIRI_GENERALIST_STATE", str(default))).expanduser().resolve()


def enabled() -> bool:
    return os.environ.get("AIRI_GENERALIST_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}


def status() -> dict[str, Any]:
    root = state_dir()
    qualification = qualification_status(root)
    files = {
        "config": (root / "config.json").exists(),
        "weights": (root / "model.pt").exists(),
        "metadata": (root / "metadata.json").exists(),
        "benchmark": (root / "benchmark.json").exists(),
    }
    checkpoint_complete = all(files.values())
    qualified = bool(qualification.get("qualified"))
    available = bool(enabled() and checkpoint_complete and qualified)
    return {
        "available": available,
        "enabled": enabled(),
        "qualified": qualified,
        "provider": "airi-generalist",
        "model": "local-causal-lm" if checkpoint_complete else None,
        "state_dir": str(root),
        "checkpoint_complete": checkpoint_complete,
        "files": files,
        "qualification": qualification,
        "reason": (
            "qualified local generalist checkpoint is enabled"
            if available
            else "provider requires AIRI_GENERALIST_ENABLE=1 and a complete qualified checkpoint"
        ),
    }


def _runtime() -> GeneralistRuntime:
    row = status()
    if not row["available"]:
        raise RuntimeError(row["reason"])
    device = os.environ.get("AIRI_GENERALIST_DEVICE", "cpu")
    return GeneralistRuntime.from_checkpoint(state_dir(), device=device)


def chat(messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    safe_messages = []
    for row in messages:
        if not isinstance(row, dict):
            raise ValueError("each message must be an object")
        role = str(row.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {role!r}")
        safe_messages.append({"role": role, "content": str(row.get("content", ""))[:100_000]})
    return _runtime().chat(safe_messages, max_new_tokens=max(1, min(int(max_new_tokens), 2048)))
