from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from generalist_lm.qualification import (
    qualification_status,
    transformers_qualification_status,
)
from generalist_lm.runtime import GeneralistRuntime


def state_dir() -> Path:
    default = Path(os.environ.get("AIRI_ROOT", "/home/user/airi")) / ".ai" / "generalist-lm" / "champion"
    return Path(os.environ.get("AIRI_GENERALIST_STATE", str(default))).expanduser().resolve()


def transformers_model_dir() -> Path | None:
    raw = os.environ.get("AIRI_GENERALIST_TRANSFORMERS_MODEL", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def transformers_attestation_path(model_dir: Path) -> Path:
    raw = os.environ.get("AIRI_GENERALIST_TRANSFORMERS_ATTESTATION", "").strip()
    return Path(raw).expanduser().resolve() if raw else model_dir / ".airi-qualification.json"


def enabled() -> bool:
    return os.environ.get("AIRI_GENERALIST_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}


def status() -> dict[str, Any]:
    torch_ready = importlib.util.find_spec("torch") is not None
    hf_dir = transformers_model_dir()

    if hf_dir is not None:
        transformers_ready = importlib.util.find_spec("transformers") is not None
        attestation = transformers_qualification_status(
            hf_dir,
            attestation_path=transformers_attestation_path(hf_dir),
        )
        qualified = bool(attestation.get("qualified"))
        checkpoint_complete = bool(hf_dir.exists() and hf_dir.is_dir())
        available = bool(
            enabled()
            and checkpoint_complete
            and qualified
            and torch_ready
            and transformers_ready
        )
        return {
            "available": available,
            "enabled": enabled(),
            "qualified": qualified,
            "provider": "airi-generalist",
            "backend": "transformers",
            "model": hf_dir.name if checkpoint_complete else None,
            "state_dir": str(hf_dir),
            "checkpoint_complete": checkpoint_complete,
            "runtime_dependency": bool(torch_ready and transformers_ready),
            "files": {"model_dir": checkpoint_complete, "attestation": transformers_attestation_path(hf_dir).exists()},
            "qualification": attestation,
            "reason": (
                "qualified local Transformers generalist model is enabled"
                if available
                else "provider requires AIRI_GENERALIST_ENABLE=1, torch+transformers, and an exact-digest qualified local model"
            ),
        }

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
    available = bool(enabled() and checkpoint_complete and qualified and torch_ready)
    return {
        "available": available,
        "enabled": enabled(),
        "qualified": qualified,
        "provider": "airi-generalist",
        "backend": "native",
        "model": "local-causal-lm" if checkpoint_complete else None,
        "state_dir": str(root),
        "checkpoint_complete": checkpoint_complete,
        "runtime_dependency": torch_ready,
        "files": files,
        "qualification": qualification,
        "reason": (
            "qualified local generalist checkpoint is enabled"
            if available
            else "provider requires AIRI_GENERALIST_ENABLE=1, PyTorch, and a complete qualified checkpoint"
        ),
    }


def _backend():
    row = status()
    if not row["available"]:
        raise RuntimeError(row["reason"])
    device = os.environ.get("AIRI_GENERALIST_DEVICE", "cpu")
    if row.get("backend") == "transformers":
        from generalist_lm.hf_backend import LocalTransformersBackend
        return LocalTransformersBackend(
            transformers_model_dir(),
            device=device,
            local_files_only=True,
        )
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
    backend = _backend()
    return backend.chat(
        safe_messages,
        max_new_tokens=max(1, min(int(max_new_tokens), 2048)),
    )
