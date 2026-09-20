from __future__ import annotations

import importlib.util
import json
import os
import threading
from pathlib import Path
from typing import Any

from generalist_lm.qualification import (
    qualification_status,
    transformers_qualification_status,
)
from generalist_lm.runtime import GeneralistRuntime


_BACKEND_CACHE: dict[str, Any] = {"key": None, "backend": None}
_BACKEND_LOCK = threading.RLock()
_GENERATION_LOCK = threading.RLock()


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
    config_path = root / "config.json"
    tokenizer_version = None
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(raw_config, dict):
            tokenizer_version = str(raw_config.get("tokenizer_version", "byte-v1"))
    except Exception:
        tokenizer_version = None
    files = {
        "config": config_path.exists(),
        "weights": (root / "model.pt").exists(),
        "metadata": (root / "metadata.json").exists(),
        "benchmark": (root / "benchmark.json").exists(),
    }
    if tokenizer_version == "bpe-v1":
        files["tokenizer"] = (root / "tokenizer.json").exists()
    elif tokenizer_version not in {None, "byte-v1"}:
        files["tokenizer_version_supported"] = False
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
    qualification = row.get("qualification") or {}
    identity = (
        row.get("backend"),
        row.get("state_dir"),
        qualification.get("current_checkpoint_digest")
        or qualification.get("current_model_digest")
        or qualification.get("checkpoint_digest")
        or qualification.get("model_digest"),
        device,
    )
    key = repr(identity)
    with _BACKEND_LOCK:
        if _BACKEND_CACHE.get("key") == key and _BACKEND_CACHE.get("backend") is not None:
            return _BACKEND_CACHE["backend"]
        if row.get("backend") == "transformers":
            from generalist_lm.hf_backend import LocalTransformersBackend
            backend = LocalTransformersBackend(
                transformers_model_dir(),
                device=device,
                local_files_only=True,
            )
        else:
            backend = GeneralistRuntime.from_checkpoint(state_dir(), device=device)
        _BACKEND_CACHE["key"] = key
        _BACKEND_CACHE["backend"] = backend
        return backend


def chat(messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if len(messages) > 128:
        raise ValueError("conversation exceeds 128 messages")
    safe_messages = []
    total_chars = 0
    for row in messages:
        if not isinstance(row, dict):
            raise ValueError("each message must be an object")
        role = str(row.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {role!r}")
        content = str(row.get("content", ""))
        if len(content) > 100_000:
            raise ValueError("individual message exceeds 100000 characters")
        total_chars += len(content)
        if total_chars > 250_000:
            raise ValueError("conversation exceeds 250000 characters")
        safe_messages.append({"role": role, "content": content})
    backend = _backend()
    with _GENERATION_LOCK:
        return backend.chat(
            safe_messages,
            max_new_tokens=max(1, min(int(max_new_tokens), 2048)),
        )
