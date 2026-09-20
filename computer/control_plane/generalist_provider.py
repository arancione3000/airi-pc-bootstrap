from __future__ import annotations

import importlib.util
import json
import os
import threading
from pathlib import Path
from typing import Any

from generalist_lm.foundation import FOUNDATION_MANIFEST_FILENAME
from generalist_lm.qualification import (
    foundation_qualification_status,
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


def foundation_model_dir() -> Path | None:
    raw = os.environ.get("AIRI_GENERALIST_FOUNDATION_MODEL", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def foundation_attestation_path(model_dir: Path) -> Path:
    raw = os.environ.get("AIRI_GENERALIST_FOUNDATION_ATTESTATION", "").strip()
    return Path(raw).expanduser().resolve() if raw else model_dir / ".airi-foundation-qualification.json"


def _foundation_runtime_options() -> tuple[dict[str, Any], str | None]:
    raw_map = os.environ.get("AIRI_GENERALIST_FOUNDATION_DEVICE_MAP", "auto").strip().lower()
    if raw_map in {"", "none", "off"}:
        device_map: str | None = None
    elif raw_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
        device_map = raw_map
    else:
        return {}, "invalid AIRI_GENERALIST_FOUNDATION_DEVICE_MAP"

    dtype = os.environ.get("AIRI_GENERALIST_FOUNDATION_DTYPE", "auto").strip().lower()
    if dtype not in {"auto", "float16", "bfloat16", "float32"}:
        return {}, "invalid AIRI_GENERALIST_FOUNDATION_DTYPE"

    max_memory = None
    raw_memory = os.environ.get("AIRI_GENERALIST_FOUNDATION_MAX_MEMORY", "").strip()
    if raw_memory:
        try:
            parsed = json.loads(raw_memory)
        except Exception:
            return {}, "AIRI_GENERALIST_FOUNDATION_MAX_MEMORY must be valid JSON"
        if not isinstance(parsed, dict) or not parsed:
            return {}, "AIRI_GENERALIST_FOUNDATION_MAX_MEMORY must be a non-empty JSON object"
        try:
            from generalist_lm.hf_backend import _normalized_max_memory
            max_memory = _normalized_max_memory(parsed)
        except ValueError as exc:
            return {}, f"invalid AIRI_GENERALIST_FOUNDATION_MAX_MEMORY: {exc}"
        if device_map is None:
            return {}, "Foundation max-memory policy requires a device map"

    offload = os.environ.get("AIRI_GENERALIST_FOUNDATION_OFFLOAD", "").strip()
    if offload and device_map is None:
        return {}, "Foundation offload folder requires a device map"

    return {
        "device_map": device_map,
        "torch_dtype": dtype,
        "max_memory": max_memory,
        "offload_folder": offload or None,
    }, None


def enabled() -> bool:
    return os.environ.get("AIRI_GENERALIST_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}


def status() -> dict[str, Any]:
    torch_ready = importlib.util.find_spec("torch") is not None
    hf_dir = transformers_model_dir()
    foundation_dir = foundation_model_dir()

    if foundation_dir is not None and hf_dir is not None:
        return {
            "available": False,
            "enabled": enabled(),
            "qualified": False,
            "provider": "airi-generalist",
            "backend": "configuration-error",
            "model": None,
            "state_dir": None,
            "checkpoint_complete": False,
            "runtime_dependency": False,
            "files": {},
            "qualification": {},
            "reason": "configure either Foundation or generic Transformers model, never both",
        }

    if foundation_dir is not None:
        transformers_ready = importlib.util.find_spec("transformers") is not None
        accelerate_ready = importlib.util.find_spec("accelerate") is not None
        load_policy, policy_error = _foundation_runtime_options()
        accelerate_required = bool(load_policy.get("device_map")) if policy_error is None else True
        attestation = foundation_qualification_status(
            foundation_dir,
            attestation_path=foundation_attestation_path(foundation_dir),
        )
        qualified = bool(attestation.get("qualified"))
        manifest_path = foundation_dir / FOUNDATION_MANIFEST_FILENAME
        checkpoint_complete = bool(
            foundation_dir.exists()
            and foundation_dir.is_dir()
            and manifest_path.is_file()
        )
        dependencies_ok = bool(
            torch_ready
            and transformers_ready
            and (accelerate_ready or not accelerate_required)
        )
        available = bool(
            enabled()
            and checkpoint_complete
            and qualified
            and dependencies_ok
            and policy_error is None
        )
        return {
            "available": available,
            "enabled": enabled(),
            "qualified": qualified,
            "provider": "airi-generalist",
            "backend": "transformers-foundation",
            "model": foundation_dir.name if checkpoint_complete else None,
            "state_dir": str(foundation_dir),
            "checkpoint_complete": checkpoint_complete,
            "runtime_dependency": dependencies_ok,
            "dependencies": {
                "torch": torch_ready,
                "transformers": transformers_ready,
                "accelerate": accelerate_ready,
                "accelerate_required": accelerate_required,
            },
            "files": {
                "model_dir": bool(foundation_dir.exists() and foundation_dir.is_dir()),
                "manifest": manifest_path.is_file(),
                "attestation": foundation_attestation_path(foundation_dir).exists(),
            },
            "qualification": attestation,
            "load_policy": load_policy,
            "configuration_error": policy_error,
            "reason": (
                "qualified Foundation model is enabled with a valid hardware load policy"
                if available
                else policy_error
                or "provider requires AIRI_GENERALIST_ENABLE=1, torch+transformers, Accelerate when device_map is active, a Foundation manifest, and an exact-digest Foundation qualification"
            ),
        }

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
    load_policy = row.get("load_policy") if isinstance(row.get("load_policy"), dict) else {}
    identity = (
        row.get("backend"),
        row.get("state_dir"),
        qualification.get("current_checkpoint_digest")
        or qualification.get("current_model_digest")
        or qualification.get("checkpoint_digest")
        or qualification.get("model_digest"),
        qualification.get("current_manifest_digest") or qualification.get("manifest_digest"),
        qualification.get("current_suite_digest") or qualification.get("suite_digest"),
        device,
        json.dumps(load_policy, sort_keys=True, default=str),
    )
    key = repr(identity)
    with _BACKEND_LOCK:
        if _BACKEND_CACHE.get("key") == key and _BACKEND_CACHE.get("backend") is not None:
            return _BACKEND_CACHE["backend"]
        if row.get("backend") in {"transformers", "transformers-foundation"}:
            from generalist_lm.hf_backend import LocalTransformersBackend
            model_dir = (
                foundation_model_dir()
                if row.get("backend") == "transformers-foundation"
                else transformers_model_dir()
            )
            backend = LocalTransformersBackend(
                model_dir,
                device=device,
                local_files_only=True,
                **load_policy,
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
