from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .foundation import load_foundation_manifest


FOUNDATION_PREFLIGHT_VERSION = 1
_CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "n_ctx", "seq_length")
_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_MEMORY_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)$", re.IGNORECASE)
_DECIMAL_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}
_BINARY_UNITS = {
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is invalid JSON: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _safe_model_file(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("weight index contains an empty shard path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("weight index contains an absolute shard path")
    supplied = root / candidate
    if supplied.is_symlink():
        raise ValueError("weight index shard must not be a symlink")
    resolved = supplied.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("weight index shard escapes model directory") from exc
    return resolved


def _context_limit(config: dict[str, Any]) -> int | None:
    values = [
        int(config[key])
        for key in _CONTEXT_KEYS
        if type(config.get(key)) is int and int(config[key]) > 0
    ]
    return min(values) if values else None


def _chat_template_probe(root: Path) -> dict[str, Any]:
    sources: list[str] = []
    external = root / "chat_template.jinja"
    if external.is_file() and external.stat().st_size > 0:
        sources.append("chat_template.jinja")

    json_template = root / "chat_template.json"
    if json_template.is_file() and json_template.stat().st_size > 0:
        sources.append("chat_template.json")

    tokenizer_cfg = root / "tokenizer_config.json"
    embedded = False
    if tokenizer_cfg.is_file():
        try:
            raw = _json_object(tokenizer_cfg, label="tokenizer_config.json")
            template = raw.get("chat_template")
            embedded = bool(
                (isinstance(template, str) and template.strip())
                or (isinstance(template, dict) and template)
                or (isinstance(template, list) and template)
            )
        except ValueError:
            embedded = False
        if embedded:
            sources.append("tokenizer_config.json:chat_template")

    return {
        "available": bool(sources),
        "sources": sorted(set(sources)),
        "embedded": embedded,
    }


def _weight_layout(root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    weight_files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and (
            path.suffix == ".safetensors"
            or (path.name.startswith("pytorch_model") and path.suffix == ".bin")
        )
    )
    index_paths = [root / name for name in _INDEX_FILENAMES if (root / name).is_file()]
    if len(index_paths) > 1:
        blockers.append("multiple_weight_indexes")
        return {
            "index": None,
            "sharded": True,
            "weight_files": [p.relative_to(root).as_posix() for p in weight_files],
            "weight_bytes": sum(p.stat().st_size for p in weight_files),
        }, blockers, warnings

    if not weight_files:
        blockers.append("missing_weight_files")
        return {
            "index": None,
            "sharded": False,
            "weight_files": [],
            "weight_bytes": 0,
        }, blockers, warnings

    if not index_paths:
        if len(weight_files) > 1:
            blockers.append("multiple_weight_files_without_index")
        return {
            "index": None,
            "sharded": len(weight_files) > 1,
            "weight_files": [p.relative_to(root).as_posix() for p in weight_files],
            "weight_bytes": sum(p.stat().st_size for p in weight_files),
        }, blockers, warnings

    index_path = index_paths[0]
    try:
        index = _json_object(index_path, label=index_path.name)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight index weight_map must be a non-empty object")
        if not all(isinstance(key, str) and key for key in weight_map):
            raise ValueError("weight index parameter names must be non-empty strings")
        if not all(isinstance(value, str) and value.strip() for value in weight_map.values()):
            raise ValueError("weight index shard paths must be non-empty strings")

        shard_names = sorted(set(weight_map.values()))
        shard_files: list[Path] = []
        for raw in shard_names:
            shard = _safe_model_file(root, raw)
            if not shard.is_file():
                raise FileNotFoundError(f"weight index references missing shard: {raw}")
            if not (
                shard.suffix == ".safetensors"
                or (shard.name.startswith("pytorch_model") and shard.suffix == ".bin")
            ):
                raise ValueError(f"weight index references unsupported shard type: {raw}")
            shard_files.append(shard)
    except (ValueError, FileNotFoundError) as exc:
        blockers.append(f"invalid_weight_index:{exc}")
        return {
            "index": index_path.name,
            "sharded": True,
            "weight_files": [p.relative_to(root).as_posix() for p in weight_files],
            "weight_bytes": sum(p.stat().st_size for p in weight_files),
        }, blockers, warnings

    indexed_names = {p.relative_to(root).as_posix() for p in shard_files}
    unreferenced = sorted(p.relative_to(root).as_posix() for p in weight_files if p.relative_to(root).as_posix() not in indexed_names)
    if unreferenced:
        warnings.append("unreferenced_weight_files")

    metadata = index.get("metadata") if isinstance(index.get("metadata"), dict) else {}
    return {
        "index": index_path.name,
        "sharded": len(shard_files) > 1,
        "weight_files": [p.relative_to(root).as_posix() for p in shard_files],
        "weight_bytes": sum(p.stat().st_size for p in shard_files),
        "parameter_entries": len(weight_map),
        "index_total_size": metadata.get("total_size"),
        "unreferenced_weight_files": unreferenced,
    }, blockers, warnings


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _system_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        value = pages * page_size
        return value if value > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def _hardware_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "system_memory_bytes": _system_memory_bytes(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "triton": _package_version("triton"),
        "kernels": _package_version("kernels"),
        "cuda_available": False,
        "cuda_devices": [],
        "mps_available": False,
    }
    if importlib.util.find_spec("torch") is None:
        return result
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            devices = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                capability = torch.cuda.get_device_capability(index)
                devices.append({
                    "index": index,
                    "name": str(props.name),
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": [int(capability[0]), int(capability[1])],
                })
            result["cuda_devices"] = devices
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        result["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception as exc:
        result["probe_error"] = type(exc).__name__
    return result


def _memory_to_bytes(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError("memory byte values must be positive")
        return value
    if not isinstance(value, str):
        raise ValueError("memory budget values must be positive integers or unit strings")
    match = _MEMORY_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid memory budget value: {value!r}")
    amount = float(match.group(1))
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("memory budget values must be finite and positive")
    unit = match.group(2).upper()
    multiplier = _DECIMAL_UNITS.get(unit, _BINARY_UNITS.get(unit))
    if multiplier is None:
        raise ValueError(f"unsupported memory unit: {unit}")
    return int(amount * multiplier)


def _memory_budget_probe(max_memory: dict[Any, Any] | None, *, weight_bytes: int) -> dict[str, Any]:
    if max_memory is None:
        return {
            "declared": False,
            "devices": {},
            "total_bytes": None,
            "covers_weight_files": None,
        }
    if not isinstance(max_memory, dict) or not max_memory:
        raise ValueError("max_memory must be a non-empty object")
    devices = {str(key): _memory_to_bytes(value) for key, value in max_memory.items()}
    total = sum(devices.values())
    return {
        "declared": True,
        "devices": devices,
        "total_bytes": total,
        "covers_weight_files": bool(total >= int(weight_bytes)),
    }


def _transformers_config_probe(root: Path) -> dict[str, Any]:
    version = _package_version("transformers")
    if importlib.util.find_spec("transformers") is None:
        return {
            "installed": False,
            "version": version,
            "config_loadable_without_remote_code": False,
            "error": "transformers_not_installed",
        }
    try:
        from transformers import AutoConfig

        loaded = AutoConfig.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
        return {
            "installed": True,
            "version": version,
            "config_loadable_without_remote_code": True,
            "resolved_config_class": type(loaded).__name__,
        }
    except Exception as exc:
        return {
            "installed": True,
            "version": version,
            "config_loadable_without_remote_code": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def foundation_preflight(
    model_dir: str | Path,
    *,
    max_memory: dict[Any, Any] | None = None,
    probe_hardware: bool = True,
) -> dict[str, Any]:
    """Inspect a local Foundation candidate without loading model tensors."""

    supplied = Path(model_dir).expanduser()
    root = supplied.resolve()
    blockers: list[str] = []
    warnings: list[str] = []
    if supplied.is_symlink():
        return {
            "preflight_version": FOUNDATION_PREFLIGHT_VERSION,
            "ok": False,
            "model_dir": str(root),
            "blockers": ["foundation_model_directory_is_symlink"],
            "warnings": [],
            "policy": "metadata-only; no tensor load; local files only",
        }

    try:
        manifest = load_foundation_manifest(root)
    except Exception as exc:
        return {
            "preflight_version": FOUNDATION_PREFLIGHT_VERSION,
            "ok": False,
            "model_dir": str(root),
            "blockers": [f"invalid_foundation_manifest:{type(exc).__name__}:{exc}"],
            "warnings": [],
            "policy": "metadata-only; no tensor load; local files only",
        }

    try:
        config = _json_object(root / "config.json", label="config.json")
    except Exception as exc:
        return {
            "preflight_version": FOUNDATION_PREFLIGHT_VERSION,
            "ok": False,
            "model_dir": str(root),
            "manifest": manifest.to_dict(),
            "blockers": [f"invalid_config:{type(exc).__name__}:{exc}"],
            "warnings": [],
            "policy": "metadata-only; no tensor load; local files only",
        }

    weight_layout, weight_blockers, weight_warnings = _weight_layout(root)
    blockers.extend(weight_blockers)
    warnings.extend(weight_warnings)

    context_limit = _context_limit(config)
    if context_limit is None:
        warnings.append("config_context_limit_unknown")
    elif manifest.context_length > context_limit:
        blockers.append("manifest_context_exceeds_config")

    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        architectures = []
    architectures = [str(x) for x in architectures if isinstance(x, str) and x.strip()]
    model_type = str(config.get("model_type", "")).strip()
    if not model_type:
        blockers.append("config_model_type_missing")

    auto_map = config.get("auto_map")
    if auto_map is not None:
        warnings.append("config_declares_auto_map_remote_code_paths")

    chat_template = _chat_template_probe(root)
    quantization_cfg = config.get("quantization_config")
    quantization_cfg = quantization_cfg if isinstance(quantization_cfg, dict) else {}
    detected_quantization = str(quantization_cfg.get("quant_method", "none")).strip().lower() or "none"
    declared_quantization = manifest.quantization.strip().lower()
    if declared_quantization != detected_quantization:
        blockers.append("manifest_quantization_mismatch")

    protocol_requirements: list[str] = []
    if model_type == "gpt_oss":
        protocol_requirements.append("harmony")
        if not chat_template["available"]:
            blockers.append("gpt_oss_chat_template_missing")
        blockers.append("gpt_oss_harmony_adapter_required")

    if not chat_template["available"]:
        warnings.append("chat_template_not_declared")

    transformers_probe = _transformers_config_probe(root)
    if not transformers_probe.get("config_loadable_without_remote_code"):
        blockers.append("transformers_config_not_loadable_without_remote_code")

    try:
        memory_budget = _memory_budget_probe(
            max_memory,
            weight_bytes=int(weight_layout.get("weight_bytes", 0)),
        )
    except ValueError as exc:
        blockers.append(f"invalid_memory_budget:{exc}")
        memory_budget = {
            "declared": True,
            "devices": {},
            "total_bytes": None,
            "covers_weight_files": False,
        }

    if memory_budget.get("covers_weight_files") is False:
        warnings.append("declared_memory_budget_below_weight_file_bytes")

    hardware = _hardware_probe() if probe_hardware else None
    special_runtime: dict[str, Any] = {}
    if detected_quantization == "mxfp4":
        cuda_devices = (hardware or {}).get("cuda_devices") or []
        capable = any(
            tuple(device.get("compute_capability") or (0, 0)) >= (7, 5)
            for device in cuda_devices
        )
        special_runtime["mxfp4"] = {
            "accelerate_installed": bool(_package_version("accelerate")),
            "triton_installed": bool(_package_version("triton")),
            "kernels_loader_installed": bool(_package_version("kernels")),
            "compatible_cuda_capability_detected": capable if probe_hardware else None,
            "note": (
                "MXFP4 uses specialized kernels; Airi must not fetch unreviewed runtime code autonomously."
            ),
        }
        if probe_hardware and not capable:
            warnings.append("mxfp4_native_cuda_capability_not_detected")
        if not _package_version("triton"):
            warnings.append("mxfp4_triton_not_detected")

    blockers = sorted(set(blockers))
    warnings = sorted(set(warnings))
    return {
        "preflight_version": FOUNDATION_PREFLIGHT_VERSION,
        "ok": not blockers,
        "compatible": not blockers,
        "model_dir": str(root),
        "manifest": manifest.to_dict(),
        "config": {
            "model_type": model_type or None,
            "architectures": architectures,
            "context_limit": context_limit,
            "transformers_version_declared": config.get("transformers_version"),
            "auto_map_declared": auto_map is not None,
        },
        "chat_template": chat_template,
        "quantization": {
            "declared": declared_quantization,
            "detected": detected_quantization,
            "config": quantization_cfg,
        },
        "weights": weight_layout,
        "memory_budget": memory_budget,
        "transformers": transformers_probe,
        "hardware": hardware,
        "protocol_requirements": protocol_requirements,
        "special_runtime": special_runtime,
        "blockers": blockers,
        "warnings": warnings,
        "policy": "metadata-only; no tensor load; no downloads; trust_remote_code disabled",
    }
