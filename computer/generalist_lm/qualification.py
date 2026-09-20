from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmarks import run_benchmark
from .runtime import GeneralistRuntime

QUALIFICATION_VERSION = 1
_DIGEST_CACHE: dict[tuple, str] = {}


def checkpoint_digest(state_dir: str | Path) -> str:
    root = Path(state_dir)
    files = [root / "config.json", root / "model.pt", root / "metadata.json"]
    for path in files:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"missing checkpoint component: {path.name}")
    key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    cached = _DIGEST_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    value = digest.hexdigest()
    _DIGEST_CACHE.clear()
    _DIGEST_CACHE[key] = value
    return value


def qualify_checkpoint(
    state_dir: str | Path,
    *,
    minimum_score: float = 85.0,
) -> dict[str, Any]:
    root = Path(state_dir)
    runtime = GeneralistRuntime.from_checkpoint(root)
    report = run_benchmark(runtime)
    digest = checkpoint_digest(root)
    qualified = bool(
        report.get("ok")
        and float(report.get("score", 0.0)) >= float(minimum_score)
        and not report.get("critical_failures")
    )
    result = {
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-qualification-v1",
        "checkpoint_digest": digest,
        "qualified": qualified,
        "minimum_score": float(minimum_score),
        "report": report,
        "policy": "qualification is bound to the exact checkpoint digest and critical-domain benchmark result",
    }
    (root / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def qualification_status(state_dir: str | Path) -> dict[str, Any]:
    root = Path(state_dir)
    path = root / "benchmark.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("benchmark attestation must be an object")
        expected = str(value.get("checkpoint_digest", ""))
        current = checkpoint_digest(root)
        integrity_ok = bool(
            expected
            and expected == current
            and int(value.get("qualification_version", 0)) == QUALIFICATION_VERSION
            and value.get("attested_by") == "airi-generalist-qualification-v1"
        )
        return {
            **value,
            "integrity_ok": integrity_ok,
            "qualified": bool(value.get("qualified") and integrity_ok),
            "current_checkpoint_digest": current,
        }
    except Exception as exc:
        return {
            "qualified": False,
            "integrity_ok": False,
            "reason": f"missing_or_invalid_benchmark:{type(exc).__name__}",
        }


_TRANSFORMERS_DIGEST_CACHE: dict[tuple, str] = {}


def transformers_model_digest(model_dir: str | Path, *, exclude_path: str | Path | None = None) -> str:
    root = Path(model_dir).expanduser().resolve()
    excluded = Path(exclude_path).expanduser().resolve() if exclude_path is not None else None
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("transformers model directory does not exist")
    files = [
        path for path in sorted(root.rglob("*"))
        if path.is_file()
        and ".git" not in path.parts
        and path.name not in {".airi-qualification.json"}
        and (excluded is None or path.resolve() != excluded)
    ]
    if not files:
        raise FileNotFoundError("transformers model directory is empty")
    key = tuple((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    cached = _TRANSFORMERS_DIGEST_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    value = digest.hexdigest()
    _TRANSFORMERS_DIGEST_CACHE.clear()
    _TRANSFORMERS_DIGEST_CACHE[key] = value
    return value


def qualify_transformers_model(
    model_dir: str | Path,
    *,
    attestation_path: str | Path | None = None,
    minimum_score: float = 85.0,
) -> dict[str, Any]:
    from .hf_backend import LocalTransformersBackend

    root = Path(model_dir).expanduser().resolve()
    backend = LocalTransformersBackend(root, local_files_only=True)
    report = run_benchmark(backend)
    target = Path(attestation_path or (root / ".airi-qualification.json"))
    digest = transformers_model_digest(root, exclude_path=target)
    qualified = bool(
        report.get("ok")
        and float(report.get("score", 0.0)) >= float(minimum_score)
        and not report.get("critical_failures")
    )
    result = {
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-transformers-qualification-v1",
        "backend_type": "transformers",
        "model_dir": str(root),
        "model_digest": digest,
        "qualified": qualified,
        "minimum_score": float(minimum_score),
        "report": report,
        "policy": "local-files-only, trust_remote_code disabled, exact model digest bound",
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def transformers_qualification_status(
    model_dir: str | Path,
    *,
    attestation_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    target = Path(attestation_path or (root / ".airi-qualification.json"))
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("transformers qualification attestation must be an object")
        current = transformers_model_digest(root, exclude_path=target)
        integrity_ok = bool(
            value.get("backend_type") == "transformers"
            and value.get("attested_by") == "airi-generalist-transformers-qualification-v1"
            and int(value.get("qualification_version", 0)) == QUALIFICATION_VERSION
            and value.get("model_digest") == current
        )
        return {
            **value,
            "integrity_ok": integrity_ok,
            "qualified": bool(value.get("qualified") and integrity_ok),
            "current_model_digest": current,
        }
    except Exception as exc:
        return {
            "qualified": False,
            "integrity_ok": False,
            "reason": f"missing_or_invalid_transformers_attestation:{type(exc).__name__}",
        }
