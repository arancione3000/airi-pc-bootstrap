from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmarks import qualification_suite, run_benchmark
from .foundation import (
    FOUNDATION_DOMAINS,
    FOUNDATION_MANIFEST_FILENAME,
    foundation_manifest_digest,
    load_foundation_manifest,
)
from .foundation_benchmarks import (
    FOUNDATION_SUITE_VERSION,
    foundation_suite,
    foundation_suite_digest,
)
from .foundation_probe import FOUNDATION_PREFLIGHT_VERSION, foundation_preflight
from .runtime import GeneralistRuntime, checkpoint_model_files

QUALIFICATION_VERSION = 2
FOUNDATION_QUALIFICATION_VERSION = 4
FOUNDATION_MINIMUM_SCORE = 90.0
_DIGEST_CACHE: dict[tuple, str] = {}


def _checkpoint_files(root: Path) -> list[Path]:
    config_path = root / "config.json"
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError("missing checkpoint component: config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a JSON object")
    tokenizer_version = str(config.get("tokenizer_version", "byte-v1"))
    files = [config_path, *checkpoint_model_files(root), root / "metadata.json"]
    if tokenizer_version == "bpe-v1":
        files.append(root / "tokenizer.json")
    elif tokenizer_version != "byte-v1":
        raise ValueError(f"unsupported checkpoint tokenizer version: {tokenizer_version}")
    return files


def checkpoint_digest(state_dir: str | Path) -> str:
    root = Path(state_dir)
    files = _checkpoint_files(root)
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
    report = run_benchmark(runtime, qualification_suite())
    digest = checkpoint_digest(root)
    qualified = bool(
        report.get("ok")
        and float(report.get("score", 0.0)) >= float(minimum_score)
        and not report.get("critical_failures")
    )
    result = {
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-qualification-v2",
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
            and value.get("attested_by") == "airi-generalist-qualification-v2"
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
        and path.name not in {".airi-qualification.json", ".airi-foundation-qualification.json"}
        and (excluded is None or path.resolve() != excluded)
    ]
    if not files:
        raise FileNotFoundError("transformers model directory is empty")
    key = tuple(
        (
            str(p.relative_to(root)),
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        for p in files
    )
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
    if (root / FOUNDATION_MANIFEST_FILENAME).exists():
        raise ValueError(
            "foundation manifest present; use qualify_foundation_model instead of generic Transformers qualification"
        )
    backend = LocalTransformersBackend(root, local_files_only=True)
    report = run_benchmark(backend, qualification_suite())
    target = Path(attestation_path or (root / ".airi-qualification.json"))
    digest = transformers_model_digest(root, exclude_path=target)
    qualified = bool(
        report.get("ok")
        and float(report.get("score", 0.0)) >= float(minimum_score)
        and not report.get("critical_failures")
    )
    result = {
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-transformers-qualification-v2",
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
    if (root / FOUNDATION_MANIFEST_FILENAME).exists():
        return {
            "qualified": False,
            "integrity_ok": False,
            "reason": "foundation_model_requires_foundation_qualification",
        }
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("transformers qualification attestation must be an object")
        current = transformers_model_digest(root, exclude_path=target)
        integrity_ok = bool(
            value.get("backend_type") == "transformers"
            and value.get("attested_by") == "airi-generalist-transformers-qualification-v2"
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


def qualify_foundation_model(
    model_dir: str | Path,
    *,
    attestation_path: str | Path | None = None,
    minimum_score: float = 90.0,
    device: str = "cpu",
    device_map: str | dict[str, Any] | None = None,
    torch_dtype: str | None = None,
    max_memory: dict[Any, Any] | None = None,
    offload_folder: str | Path | None = None,
) -> dict[str, Any]:
    """Qualify a first-class open-weight foundation candidate.

    Foundation candidates are stricter than the generic Transformers adapter:
    they require a reviewed local manifest and the dedicated harder protected
    suite. The attestation is bound to the exact model tree, manifest, suite and
    inference precision profile.
    """
    threshold = float(minimum_score)
    if not math.isfinite(threshold) or threshold < FOUNDATION_MINIMUM_SCORE or threshold > 100.0:
        raise ValueError(
            f"foundation minimum_score must be between {FOUNDATION_MINIMUM_SCORE:.1f} and 100.0"
        )
    canonical_dtype = str(torch_dtype or "auto").strip().lower()
    if canonical_dtype not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError("unsupported Foundation inference dtype")
    from .hf_backend import LocalTransformersBackend

    supplied = Path(model_dir).expanduser()
    preflight = foundation_preflight(
        supplied,
        max_memory=max_memory,
        probe_hardware=False,
    )
    root = supplied.resolve()
    if not preflight.get("ok"):
        blockers = preflight.get("blockers") or ["unknown_preflight_failure"]
        raise ValueError("foundation preflight failed: " + "; ".join(str(x) for x in blockers))
    manifest = load_foundation_manifest(root)
    manifest_digest = foundation_manifest_digest(root)
    suite_digest = foundation_suite_digest()
    backend = LocalTransformersBackend(
        root,
        device=device,
        device_map=device_map,
        torch_dtype=canonical_dtype,
        max_memory=max_memory,
        offload_folder=offload_folder,
        local_files_only=True,
    )
    report = run_benchmark(backend, foundation_suite())
    target = Path(attestation_path or (root / ".airi-foundation-qualification.json"))
    digest = transformers_model_digest(root, exclude_path=target)
    qualified = bool(
        report.get("ok")
        and float(report.get("score", 0.0)) >= threshold
        and not report.get("critical_failures")
    )
    result = {
        "foundation_qualification_version": FOUNDATION_QUALIFICATION_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-foundation-qualification-v4",
        "backend_type": "transformers-foundation",
        "model_digest": digest,
        "manifest_digest": manifest_digest,
        "suite_version": FOUNDATION_SUITE_VERSION,
        "suite_digest": suite_digest,
        "qualified": qualified,
        "minimum_score": threshold,
        "manifest": manifest.to_dict(),
        "preflight": preflight,
        "report": report,
        "inference_profile": {
            "torch_dtype": canonical_dtype,
        },
        "load_policy": {
            "device": str(device),
            "device_map": device_map,
            "torch_dtype": canonical_dtype,
            "max_memory": (
                {str(key): value for key, value in max_memory.items()}
                if isinstance(max_memory, dict)
                else None
            ),
            "offload_folder": str(Path(offload_folder).expanduser().resolve()) if offload_folder else None,
        },
        "policy": (
            "reviewed local manifest + metadata-only Foundation preflight + dedicated protected suite; "
            "local-files-only; trust_remote_code disabled; exact digests + inference dtype bound"
        ),
    }
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def foundation_qualification_status(
    model_dir: str | Path,
    *,
    attestation_path: str | Path | None = None,
) -> dict[str, Any]:
    supplied = Path(model_dir).expanduser()
    if supplied.is_symlink():
        return {
            "qualified": False,
            "integrity_ok": False,
            "qualification_semantics_ok": False,
            "preflight_ok": False,
            "reason": "foundation_model_directory_is_symlink",
        }
    root = supplied.resolve()
    target = Path(attestation_path or (root / ".airi-foundation-qualification.json"))
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("foundation qualification attestation must be an object")
        current_model = transformers_model_digest(root, exclude_path=target)
        current_manifest = foundation_manifest_digest(root)
        current_suite = foundation_suite_digest()
        stored_load_policy = value.get("load_policy") if isinstance(value.get("load_policy"), dict) else {}
        current_preflight = foundation_preflight(
            root,
            max_memory=stored_load_policy.get("max_memory"),
            probe_hardware=False,
        )

        report = value.get("report")
        minimum_score = value.get("minimum_score")
        report_score = report.get("score") if isinstance(report, dict) else None
        domain_scores = report.get("domain_scores") if isinstance(report, dict) else None
        critical_failures = report.get("critical_failures") if isinstance(report, dict) else None
        try:
            minimum_score_value = float(minimum_score)
            report_score_value = float(report_score)
            score_values = (
                [float(domain_scores[name]) for name in FOUNDATION_DOMAINS]
                if isinstance(domain_scores, dict)
                and all(name in domain_scores for name in FOUNDATION_DOMAINS)
                else []
            )
            semantics_ok = bool(
                math.isfinite(minimum_score_value)
                and FOUNDATION_MINIMUM_SCORE <= minimum_score_value <= 100.0
                and math.isfinite(report_score_value)
                and report_score_value >= minimum_score_value
                and report_score_value <= 100.0
                and report.get("ok") is True
                and isinstance(critical_failures, list)
                and not critical_failures
                and len(score_values) == len(FOUNDATION_DOMAINS)
                and all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in score_values)
            )
        except (TypeError, ValueError, OverflowError):
            semantics_ok = False

        stored_preflight = value.get("preflight")
        preflight_ok = bool(
            isinstance(stored_preflight, dict)
            and int(stored_preflight.get("preflight_version", 0)) == FOUNDATION_PREFLIGHT_VERSION
            and stored_preflight.get("ok") is True
            and not stored_preflight.get("blockers")
            and current_preflight.get("ok") is True
            and not current_preflight.get("blockers")
        )

        integrity_ok = bool(
            value.get("backend_type") == "transformers-foundation"
            and value.get("attested_by") == "airi-generalist-foundation-qualification-v4"
            and int(value.get("qualification_version", 0)) == QUALIFICATION_VERSION
            and int(value.get("foundation_qualification_version", 0)) == FOUNDATION_QUALIFICATION_VERSION
            and int(value.get("suite_version", 0)) == FOUNDATION_SUITE_VERSION
            and isinstance(value.get("inference_profile"), dict)
            and value.get("inference_profile", {}).get("torch_dtype") in {"auto", "float16", "bfloat16", "float32"}
            and value.get("model_digest") == current_model
            and value.get("manifest_digest") == current_manifest
            and value.get("suite_digest") == current_suite
            and semantics_ok
            and preflight_ok
        )
        return {
            **value,
            "integrity_ok": integrity_ok,
            "qualification_semantics_ok": semantics_ok,
            "preflight_ok": preflight_ok,
            "current_preflight": current_preflight,
            "qualified": bool(value.get("qualified") and integrity_ok),
            "current_model_digest": current_model,
            "current_manifest_digest": current_manifest,
            "current_suite_digest": current_suite,
        }
    except Exception as exc:
        return {
            "qualified": False,
            "integrity_ok": False,
            "qualification_semantics_ok": False,
            "preflight_ok": False,
            "reason": f"missing_or_invalid_foundation_attestation:{type(exc).__name__}",
        }
