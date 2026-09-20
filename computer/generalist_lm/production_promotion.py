from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from .evolution import promotion_decision
from .qualification import QUALIFICATION_VERSION, checkpoint_digest, qualification_status, qualify_checkpoint


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _copy_checkpoint(source: Path, target: Path, *, source_digest: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    config_path = source / "config.json"
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError("research checkpoint is missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("research checkpoint config must be a JSON object")
    tokenizer_version = str(config.get("tokenizer_version", "byte-v1"))

    names = ["config.json", "model.pt", "metadata.json"]
    if tokenizer_version == "bpe-v1":
        names.append("tokenizer.json")
    elif tokenizer_version != "byte-v1":
        raise ValueError(f"unsupported checkpoint tokenizer version: {tokenizer_version}")

    for name in names:
        src = source / name
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"research checkpoint is missing {name}")
        shutil.copy2(src, target / name)

    # A byte checkpoint must never inherit a stale tokenizer artifact from an
    # interrupted/previous candidate directory.
    if tokenizer_version == "byte-v1":
        (target / "tokenizer.json").unlink(missing_ok=True)

    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("research checkpoint metadata must be a JSON object")
    metadata["source_research_checkpoint_digest"] = source_digest
    metadata["source_role"] = metadata.get("role")
    metadata["role"] = "production_champion"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def attempt_production_promotion(
    research_checkpoint: str | Path,
    production_checkpoint: str | Path,
    *,
    minimum_score: float = 85.0,
    minimum_gain: float = 2.0,
    max_domain_regression: float = 0.0,
    force_requalify: bool = False,
) -> dict[str, Any]:
    """Qualify a research champion and atomically promote it if it is better.

    Research and production remain distinct. A research checkpoint can only
    enter production after the protected generative qualification suite. When
    a qualified production model already exists, the new model must additionally
    beat it by a meaningful score margin without regressing any protected
    benchmark domain.
    """
    source = Path(research_checkpoint).expanduser().resolve()
    target = Path(production_checkpoint).expanduser().resolve()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    source_digest = checkpoint_digest(source)
    attempt_path = parent / "promotion-attempt.json"
    history_path = parent / "promotion-history.jsonl"

    if not force_requalify:
        try:
            previous = json.loads(attempt_path.read_text(encoding="utf-8"))
            if (
                isinstance(previous, dict)
                and previous.get("source_checkpoint_digest") == source_digest
                and previous.get("qualification_version") == QUALIFICATION_VERSION
            ):
                if previous.get("promoted"):
                    current = qualification_status(target) if target.exists() else {"qualified": False}
                    if not current.get("qualified"):
                        previous = {}
                    else:
                        return {
                            **previous,
                            "cached": True,
                            "reason": "research champion digest already promoted and production integrity is healthy",
                        }
                else:
                    return {
                        **previous,
                        "cached": True,
                        "reason": "research champion digest already evaluated",
                    }
        except Exception:
            pass

    candidate = parent / ".production-candidate"
    backup = parent / ".production-backup"
    shutil.rmtree(candidate, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    _copy_checkpoint(source, candidate, source_digest=source_digest)

    qualification = qualify_checkpoint(candidate, minimum_score=minimum_score)
    candidate_report = qualification.get("report") or {}
    existing = qualification_status(target) if target.exists() else {"qualified": False}

    eligible = False
    reason = ""
    if not qualification.get("qualified"):
        reason = "research champion did not pass production qualification"
    elif existing.get("qualified"):
        eligible, reason = promotion_decision(
            existing.get("report") or {},
            candidate_report,
            minimum_gain=minimum_gain,
            max_domain_regression=max_domain_regression,
        )
    else:
        eligible = True
        reason = "first qualified production champion"

    result = {
        "ok": True,
        "promoted": False,
        "cached": False,
        "source_checkpoint_digest": source_digest,
        "qualification_version": int(qualification.get("qualification_version", 0) or 0),
        "qualified": bool(qualification.get("qualified")),
        "eligible": bool(eligible),
        "reason": reason,
        "candidate_score": float(candidate_report.get("score", 0.0) or 0.0),
        "previous_score": (
            float((existing.get("report") or {}).get("score", 0.0) or 0.0)
            if existing.get("qualified")
            else None
        ),
        "minimum_score": float(minimum_score),
        "minimum_gain": float(minimum_gain),
        "max_domain_regression": float(max_domain_regression),
        "updated_at": time.time(),
    }

    if eligible:
        try:
            if target.exists():
                target.replace(backup)
            candidate.replace(target)
            final_status = qualification_status(target)
            if not final_status.get("qualified"):
                raise RuntimeError("promoted checkpoint failed post-swap qualification integrity")
            result["promoted"] = True
            result["production_checkpoint_digest"] = final_status.get("current_checkpoint_digest")
            result["production_score"] = float((final_status.get("report") or {}).get("score", 0.0) or 0.0)
            shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if backup.exists():
                backup.replace(target)
            shutil.rmtree(candidate, ignore_errors=True)
            raise
    else:
        shutil.rmtree(candidate, ignore_errors=True)

    _atomic_json(attempt_path, result)
    _append_history(history_path, result)
    return result


def main() -> int:
    research = Path(
        os.environ.get(
            "AIRI_GENERALIST_RESEARCH_CHECKPOINT",
            str(Path(os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research")) / "champion"),
        )
    )
    production = Path(
        os.environ.get(
            "AIRI_GENERALIST_PRODUCTION_STATE",
            str(Path(os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research")) / "production"),
        )
    )
    result = attempt_production_promotion(
        research,
        production,
        minimum_score=float(os.environ.get("AIRI_GENERALIST_PRODUCTION_MIN_SCORE", "85")),
        minimum_gain=float(os.environ.get("AIRI_GENERALIST_PRODUCTION_MIN_GAIN", "2")),
        max_domain_regression=float(os.environ.get("AIRI_GENERALIST_PRODUCTION_MAX_DOMAIN_REGRESSION", "0")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
