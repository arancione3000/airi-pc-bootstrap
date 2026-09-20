from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .training import SFTExample

_ALLOWED_PROVENANCE = {
    "builtin_verified",
    "human_verified",
    "control_plane_success",
    "tool_verified",
    "mathesis_proof_gated",
}


def _canonical_messages(messages: list[dict[str, str]]) -> str:
    normalized = [
        {
            "role": str(row.get("role", "")).strip().lower(),
            "content": " ".join(str(row.get("content", "")).strip().split()),
        }
        for row in messages
    ]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def example_id(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(_canonical_messages(messages).encode("utf-8")).hexdigest()[:24]


def _validate_record(raw: dict[str, Any]) -> dict[str, Any]:
    example = SFTExample.from_dict(raw)
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("verified SFT record requires provenance")
    kind = str(provenance.get("kind", "")).strip()
    if kind not in _ALLOWED_PROVENANCE:
        raise ValueError(f"unapproved provenance kind: {kind!r}")
    source = str(provenance.get("source", "")).strip()
    evidence = str(provenance.get("evidence", "")).strip()
    if not source or not evidence:
        raise ValueError("verified SFT provenance requires source and evidence")
    messages = [{"role": m["role"].strip().lower(), "content": m["content"]} for m in example.messages]
    return {
        "id": example_id(messages),
        "messages": messages,
        "provenance": {
            "kind": kind,
            "source": source[:1000],
            "evidence": evidence[:4000],
        },
    }


def load_verified(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = _validate_record(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid verified SFT row at line {line_no}: {exc}") from exc
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append(row)
    return rows


def append_verified_many(path: str | Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")
    accepted: list[dict[str, Any]] = []
    duplicates = 0
    invalid = 0

    with lock.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = load_verified(target) if target.exists() else []
        known = {row["id"] for row in existing}
        with target.open("a", encoding="utf-8") as handle:
            for raw in records:
                try:
                    row = _validate_record(raw)
                except Exception:
                    invalid += 1
                    continue
                if row["id"] in known:
                    duplicates += 1
                    continue
                known.add(row["id"])
                accepted.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    return {
        "accepted": len(accepted),
        "duplicates": duplicates,
        "invalid": invalid,
        "total": len(load_verified(target)),
        "accepted_rows": accepted,
    }


def as_sft_examples(rows: Iterable[dict[str, Any]]) -> list[SFTExample]:
    return [SFTExample.from_dict(row) for row in rows]


def deterministic_split(rows: list[dict[str, Any]], *, holdout_mod: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if holdout_mod < 3:
        raise ValueError("holdout_mod must be at least 3")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in rows:
        bucket = int(hashlib.sha256(row["id"].encode("ascii")).hexdigest()[:8], 16) % holdout_mod
        (holdout if bucket == 0 else train).append(row)
    if rows and not holdout:
        holdout.append(rows[-1])
        train = rows[:-1]
    if len(rows) > 1 and not train:
        train = rows[:-1]
        holdout = rows[-1:]
    return train, holdout


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
