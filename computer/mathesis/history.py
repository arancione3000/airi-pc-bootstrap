from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .knowledge import default_state_dir


HISTORY_SCHEMA = "mathesis-evolution-history-v2"
HISTORY_VERSION = 2
HISTORY_MANIFEST_VERSION = 1
ACTIVE_HISTORY_MAX_BYTES = 8 * 1024 * 1024
ACTIVE_HISTORY_MAX_RECORDS = 2048
HISTORY_NAME = "history.jsonl"
MANIFEST_NAME = "history-manifest.json"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_bytes(value) + b"\n")


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _genome_summary(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    return {
        "genome_id": str(row.get("genome_id", "")),
        "generation": int(row.get("generation", 0) or 0),
        "parent_id": row.get("parent_id"),
        "neural_hidden": int(row.get("neural_hidden", 0) or 0),
        "symbolic_depth": int(row.get("symbolic_depth", 0) or 0),
        "discovery_beam": int(row.get("discovery_beam", 0) or 0),
        "max_proof_cells": int(row.get("max_proof_cells", 0) or 0),
        "research_budget": int(row.get("research_budget", 0) or 0),
        "expert_count": len(row.get("experts") or []),
        "strategy_count": len(row.get("strategy_portfolio") or []),
    }


def _benchmark_summary(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    critical = [str(x) for x in (row.get("critical_failures") or [])]
    weaknesses = [str(x) for x in (row.get("weaknesses") or [])]
    return {
        "ok": bool(row.get("ok", False)),
        "score": float(row.get("score", 0.0) or 0.0),
        "capability_score": float(row.get("capability_score", 0.0) or 0.0),
        "complexity_penalty": float(row.get("complexity_penalty", 0.0) or 0.0),
        "neural_accuracy": float(row.get("neural_accuracy", 0.0) or 0.0),
        "critical_failures": critical,
        "critical_failure_count": len(critical),
        "weaknesses": weaknesses,
    }


def compact_evolution_row(
    row: dict[str, Any],
    *,
    raw_record_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Compact a full evolution audit row without changing evolution semantics.

    The active audit history intentionally stores summaries and cryptographic
    fingerprints, while mathematical memory remains in discoveries/knowledge
    and the current model state remains in champion/router.  A legacy full row
    can therefore be compacted without becoming training or proof input.
    """

    learned = list(row.get("learned_theorems") or [])
    learned_payload = _canonical_bytes(learned)
    source_payload = (
        raw_record_bytes
        if raw_record_bytes is not None
        else _canonical_bytes(row)
    )

    trials: list[dict[str, Any]] = []
    for item in row.get("trials") or []:
        if not isinstance(item, dict):
            continue
        trials.append({
            "trial": int(item.get("trial", 0) or 0),
            "genome": _genome_summary(item.get("genome")),
            "benchmark": _benchmark_summary(item.get("benchmark")),
        })

    integrity = row.get("integrity") if isinstance(row.get("integrity"), dict) else {}
    integrity_current = integrity.get("current") if isinstance(integrity.get("current"), dict) else {}

    return {
        "schema": HISTORY_SCHEMA,
        "version": HISTORY_VERSION,
        "at": float(row.get("at", 0.0) or 0.0),
        "promoted": bool(row.get("promoted", False)),
        "reason": str(row.get("reason", "")),
        "champion": _genome_summary(row.get("champion")),
        "candidate": _genome_summary(row.get("candidate")),
        "selected": _genome_summary(row.get("selected")),
        "selected_trial": int(row.get("selected_trial", 0) or 0),
        "champion_benchmark": _benchmark_summary(row.get("champion_benchmark")),
        "candidate_benchmark": _benchmark_summary(row.get("candidate_benchmark")),
        "trial_summaries": trials,
        "weakness_hints": [str(x) for x in (row.get("weakness_hints") or [])],
        "learned_theorems": {
            "count": len(learned),
            "sha256": _sha256_bytes(learned_payload),
        },
        "kernel_integrity": {
            "ok": bool(integrity.get("ok", False)),
            "changed": [str(x) for x in (integrity.get("changed") or [])],
            "current_sha256": _sha256_bytes(_canonical_bytes(integrity_current)),
        },
        "full_record_sha256": _sha256_bytes(source_payload),
        "full_record_bytes": len(source_payload),
    }


def _bounded_push(
    rows: deque[bytes],
    line: bytes,
    total_bytes: int,
) -> int:
    if len(line) > ACTIVE_HISTORY_MAX_BYTES:
        raise ValueError("one compact MATHESIS history record exceeds the active byte budget")
    rows.append(line)
    total_bytes += len(line)
    while (
        len(rows) > ACTIVE_HISTORY_MAX_RECORDS
        or total_bytes > ACTIVE_HISTORY_MAX_BYTES
    ):
        total_bytes -= len(rows.popleft())
    return total_bytes


def _active_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    records = 0
    first_at: float | None = None
    last_at: float | None = None
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            stripped = raw.rstrip(b"\r\n")
            if not stripped.strip():
                continue
            row = json.loads(stripped.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("history row must be a JSON object")
            if row.get("schema") != HISTORY_SCHEMA or int(row.get("version", 0) or 0) != HISTORY_VERSION:
                raise ValueError("active history contains a non-compact schema")
            at = float(row.get("at", 0.0) or 0.0)
            first_at = at if first_at is None else first_at
            last_at = at
            records += 1
    return {
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "records": records,
        "first_at": first_at,
        "last_at": last_at,
    }


def _write_manifest(
    state_dir: Path,
    *,
    active: dict[str, Any],
    legacy_compaction: dict[str, Any] | None,
) -> None:
    manifest = {
        "version": HISTORY_MANIFEST_VERSION,
        "active_schema": HISTORY_SCHEMA,
        "active_file": HISTORY_NAME,
        "policy": {
            "max_bytes": ACTIVE_HISTORY_MAX_BYTES,
            "max_records": ACTIVE_HISTORY_MAX_RECORDS,
            "history_is_audit_only": True,
            "mathematical_memory_files": [
                "champion.json",
                "router.json",
                "discoveries.json",
                "knowledge.json",
                "curriculum.json",
            ],
        },
        "active": active,
        "legacy_compaction": legacy_compaction,
    }
    _atomic_json(state_dir / MANIFEST_NAME, manifest)


def normalize_history(state_dir: str | Path | None = None) -> dict[str, Any]:
    """Migrate/compact history.jsonl using streaming, byte-bounded retention.

    Legacy full rows are replaced by compact audit rows carrying an SHA-256 of
    the original exact JSON record.  This is deterministic and idempotent; the
    mathematical state files are never opened for writing.
    """

    root = Path(state_dir or default_state_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / HISTORY_NAME
    manifest_path = root / MANIFEST_NAME
    existing_manifest = _read_json(manifest_path, {})
    previous_legacy = (
        existing_manifest.get("legacy_compaction")
        if isinstance(existing_manifest, dict)
        else None
    )

    if not path.exists():
        _atomic_bytes(path, b"")
        active = _active_metadata(path)
        _write_manifest(root, active=active, legacy_compaction=previous_legacy)
        return {
            "ok": True,
            "migrated": False,
            "legacy_records": 0,
            "dropped_records": 0,
            "active": active,
        }

    source_digest = hashlib.sha256()
    source_bytes = 0
    source_records = 0
    legacy_records = 0
    rows: deque[bytes] = deque()
    retained_bytes = 0

    with path.open("rb") as handle:
        for raw in handle:
            source_digest.update(raw)
            source_bytes += len(raw)
            stripped = raw.rstrip(b"\r\n")
            if not stripped.strip():
                continue
            source_records += 1
            try:
                parsed = json.loads(stripped.decode("utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"invalid MATHESIS history JSON at record {source_records}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"MATHESIS history record {source_records} is not an object"
                )

            if (
                parsed.get("schema") == HISTORY_SCHEMA
                and int(parsed.get("version", 0) or 0) == HISTORY_VERSION
            ):
                compact = parsed
            else:
                legacy_records += 1
                compact = compact_evolution_row(
                    parsed,
                    raw_record_bytes=stripped,
                )

            line = _canonical_bytes(compact) + b"\n"
            retained_bytes = _bounded_push(rows, line, retained_bytes)

    payload = b"".join(rows)
    _atomic_bytes(path, payload)
    active = _active_metadata(path)

    legacy_compaction = previous_legacy
    if legacy_records:
        legacy_compaction = {
            "source_format": "legacy-full-evolution-row-jsonl",
            "source_sha256": source_digest.hexdigest(),
            "source_bytes": source_bytes,
            "source_records": source_records,
            "legacy_records_compacted": legacy_records,
            "retained_records": active["records"],
            "dropped_records": max(0, source_records - active["records"]),
            "proof": (
                "each retained compact record stores full_record_sha256 over "
                "the exact original JSON record bytes"
            ),
        }

    _write_manifest(
        root,
        active=active,
        legacy_compaction=legacy_compaction,
    )
    return {
        "ok": True,
        "migrated": bool(legacy_records),
        "legacy_records": legacy_records,
        "dropped_records": max(0, source_records - active["records"]),
        "source_bytes": source_bytes,
        "source_records": source_records,
        "active": active,
        "manifest": str(manifest_path),
    }


def append_evolution_history(
    state_dir: str | Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    root = Path(state_dir).resolve()
    normalized = normalize_history(root)
    path = root / HISTORY_NAME

    rows: deque[bytes] = deque()
    retained_bytes = 0
    with path.open("rb") as handle:
        for raw in handle:
            stripped = raw.rstrip(b"\r\n")
            if not stripped.strip():
                continue
            parsed = json.loads(stripped.decode("utf-8"))
            line = _canonical_bytes(parsed) + b"\n"
            retained_bytes = _bounded_push(rows, line, retained_bytes)

    compact = compact_evolution_row(row)
    new_line = _canonical_bytes(compact) + b"\n"
    retained_bytes = _bounded_push(rows, new_line, retained_bytes)
    _atomic_bytes(path, b"".join(rows))

    active = _active_metadata(path)
    manifest = _read_json(root / MANIFEST_NAME, {})
    legacy = manifest.get("legacy_compaction") if isinstance(manifest, dict) else None
    _write_manifest(root, active=active, legacy_compaction=legacy)

    return {
        "ok": True,
        "normalized": normalized,
        "active": active,
        "appended_record_sha256": compact["full_record_sha256"],
    }


def audit_history_store(state_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(state_dir or default_state_dir()).resolve()
    path = root / HISTORY_NAME
    manifest_path = root / MANIFEST_NAME
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if not path.exists():
        check("history:present", False, str(path))
        return {"ok": False, "checks": checks, "active": None}

    check("history:present", True, str(path))
    check(
        "history:byte_budget",
        path.stat().st_size <= ACTIVE_HISTORY_MAX_BYTES,
        {"bytes": path.stat().st_size, "max": ACTIVE_HISTORY_MAX_BYTES},
    )

    parse_error: str | None = None
    metadata: dict[str, Any] | None = None
    last_selected = ""
    try:
        metadata = _active_metadata(path)
        check(
            "history:record_budget",
            metadata["records"] <= ACTIVE_HISTORY_MAX_RECORDS,
            {"records": metadata["records"], "max": ACTIVE_HISTORY_MAX_RECORDS},
        )
        with path.open("rb") as handle:
            for raw in handle:
                stripped = raw.rstrip(b"\r\n")
                if not stripped.strip():
                    continue
                row = json.loads(stripped.decode("utf-8"))
                last_selected = str((row.get("selected") or {}).get("genome_id", ""))
    except Exception as exc:
        parse_error = repr(exc)
        check("history:parseable_compact_jsonl", False, parse_error)
    else:
        check("history:parseable_compact_jsonl", True, metadata)

    manifest = _read_json(manifest_path, None)
    manifest_ok = (
        isinstance(manifest, dict)
        and int(manifest.get("version", 0) or 0) == HISTORY_MANIFEST_VERSION
        and manifest.get("active_schema") == HISTORY_SCHEMA
    )
    check("history:manifest_valid", manifest_ok, manifest)

    if manifest_ok and metadata is not None:
        recorded = manifest.get("active") or {}
        check(
            "history:manifest_matches_active",
            recorded.get("sha256") == metadata.get("sha256")
            and int(recorded.get("bytes", -1)) == metadata.get("bytes")
            and int(recorded.get("records", -1)) == metadata.get("records"),
            {"recorded": recorded, "actual": metadata},
        )

    champion = _read_json(root / "champion.json", {})
    champion_id = (
        str(champion.get("genome_id", ""))
        if isinstance(champion, dict)
        else ""
    )
    if metadata is not None and metadata.get("records", 0) > 0:
        check(
            "history:latest_selected_matches_champion",
            bool(champion_id) and last_selected == champion_id,
            {"champion": champion_id, "history_selected": last_selected},
        )

    failed = [row for row in checks if not row["ok"]]
    return {
        "ok": not failed,
        "checks": checks,
        "active": metadata,
        "manifest": manifest if manifest_ok else None,
        "failed": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and audit MATHESIS history")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Do not mutate; only audit the current bounded history store.",
    )
    args = parser.parse_args()

    if args.audit_only:
        result = audit_history_store(args.state_dir)
    else:
        migration = normalize_history(args.state_dir)
        audit = audit_history_store(args.state_dir)
        result = {"ok": bool(migration["ok"] and audit["ok"]), "migration": migration, "audit": audit}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
