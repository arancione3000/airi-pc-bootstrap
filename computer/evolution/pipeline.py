from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .claimreview import verify_consensus
from .data import append_verified, load_records


def _json_write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _json_read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def claim_id(claim: str) -> str:
    normalized = " ".join(str(claim).strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def queue_claim(state_dir: Path, claim: str, metadata: dict | None = None) -> dict[str, Any]:
    claim = str(claim).strip()
    if len(claim) < 8:
        raise ValueError("claim is too short")
    state_dir = Path(state_dir)
    qid = claim_id(claim)
    path = state_dir / "queue" / f"{qid}.json"
    existing = _json_read(path, None)
    if existing:
        return {**existing, "duplicate": True}
    row = {
        "id": qid,
        "claim": claim,
        "status": "pending",
        "created_at": time.time(),
        "metadata": metadata or {},
        "attempts": [],
    }
    _json_write(path, row)
    return {**row, "duplicate": False}


def queue_list(state_dir: Path, status: str | None = None, limit: int = 100) -> list[dict]:
    root = Path(state_dir) / "queue"
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        row = _json_read(path, None)
        if not isinstance(row, dict):
            continue
        if status and row.get("status") != status:
            continue
        rows.append(row)
        if len(rows) >= max(1, min(1000, int(limit))):
            break
    return rows


def verify_queued_claim(
    state_dir: Path,
    qid: str,
    urls: list[str],
    min_sources: int = 2,
    auto_ingest: bool = True,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    path = state_dir / "queue" / f"{qid}.json"
    row = _json_read(path, None)
    if not isinstance(row, dict):
        raise FileNotFoundError(f"queue claim not found: {qid}")
    result = verify_consensus(row["claim"], urls, min_sources=min_sources)
    attempt = {"at": time.time(), "urls": urls, "result": result}
    row.setdefault("attempts", []).append(attempt)
    if result.get("verified"):
        row["status"] = "verified"
        row["verified_at"] = time.time()
        row["verification"] = result
        if auto_ingest:
            try:
                added = append_verified(state_dir / "data" / "verified.jsonl", {
                    "text": row["claim"],
                    "label": result["label"],
                    "source": "ClaimReview consensus: " + ", ".join(result.get("domains", [])),
                    "evidence": json.dumps(result, ensure_ascii=False, sort_keys=True),
                })
                row["ingest"] = added
            except ValueError as exc:
                row["status"] = "conflict"
                row["ingest_error"] = str(exc)
    elif result.get("reason") == "independent ClaimReview sources disagree":
        row["status"] = "conflict"
    else:
        row["status"] = "pending"
    _json_write(path, row)
    return row


def pipeline_status(state_dir: Path) -> dict[str, Any]:
    rows = queue_list(state_dir, limit=1000)
    counts = {}
    for row in rows:
        key = row.get("status", "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {
        "queued": len(rows),
        "queue_status": counts,
        "verified_dataset_records": len(load_records(Path(state_dir) / "data" / "verified.jsonl")),
    }
