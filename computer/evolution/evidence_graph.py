from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .data import _dataset_lock

GRAPH_VERSION = 1
MAX_CLAIMS = 2000
MAX_ATTEMPTS_PER_CLAIM = 20


def _claim_id(claim: str) -> str:
    normalized = " ".join(str(claim or "").strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def graph_path(state_dir: Path) -> Path:
    return Path(state_dir) / "evidence" / "graph.json"


def _confidence(verification: dict[str, Any]) -> float:
    if not verification.get("verified"):
        return 0.0
    evidence = verification.get("evidence") or []
    similarities = [
        float(item.get("similarity", 0.0))
        for item in evidence
        if isinstance(item, dict)
    ]
    similarity = sum(similarities) / len(similarities) if similarities else 0.0
    source_count = max(1, int(verification.get("source_count", len(evidence)) or 1))
    # This is an evidence-strength score, not a probability that the claim is true.
    diversity = min(1.0, 0.55 + 0.15 * source_count)
    return round(max(0.0, min(0.99, similarity * diversity)), 4)


def decision_from_verification(verification: dict[str, Any]) -> str:
    if verification.get("verified") and verification.get("label") == 1:
        return "verified_true"
    if verification.get("verified") and verification.get("label") == 0:
        return "verified_false"
    if verification.get("reason") == "independent ClaimReview sources disagree":
        return "conflict"
    return "insufficient_evidence"


def record_verification(
    state_dir: Path,
    claim: str,
    verification: dict[str, Any],
    *,
    candidate_urls: list[str] | None = None,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    path = graph_path(state_dir)
    qid = _claim_id(claim)
    stamp = time.time()
    decision = decision_from_verification(verification)
    attempt = {
        "at": stamp,
        "decision": decision,
        "label": verification.get("label") if verification.get("verified") else None,
        "evidence_confidence": _confidence(verification),
        "source_count": int(verification.get("source_count", 0) or 0),
        "domains": sorted({
            str(x) for x in (verification.get("domains") or []) if str(x).strip()
        }),
        "candidate_urls": list(dict.fromkeys(str(x) for x in (candidate_urls or []) if str(x).strip()))[:20],
        "evidence": (verification.get("evidence") or [])[:20],
        "errors": (verification.get("errors") or [])[:20],
        "reason": verification.get("reason"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with _dataset_lock(path, timeout=5.0, stale_after=900.0):
        graph = _read_json(path, {"version": GRAPH_VERSION, "claims": {}})
        if not isinstance(graph, dict) or not isinstance(graph.get("claims"), dict):
            graph = {"version": GRAPH_VERSION, "claims": {}}
        claims = graph["claims"]
        node = claims.get(qid) if isinstance(claims.get(qid), dict) else {}
        history = list(node.get("history") or [])
        history.append(attempt)
        history = history[-MAX_ATTEMPTS_PER_CLAIM:]
        node = {
            "id": qid,
            "claim": str(claim).strip(),
            "decision": decision,
            "label": attempt["label"],
            "evidence_confidence": attempt["evidence_confidence"],
            "source_count": attempt["source_count"],
            "domains": attempt["domains"],
            "first_seen_at": float(node.get("first_seen_at") or stamp),
            "updated_at": stamp,
            "history": history,
        }
        claims[qid] = node

        if len(claims) > MAX_CLAIMS:
            ordered = sorted(
                claims.items(),
                key=lambda item: float((item[1] or {}).get("updated_at", 0.0)),
                reverse=True,
            )
            graph["claims"] = dict(ordered[:MAX_CLAIMS])
        graph["version"] = GRAPH_VERSION
        graph["updated_at"] = stamp
        _write_json(path, graph)

    return node


def status(state_dir: Path) -> dict[str, Any]:
    path = graph_path(Path(state_dir))
    graph = _read_json(path, {"version": GRAPH_VERSION, "claims": {}})
    claims = graph.get("claims") if isinstance(graph, dict) else {}
    if not isinstance(claims, dict):
        claims = {}
    decisions: dict[str, int] = {}
    for node in claims.values():
        decision = str((node or {}).get("decision") or "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
    return {
        "ok": True,
        "version": int(graph.get("version", GRAPH_VERSION) or GRAPH_VERSION) if isinstance(graph, dict) else GRAPH_VERSION,
        "claims": len(claims),
        "decisions": decisions,
        "path": str(path),
        "updated_at": graph.get("updated_at") if isinstance(graph, dict) else None,
    }
