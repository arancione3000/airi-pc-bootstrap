from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .kernel import atomic_json


def default_state_dir() -> Path:
    configured = os.environ.get("MATHESIS_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home().joinpath(".airi", "mathesis").resolve()


class KnowledgeGraph:
    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()
        self.path = self.state_dir / "knowledge.json"

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("nodes"), dict):
                return value
        except Exception:
            pass
        return {"version": 1, "nodes": {}}

    def record(
        self,
        statement: str,
        *,
        status: str,
        certificate: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        dependencies: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized = " ".join(str(statement or "").split())
        node_id = hashlib.sha256(normalized.lower().encode("utf-8")).hexdigest()[:20]
        graph = self._load()
        now = time.time()
        previous = graph["nodes"].get(node_id, {})
        node = {
            "id": node_id,
            "statement": normalized,
            "status": status,
            "certificate": certificate or {},
            "sources": [
                {
                    "url": source.get("url"),
                    "domain": source.get("domain"),
                    "authority_hint": source.get("authority_hint"),
                }
                for source in (sources or [])[:10]
            ],
            "dependencies": list(dependencies or []),
            "first_seen_at": previous.get("first_seen_at", now),
            "updated_at": now,
        }
        graph["nodes"][node_id] = node
        graph["updated_at"] = now
        self.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, graph)
        return node

    def get(self, statement: str) -> dict[str, Any] | None:
        normalized = " ".join(str(statement or "").split())
        node_id = hashlib.sha256(normalized.lower().encode("utf-8")).hexdigest()[:20]
        return self._load()["nodes"].get(node_id)

    def status(self) -> dict[str, Any]:
        graph = self._load()
        counts: dict[str, int] = {}
        for node in graph["nodes"].values():
            state = str(node.get("status", "unknown"))
            counts[state] = counts.get(state, 0) + 1
        return {
            "ok": True,
            "nodes": len(graph["nodes"]),
            "status_counts": counts,
            "path": str(self.path),
        }
