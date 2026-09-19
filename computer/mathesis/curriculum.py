from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .kernel import atomic_json
from .knowledge import default_state_dir
from .research import ReadOnlyResearcher
from .sympy_lab import DOMAIN_ATLAS


class MathematicalCurriculum:
    """Rotating read-only mathematical curriculum.

    It samples broad mathematical domains and records source evidence from
    public documentation. Retrieved web text can influence what MATHESIS studies
    next, but cannot directly become a VERIFIED theorem.
    """

    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()
        self.path = self.state_dir / "curriculum.json"
        self.researcher = ReadOnlyResearcher()

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("cursor", 0)
                value.setdefault("studies", [])
                return value
        except Exception:
            pass
        return {"version": 1, "cursor": 0, "studies": []}

    def study_once(self) -> dict[str, Any]:
        state = self._load()
        domains = tuple(DOMAIN_ATLAS)
        cursor = int(state.get("cursor", 0))
        domain = domains[cursor % len(domains)]
        query = (
            f"{domain} mathematics SymPy documentation Lean mathlib theorem proving "
            "definitions theorems algorithms"
        )
        try:
            research = self.researcher.search(query, max_sources=3)
            sources = [
                {
                    "url": row.get("url"),
                    "domain": row.get("domain"),
                    "authority_hint": row.get("authority_hint"),
                    "excerpt_sha256": hashlib.sha256(str(row.get("excerpt", "")).encode("utf-8")).hexdigest(),
                    "excerpt": str(row.get("excerpt", ""))[:600],
                }
                for row in research.get("sources", [])
            ]
            status = "studied" if sources else "no_sources"
        except Exception as exc:
            sources = []
            status = "research_error"
            research = {"error": repr(exc)}

        row = {
            "at": time.time(),
            "domain": domain,
            "query": query,
            "status": status,
            "sources": sources,
            "web_truth_policy": "source evidence is not a formal proof",
        }
        state["cursor"] = cursor + 1
        state["studies"].append(row)
        state["studies"] = state["studies"][-100:]
        state["updated_at"] = row["at"]
        self.state_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, state)
        return {"ok": status == "studied", **row, "raw": research if status != "studied" else None}

    def status(self) -> dict[str, Any]:
        state = self._load()
        return {
            "ok": True,
            "cursor": int(state.get("cursor", 0)),
            "studies": len(state.get("studies", [])),
            "last": state.get("studies", [])[-1] if state.get("studies") else None,
            "domains": list(DOMAIN_ATLAS),
        }
