from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .knowledge import default_state_dir
from .safe_math import parse_relation
from .types import ArchitectureGenome


_DISCOVERY_DOMAIN = {
    "binomial_expansion": "combinatorics",
    "difference_of_powers": "algebra",
    "faulhaber_interpolation": "sequences",
    "finite_geometric_identity": "sequences",
}


class ExperienceAnalyzer:
    """Turn mathematical experience into architecture-search signals.

    This is intentionally a small, auditable feedback layer. It does not let
    web text or a conjecture rewrite code directly; it only proposes weaknesses
    and priorities consumed by the bounded architecture DSL.
    """

    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()

    def _json(self, name: str, fallback: Any) -> Any:
        try:
            return json.loads((self.state_dir / name).read_text(encoding="utf-8"))
        except Exception:
            return fallback

    def replayable_theorems(self, limit: int = 16) -> list[str]:
        """Return a bounded deterministic corpus of previously verified relations.

        Only statements accepted by the safe mathematical relation parser are
        replayed as promotion tests. Richer theorem schemas remain in the
        knowledge graph but are not silently downgraded into string heuristics.
        """
        discoveries = self._json("discoveries.json", {"theorems": {}})
        rows = sorted(
            (discoveries.get("theorems") or {}).values(),
            key=lambda row: (float(row.get("discovered_at", 0)), str(row.get("id", ""))),
            reverse=True,
        )
        out: list[str] = []
        for row in rows:
            if not row.get("verified"):
                continue
            statement = str(row.get("statement", "")).strip()
            try:
                parse_relation(statement)
            except Exception:
                continue
            out.append(statement)
            if len(out) >= max(1, min(64, int(limit))):
                break
        return list(reversed(out))

    def signals(
        self,
        champion: ArchitectureGenome,
        *,
        latest_discovery: dict[str, Any] | None = None,
        latest_study: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        weaknesses: list[str] = []
        reasons: list[dict[str, str]] = []

        discovery = latest_discovery or {}
        theorem = discovery.get("theorem") or {}
        strategy = str(theorem.get("strategy", ""))
        verified = bool(theorem.get("verified", False))

        domain = _DISCOVERY_DOMAIN.get(strategy)
        if verified and domain and domain not in champion.experts:
            weaknesses.append(f"missing_{domain}_expert")
            reasons.append({
                "signal": domain,
                "reason": f"verified discovery used {strategy} but champion lacks {domain}",
            })

        if theorem and not verified:
            weaknesses.extend(["counterexample", "optimization"])
            reasons.append({
                "signal": "proof_search",
                "reason": "latest conjecture was rejected; prioritize stronger falsification/search",
            })

        complexity = int(theorem.get("complexity", 0) or 0)
        if verified and complexity >= max(2, champion.symbolic_depth):
            weaknesses.append("symbolic_depth")
            reasons.append({
                "signal": "symbolic_depth",
                "reason": "verified discovery reached current symbolic-depth frontier",
            })

        study = latest_study or {}
        studied_domain = str(study.get("domain", ""))
        if study.get("ok") and studied_domain and studied_domain not in champion.experts:
            weaknesses.append(f"missing_{studied_domain}_expert")
            reasons.append({
                "signal": studied_domain,
                "reason": "read-only curriculum studied a domain not represented by an expert",
            })
        elif study and not study.get("ok", True):
            weaknesses.append("research")
            reasons.append({
                "signal": "research",
                "reason": "curriculum research failed or returned no usable sources",
            })

        discoveries = self._json("discoveries.json", {"theorems": {}})
        rows = list((discoveries.get("theorems") or {}).values())
        rejected = sum(1 for row in rows[-50:] if not row.get("verified"))
        if rejected >= 3:
            weaknesses.append("optimization")
            reasons.append({
                "signal": "optimization",
                "reason": "recent discovery rejection rate suggests search strategy needs tuning",
            })

        curriculum = self._json("curriculum.json", {"studies": []})
        for row in list(curriculum.get("studies") or [])[-12:]:
            domain = str(row.get("domain", ""))
            if row.get("status") == "studied" and domain and domain not in champion.experts:
                weaknesses.append(f"missing_{domain}_expert")

        # Preserve deterministic order so architecture evolution is reproducible.
        deduped = list(dict.fromkeys(weaknesses))
        return {
            "ok": True,
            "weaknesses": deduped,
            "reasons": reasons,
            "champion": champion.genome_id,
            "policy": "experience proposes bounded architecture hints; it never edits verifier code",
        }
