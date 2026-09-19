from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .discovery import validate_verified_discovery
from .knowledge import default_state_dir
from .safe_math import parse_relation
from .verifiers import CompositeVerifier
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
        """Return bounded proof obligations for previously verified mathematics.

        Ordinary relation-style discoveries replay their theorem statement.
        Richer schemas such as Faulhaber sums replay the verified, structurally
        nontrivial induction recurrence stored in their certificate. This keeps
        anti-forgetting proof-gated without pretending the safe relation parser
        understands a richer theorem language than it actually does.
        """
        discoveries = self._json("discoveries.json", {"theorems": {}})
        rows = sorted(
            (discoveries.get("theorems") or {}).values(),
            key=lambda row: (float(row.get("discovered_at", 0)), str(row.get("id", ""))),
            reverse=True,
        )
        out: list[str] = []
        seen: set[str] = set()
        target = max(1, min(64, int(limit)))
        verifier = CompositeVerifier()

        for row in rows:
            if not row.get("verified"):
                continue
            still_valid, _reason = validate_verified_discovery(row, verifier)
            if not still_valid:
                continue

            candidates: list[str] = []
            statement = str(row.get("statement", "")).strip()
            try:
                parse_relation(statement)
            except Exception:
                pass
            else:
                candidates.append(statement)

            if row.get("strategy") == "faulhaber_interpolation":
                cert = row.get("certificate") or {}
                recurrence = cert.get("recurrence") or {}
                recurrence_statement = str(recurrence.get("statement", "")).strip()
                if (
                    cert.get("recurrence_nontrivial") is True
                    and recurrence.get("ok") is True
                    and recurrence_statement
                ):
                    try:
                        parse_relation(recurrence_statement)
                    except Exception:
                        pass
                    else:
                        candidates.append(recurrence_statement)

            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                out.append(candidate)
                if len(out) >= target:
                    return list(reversed(out))

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
        rows = sorted(
            (discoveries.get("theorems") or {}).values(),
            key=lambda row: (float(row.get("discovered_at", 0)), str(row.get("id", ""))),
        )

        # Verified discoveries keep exerting architectural pressure until the
        # corresponding mathematical expert actually exists. This prevents a
        # useful signal from disappearing just because a different challenger
        # won the immediately following generation.
        for learned in rows[-50:]:
            if not learned.get("verified"):
                continue
            learned_strategy = str(learned.get("strategy", ""))
            learned_domain = _DISCOVERY_DOMAIN.get(learned_strategy)
            if learned_domain and learned_domain not in champion.experts:
                weaknesses.append(f"missing_{learned_domain}_expert")
                reasons.append({
                    "signal": learned_domain,
                    "reason": f"persistent verified discovery used {learned_strategy} but champion still lacks {learned_domain}",
                })

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
