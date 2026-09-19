from __future__ import annotations

import re
from dataclasses import dataclass

from .safe_math import parse_expr, parse_relation
from .types import Intent


_EVOLVE_PATTERNS = (
    "scrivimi il tuo prossimo modello",
    "crea il tuo prossimo modello",
    "genera il tuo prossimo modello",
    "migliora te stessa",
    "migliora te stesso",
    "evolvi te stessa",
    "evolvi te stesso",
    "write your next model",
    "create your next model",
    "improve yourself",
)

_PROGRAM_TASKS = {
    "gcd": ("gcd", "mcd", "massimo comune divisore", "greatest common divisor"),
    "fibonacci": ("fibonacci",),
    "factorial": ("factorial", "fattoriale"),
    "is_prime": ("is prime", "primo", "prime number", "numero primo"),
    "sort": ("sort", "ordinamento", "ordina"),
}


@dataclass(frozen=True)
class _Vote:
    kind: str
    expression: str | None
    variable: str | None
    target: str | None
    score: float
    source: str


def _clean_prompt(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _extract_after_prefix(raw: str, prefixes: tuple[str, ...]) -> str:
    lower = raw.lower()
    for prefix in prefixes:
        pos = lower.find(prefix)
        if pos >= 0:
            return raw[pos + len(prefix) :].strip(" :")
    return raw


def _program_target(lower: str) -> str | None:
    for task, aliases in _PROGRAM_TASKS.items():
        if any(alias in lower for alias in aliases):
            return task
    return None


def _keyword_vote(raw: str) -> _Vote | None:
    lower = raw.lower().strip()

    if any(pattern in lower for pattern in _EVOLVE_PATTERNS):
        return _Vote("evolve_model", None, None, "next_architecture", 1.0, "keyword")

    target = _program_target(lower)
    if target and any(word in lower for word in ("scrivi", "crea", "implementa", "funzione", "write", "implement", "code")):
        return _Vote("synthesize_program", None, None, target, 0.97, "keyword")

    if any(word in lower for word in ("ricerca", "cerca online", "verifica online", "research", "search the web", "fact check")):
        claim = _extract_after_prefix(
            raw,
            ("ricerca", "cerca online", "verifica online", "research", "search the web", "fact check"),
        )
        return _Vote("research_claim", claim or raw, None, None, 0.92, "keyword")

    if any(word in lower for word in ("dimostra", "prova che", "prove that", "show that")):
        expr = _extract_after_prefix(raw, ("dimostra", "prova che", "prove that", "show that"))
        return _Vote("identity", expr, None, None, 0.98, "keyword")

    if any(word in lower for word in ("risolvi", "solve")):
        expr = _extract_after_prefix(raw, ("risolvi", "solve"))
        variable = None
        m = re.search(r"\b(?:per|for)\s+([A-Za-z][A-Za-z0-9_]*)\b", expr, flags=re.I)
        if m:
            variable = m.group(1)
            expr = expr[: m.start()].strip()
        return _Vote("equation", expr, variable, None, 0.98, "keyword")

    if any(word in lower for word in ("calcola", "quanto fa", "evaluate", "calculate")):
        expr = _extract_after_prefix(raw, ("calcola", "quanto fa", "evaluate", "calculate"))
        return _Vote("arithmetic", expr, None, None, 0.96, "keyword")

    return None


def _structure_vote(raw: str) -> _Vote | None:
    text = raw.strip().rstrip("?").strip()
    try:
        rel = parse_relation(text)
        variable = rel.symbols[0].name if len(rel.symbols) == 1 else None
        if rel.symbols:
            return _Vote("proposition", text, variable, None, 0.72, "structure")
        return _Vote("proposition", text, None, None, 0.75, "structure")
    except Exception:
        pass

    try:
        expr = parse_expr(text)
        if not expr.free_symbols:
            return _Vote("arithmetic", text, None, None, 0.78, "structure")
    except Exception:
        pass
    return None


def _semantic_vote(raw: str) -> _Vote | None:
    lower = raw.lower()
    if "modello" in lower or "model" in lower:
        if any(x in lower for x in ("prossimo", "next", "evol", "miglior")):
            return _Vote("evolve_model", None, None, "next_architecture", 0.82, "semantic")
    if "=" in raw and any(ch.isalpha() for ch in raw):
        return _Vote("equation", raw.strip().rstrip("?"), None, None, 0.60, "semantic")
    return None


class FormalizerMesh:
    """Small independent formalizer ensemble.

    The mesh deliberately keeps natural-language understanding separate from
    mathematical truth. It proposes an intent; downstream proof engines decide
    whether the mathematical content is correct.
    """

    def formalize(self, text: str) -> Intent:
        raw = _clean_prompt(text)
        if not raw:
            return Intent("unknown", raw, confidence=0.0, metadata={"votes": []})

        votes = [v for v in (_keyword_vote(raw), _structure_vote(raw), _semantic_vote(raw)) if v]
        if not votes:
            return Intent("unknown", raw, confidence=0.0, metadata={"votes": []})

        grouped: dict[str, float] = {}
        for vote in votes:
            grouped[vote.kind] = grouped.get(vote.kind, 0.0) + vote.score

        winner_kind = max(grouped, key=lambda kind: (grouped[kind], max(v.score for v in votes if v.kind == kind)))
        winner = max((v for v in votes if v.kind == winner_kind), key=lambda v: v.score)

        confidence = min(1.0, grouped[winner_kind] / max(1.0, sum(v.score for v in votes)))
        if winner.score >= 0.95:
            confidence = max(confidence, winner.score)

        return Intent(
            kind=winner.kind,
            raw=raw,
            expression=winner.expression,
            variable=winner.variable,
            target=winner.target,
            confidence=round(confidence, 4),
            metadata={
                "votes": [
                    {
                        "kind": v.kind,
                        "score": v.score,
                        "source": v.source,
                    }
                    for v in votes
                ],
                "formalizer": "mesh-v0",
            },
        )
