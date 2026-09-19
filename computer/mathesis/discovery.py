from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import sympy as sp

from .kernel import atomic_json
from .knowledge import KnowledgeGraph, default_state_dir
from .verifiers import CompositeVerifier


def _theorem_id(statement: str) -> str:
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()[:20]


class ConjectureDiscoveryEngine:
    """Bounded autonomous theorem/conjecture laboratory.

    "Novel" here always means new to this MATHESIS knowledge state. The engine
    never claims a result is new to humanity without external scholarly review.
    """

    def __init__(self, state_dir: str | Path | None = None, *, counterexample_radius: int = 10):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()
        self.path = self.state_dir / "discoveries.json"
        self.history_path = self.state_dir / "discovery-history.jsonl"
        self.knowledge = KnowledgeGraph(self.state_dir)
        self.verifier = CompositeVerifier(counterexample_radius=counterexample_radius)

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("cycle", 0)
                value.setdefault("theorems", {})
                return value
        except Exception:
            pass
        return {"version": 1, "cycle": 0, "theorems": {}, "strategy_counts": {}}

    def _save(self, value: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Bound persistent theorem metadata so H24 operation cannot grow without limit.
        theorems = value.get("theorems", {})
        if len(theorems) > 1000:
            newest = sorted(theorems.values(), key=lambda row: row.get("discovered_at", 0), reverse=True)[:1000]
            value["theorems"] = {row["id"]: row for row in newest}
        atomic_json(self.path, value)

    def _append_history(self, row: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        # Keep the autonomous history bounded.
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > 2000:
                self.history_path.write_text("\n".join(lines[-2000:]) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _binomial_identity(self, cycle: int) -> dict[str, Any]:
        exponent = 2 + (cycle % 10)
        x, y = sp.symbols("x y", real=True)
        lhs = (x + y) ** exponent
        rhs = sp.expand(lhs)
        statement = f"(x+y)^{exponent} = {sp.sstr(rhs)}"
        cert = self.verifier.verify_relation(statement)
        return {
            "strategy": "binomial_expansion",
            "statement": statement,
            "certificate": cert.to_dict(),
            "verified": cert.ok,
            "complexity": exponent,
        }

    def _difference_of_powers(self, cycle: int) -> dict[str, Any]:
        exponent = 2 + (cycle % 10)
        x, y = sp.symbols("x y", real=True)
        rhs = (x - y) * sum(x ** (exponent - 1 - j) * y**j for j in range(exponent))
        statement = f"x^{exponent}-y^{exponent} = {sp.sstr(sp.expand(rhs))}"
        cert = self.verifier.verify_relation(statement)
        return {
            "strategy": "difference_of_powers",
            "statement": statement,
            "certificate": cert.to_dict(),
            "verified": cert.ok,
            "complexity": exponent,
        }

    def _sum_of_powers(self, cycle: int) -> dict[str, Any]:
        power = 1 + (cycle % 6)
        n = sp.Symbol("n", integer=True, nonnegative=True)
        sample_count = power + 4
        samples = []
        for value in range(sample_count):
            total = sum(k**power for k in range(1, value + 1))
            samples.append((sp.Integer(value), sp.Integer(total)))

        polynomial = sp.expand(sp.interpolate(samples, n))
        base_ok = sp.expand(polynomial.subs(n, 0)) == 0
        difference = sp.expand(polynomial.subs(n, n + 1) - polynomial)
        target = sp.expand((n + 1) ** power)
        recurrence_statement = f"{sp.sstr(difference)} = {sp.sstr(target)}"
        recurrence_cert = self.verifier.verify_relation(recurrence_statement)
        sample_ok = all(sp.expand(polynomial.subs(n, x) - y) == 0 for x, y in samples)

        statement = f"sum(k^{power}, k=1..n) = {sp.sstr(sp.factor(polynomial))} for integer n>=0"
        verified = bool(base_ok and sample_ok and recurrence_cert.ok)
        return {
            "strategy": "faulhaber_interpolation",
            "statement": statement,
            "certificate": {
                "ok": verified,
                "status": "verified" if verified else "not_fully_verified",
                "methods": [
                    "exact_integer_samples",
                    "sympy_interpolation",
                    "base_case",
                    "finite_difference_induction_step",
                    *list(recurrence_cert.methods),
                ],
                "base_case": base_ok,
                "sample_count": len(samples),
                "recurrence": recurrence_cert.to_dict(),
                "polynomial": str(sp.factor(polynomial)),
            },
            "verified": verified,
            "complexity": power,
        }

    def _geometric_polynomial(self, cycle: int) -> dict[str, Any]:
        terms = 2 + (cycle % 10)
        x = sp.Symbol("x", real=True)
        left = (x - 1) * sum(x**j for j in range(terms))
        right = x**terms - 1
        statement = f"{sp.sstr(sp.expand(left))} = {sp.sstr(right)}"
        cert = self.verifier.verify_relation(statement)
        return {
            "strategy": "finite_geometric_identity",
            "statement": statement,
            "certificate": cert.to_dict(),
            "verified": cert.ok,
            "complexity": terms,
        }

    def discover_once(self) -> dict[str, Any]:
        state = self._load()
        cycle = int(state.get("cycle", 0))
        strategies = (
            self._binomial_identity,
            self._sum_of_powers,
            self._difference_of_powers,
            self._geometric_polynomial,
        )

        selected = None
        # Try several deterministic offsets so a previously-known theorem does not
        # stop a cycle from searching for a new internally-novel result.
        for offset in range(16):
            attempt_cycle = cycle + offset
            strategy = strategies[attempt_cycle % len(strategies)]
            row = strategy(attempt_cycle)
            theorem_id = _theorem_id(row["statement"])
            if theorem_id not in state["theorems"]:
                selected = row
                cycle = attempt_cycle
                break

        if selected is None:
            return {"ok": True, "status": "no_new_candidate_in_budget", "cycle": cycle}

        theorem_id = _theorem_id(selected["statement"])
        now = time.time()
        theorem = {
            "id": theorem_id,
            **selected,
            "internal_novelty": True,
            "human_novelty": "unassessed",
            "discovered_at": now,
        }
        state["cycle"] = cycle + 1
        state["theorems"][theorem_id] = theorem
        strategy_counts = state.setdefault("strategy_counts", {})
        strategy_counts[selected["strategy"]] = int(strategy_counts.get(selected["strategy"], 0)) + 1
        state["updated_at"] = now
        self._save(state)
        self._append_history(theorem)

        if selected["verified"]:
            self.knowledge.record(
                selected["statement"],
                status="verified",
                certificate=selected["certificate"],
                dependencies=[],
            )

        return {
            "ok": bool(selected["verified"]),
            "status": "verified_discovery" if selected["verified"] else "rejected_conjecture",
            "theorem": theorem,
            "total_discoveries": len(state["theorems"]),
        }

    def status(self) -> dict[str, Any]:
        state = self._load()
        verified = sum(1 for row in state["theorems"].values() if row.get("verified"))
        rejected = len(state["theorems"]) - verified
        return {
            "ok": True,
            "cycle": int(state.get("cycle", 0)),
            "stored": len(state["theorems"]),
            "verified": verified,
            "rejected": rejected,
            "strategies": state.get("strategy_counts", {}),
            "novelty_policy": "internal novelty only; human novelty is never inferred automatically",
            "path": str(self.path),
        }
