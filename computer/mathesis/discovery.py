from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import sympy as sp

from .kernel import atomic_json
from .knowledge import KnowledgeGraph, default_state_dir
from .safe_math import parse_expr, parse_relation
from .verifiers import CompositeVerifier


def _theorem_id(statement: str) -> str:
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()[:20]


DISCOVERY_SCHEMA_VERSION = 3
MAX_FAULHABER_POWER = 8
MAX_REVALIDATION_SAMPLES = 64
PROOF_QUALITY_GATE = "structurally_nontrivial_and_proof_gated"


def _relation_is_structurally_nontrivial(statement: str) -> bool:
    """Reject relations whose two sides are already the same expression tree."""
    try:
        rel = parse_relation(statement)
    except Exception:
        return True
    return sp.srepr(rel.lhs) != sp.srepr(rel.rhs)


def _faulhaber_reproof(
    row: dict[str, Any],
    verifier: CompositeVerifier,
) -> tuple[bool, dict[str, Any], str]:
    """Rebuild a stored Faulhaber proof from data, never from trusted metadata."""
    cert = dict(row.get("certificate") or {})
    polynomial_text = str(cert.get("polynomial", "")).strip()
    try:
        power = int(row.get("complexity", 0) or 0)
    except Exception:
        return False, {}, "invalid_power"

    if not polynomial_text or not (1 <= power <= MAX_FAULHABER_POWER):
        return False, {}, "unsupported_or_missing_polynomial"

    try:
        polynomial = parse_expr(polynomial_text)
        symbols = sorted(polynomial.free_symbols, key=lambda s: s.name)
        n = next((symbol for symbol in symbols if symbol.name == "n"), None)
        if n is None or any(symbol.name != "n" for symbol in symbols):
            return False, {}, "unexpected_polynomial_symbols"

        match = re.fullmatch(
            r"\s*sum\(k\^(\d+),\s*k=1\.\.n\)\s*=\s*(.+?)\s+for integer n>=0\s*",
            str(row.get("statement", "")),
        )
        if match is None or int(match.group(1)) != power:
            return False, {}, "statement_power_mismatch"
        stated_rhs = parse_expr(match.group(2))
        if sp.simplify(stated_rhs - polynomial) != 0:
            return False, {}, "statement_polynomial_mismatch"

        factored = sp.factor(polynomial)
        shifted = factored.subs(n, n + 1)
        recurrence_statement = (
            f"({sp.sstr(shifted)})-({sp.sstr(factored)}) = (n+1)^{power}"
        )
        recurrence_nontrivial = _relation_is_structurally_nontrivial(recurrence_statement)
        recurrence_cert = verifier.verify_relation(recurrence_statement)
        base_ok = sp.expand(polynomial.subs(n, 0)) == 0

        try:
            requested_samples = int(cert.get("sample_count", 0) or 0)
        except Exception:
            requested_samples = 0
        sample_count = min(
            MAX_REVALIDATION_SAMPLES,
            max(power + 4, max(0, requested_samples)),
        )
        sample_ok = all(
            sp.expand(
                polynomial.subs(n, value)
                - sum(k**power for k in range(1, value + 1))
            ) == 0
            for value in range(sample_count)
        )
    except Exception as exc:
        return False, {}, f"reproof_error:{type(exc).__name__}"

    verified = bool(base_ok and sample_ok and recurrence_nontrivial and recurrence_cert.ok)
    if not verified:
        return False, {}, "proof_obligation_failed"

    upgraded = {
        "ok": True,
        "status": "verified",
        "base_case": bool(base_ok),
        "sample_count": sample_count,
        "recurrence_nontrivial": True,
        "recurrence": recurrence_cert.to_dict(),
        "polynomial": str(factored),
    }
    return True, upgraded, "verified"


def validate_verified_discovery(
    row: dict[str, Any],
    verifier: CompositeVerifier | None = None,
) -> tuple[bool, str]:
    """Re-check an active VERIFIED discovery from mathematical content."""
    if row.get("verified") is not True:
        return False, "row_not_verified"

    cert = row.get("certificate") or {}
    if cert.get("ok") is not True:
        return False, "certificate_not_ok"
    if row.get("quality_gate") != PROOF_QUALITY_GATE:
        return False, "missing_current_quality_gate"

    verifier = verifier or CompositeVerifier()
    if row.get("strategy") == "faulhaber_interpolation":
        ok, _upgrade, reason = _faulhaber_reproof(row, verifier)
        return ok, reason

    statement = str(row.get("statement", "")).strip()
    try:
        parse_relation(statement)
    except Exception:
        return False, "unsupported_verified_schema"
    if not _relation_is_structurally_nontrivial(statement):
        return False, "structural_tautology"
    try:
        current = verifier.verify_relation(statement)
    except Exception as exc:
        return False, f"verification_error:{type(exc).__name__}"
    return bool(current.ok), "verified" if current.ok else f"current_verifier:{current.status}"


class ConjectureDiscoveryEngine:
    """Bounded autonomous theorem/conjecture laboratory.

    "Novel" here always means new to this MATHESIS knowledge state. The engine
    never claims a result is new to humanity without external scholarly review.
    """

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        counterexample_radius: int = 10,
        symbolic_depth: int = 2,
        discovery_beam: int = 1,
    ):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()
        self.path = self.state_dir / "discoveries.json"
        self.history_path = self.state_dir / "discovery-history.jsonl"
        self.knowledge = KnowledgeGraph(self.state_dir)
        self.verifier = CompositeVerifier(counterexample_radius=counterexample_radius)
        self.symbolic_depth = max(1, min(12, int(symbolic_depth)))
        self.discovery_beam = max(1, min(8, int(discovery_beam)))

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("cycle", 0)
                value.setdefault("theorems", {})
                value.setdefault("discarded", {})
                return value
        except Exception:
            pass
        return {"version": DISCOVERY_SCHEMA_VERSION, "cycle": 0, "theorems": {}, "discarded": {}, "strategy_counts": {}}

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
        max_degree = min(16, 3 + self.symbolic_depth)
        exponent = 2 + (cycle % max(1, max_degree - 1))
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
        max_degree = min(16, 3 + self.symbolic_depth)
        exponent = 2 + (cycle % max(1, max_degree - 1))
        x, y = sp.symbols("x y", real=True)
        series = sum(x ** (exponent - 1 - j) * y**j for j in range(exponent))
        rhs = (x - y) * series
        statement = f"x^{exponent}-y^{exponent} = ({sp.sstr(x-y)})*({sp.sstr(series)})"
        cert = self.verifier.verify_relation(statement)
        return {
            "strategy": "difference_of_powers",
            "statement": statement,
            "certificate": cert.to_dict(),
            "verified": cert.ok,
            "complexity": exponent,
        }

    def _sum_of_powers(self, cycle: int) -> dict[str, Any]:
        max_power = min(8, max(2, 1 + self.symbolic_depth // 2))
        power = 1 + (cycle % max_power)
        n = sp.Symbol("n", integer=True, nonnegative=True)
        sample_count = power + 4
        samples = []
        for value in range(sample_count):
            total = sum(k**power for k in range(1, value + 1))
            samples.append((sp.Integer(value), sp.Integer(total)))

        polynomial = sp.expand(sp.interpolate(samples, n))
        base_ok = sp.expand(polynomial.subs(n, 0)) == 0
        factored_polynomial = sp.factor(polynomial)
        shifted_factored = factored_polynomial.subs(n, n + 1)
        recurrence_statement = (
            f"({sp.sstr(shifted_factored)})-({sp.sstr(factored_polynomial)}) = (n+1)^{power}"
        )
        recurrence_nontrivial = _relation_is_structurally_nontrivial(recurrence_statement)
        recurrence_cert = self.verifier.verify_relation(recurrence_statement)
        sample_ok = all(sp.expand(polynomial.subs(n, x) - y) == 0 for x, y in samples)

        statement = f"sum(k^{power}, k=1..n) = {sp.sstr(sp.factor(polynomial))} for integer n>=0"
        verified = bool(base_ok and sample_ok and recurrence_nontrivial and recurrence_cert.ok)
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
                "recurrence_nontrivial": bool(recurrence_nontrivial),
                "polynomial": str(factored_polynomial),
            },
            "verified": verified,
            "complexity": power,
        }

    def _geometric_polynomial(self, cycle: int) -> dict[str, Any]:
        max_terms = min(16, 3 + self.symbolic_depth)
        terms = 2 + (cycle % max(1, max_terms - 1))
        x = sp.Symbol("x", real=True)
        series = sum(x**j for j in range(terms))
        left = (x - 1) * series
        right = x**terms - 1
        statement = f"(x-1)*({sp.sstr(series)}) = x^{terms}-1"
        cert = self.verifier.verify_relation(statement)
        return {
            "strategy": "finite_geometric_identity",
            "statement": statement,
            "certificate": cert.to_dict(),
            "verified": cert.ok,
            "complexity": terms,
        }

    def _revalidate_legacy_faulhaber(self, row: dict[str, Any]) -> bool:
        ok, upgraded, _reason = _faulhaber_reproof(row, self.verifier)
        if not ok:
            return False

        cert = dict(row.get("certificate") or {})
        cert.update(upgraded)
        row["certificate"] = cert
        row["nontrivial"] = True
        row["quality_gate"] = PROOF_QUALITY_GATE
        row["revalidated_at"] = time.time()
        return True

    def _revalidate_legacy_relation(self, row: dict[str, Any]) -> bool:
        statement = str(row.get("statement", "")).strip()
        if not statement or not _relation_is_structurally_nontrivial(statement):
            return False
        try:
            parse_relation(statement)
            current = self.verifier.verify_relation(statement)
        except Exception:
            return False
        if not current.ok:
            return False
        row["certificate"] = current.to_dict()
        row["verified"] = True
        row["nontrivial"] = True
        row["quality_gate"] = PROOF_QUALITY_GATE
        row["revalidated_at"] = time.time()
        return True

    def _audit_legacy_discoveries(self, state: dict[str, Any]) -> dict[str, Any]:
        theorems = state.setdefault("theorems", {})
        discarded = state.setdefault("discarded", {})
        moved: list[str] = []
        revalidated: list[str] = []
        try:
            previous_version = int(state.get("version", 1) or 1)
        except Exception:
            previous_version = 1
        schema_upgraded = previous_version < DISCOVERY_SCHEMA_VERSION
        state["version"] = max(DISCOVERY_SCHEMA_VERSION, previous_version)

        for theorem_id, row in list(theorems.items()):
            if not row.get("verified"):
                continue

            strategy = str(row.get("strategy", ""))
            cert = row.get("certificate") or {}
            needs_upgrade = (
                row.get("quality_gate") != PROOF_QUALITY_GATE
                or cert.get("ok") is not True
            )

            if strategy == "faulhaber_interpolation":
                needs_upgrade = needs_upgrade or cert.get("recurrence_nontrivial") is not True
                if needs_upgrade:
                    if self._revalidate_legacy_faulhaber(row):
                        revalidated.append(theorem_id)
                        continue
                    reason = "legacy_faulhaber_reproof_failed"
                else:
                    ok, _reason = validate_verified_discovery(row, self.verifier)
                    if ok:
                        continue
                    reason = "legacy_faulhaber_reproof_failed"
            else:
                statement = str(row.get("statement", "")).strip()
                if not _relation_is_structurally_nontrivial(statement):
                    reason = "structural_tautology"
                elif needs_upgrade:
                    if self._revalidate_legacy_relation(row):
                        revalidated.append(theorem_id)
                        continue
                    reason = "legacy_relation_reproof_failed"
                else:
                    ok, validation_reason = validate_verified_discovery(row, self.verifier)
                    if ok:
                        continue
                    reason = (
                        "structural_tautology"
                        if validation_reason == "structural_tautology"
                        else "verified_discovery_revalidation_failed"
                    )

            archived = dict(row)
            archived["discarded_reason"] = reason
            archived["discarded_at"] = time.time()
            discarded[theorem_id] = archived
            del theorems[theorem_id]
            moved.append(theorem_id)

        if len(discarded) > 500:
            newest = sorted(
                discarded.values(),
                key=lambda row: row.get("discarded_at", row.get("discovered_at", 0)),
                reverse=True,
            )[:500]
            state["discarded"] = {row["id"]: row for row in newest}

        if moved or revalidated or schema_upgraded:
            state["last_quality_migration"] = {
                "at": time.time(),
                "moved": moved,
                "revalidated": revalidated,
                "schema_upgraded": schema_upgraded,
                "reason": "proof_quality_upgrade",
            }
            self._save(state)
        return state

    def discover_once(self) -> dict[str, Any]:
        state = self._audit_legacy_discoveries(self._load())
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
        for offset in range(max(4, self.discovery_beam * 4)):
            attempt_cycle = cycle + offset
            strategy = strategies[attempt_cycle % len(strategies)]
            row = strategy(attempt_cycle)
            row["nontrivial"] = _relation_is_structurally_nontrivial(row["statement"])
            theorem_id = _theorem_id(row["statement"])
            if (
                theorem_id not in state["theorems"]
                and theorem_id not in state.get("discarded", {})
                and row["nontrivial"]
            ):
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
            "quality_gate": PROOF_QUALITY_GATE,
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
            "discarded": len(state.get("discarded", {})),
            "strategies": state.get("strategy_counts", {}),
            "novelty_policy": "internal novelty only; human novelty is never inferred automatically",
            "symbolic_depth": self.symbolic_depth,
            "discovery_beam": self.discovery_beam,
            "path": str(self.path),
        }
