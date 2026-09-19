from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .counterexample import find_counterexample
from .safe_math import ParsedRelation, exact_value, parse_expr, parse_relation
from .types import VerificationCertificate

try:
    import z3
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    z3 = None


def _z3_expr(expr: sp.Expr, env: dict[str, Any]):
    if z3 is None:
        raise RuntimeError("z3 is unavailable")
    if expr.is_Integer:
        return z3.IntVal(int(expr))
    if expr.is_Rational:
        return z3.RealVal(f"{int(expr.p)}/{int(expr.q)}")
    if expr.is_Symbol:
        name = str(expr)
        if name not in env:
            env[name] = z3.Real(name)
        return env[name]
    if expr.is_Add:
        args = [_z3_expr(arg, env) for arg in expr.args]
        return sum(args[1:], args[0]) if args else z3.IntVal(0)
    if expr.is_Mul:
        out = z3.IntVal(1)
        for arg in expr.args:
            out = out * _z3_expr(arg, env)
        return out
    if expr.is_Pow:
        base, exponent = expr.args
        if not exponent.is_Integer or int(exponent) < 0 or int(exponent) > 16:
            raise ValueError("z3 adapter only supports non-negative integer powers <= 16")
        out = z3.IntVal(1)
        zbase = _z3_expr(base, env)
        for _ in range(int(exponent)):
            out = out * zbase
        return out
    raise ValueError(f"unsupported sympy node for z3: {expr}")


def _z3_relation(rel: ParsedRelation, env: dict[str, Any]):
    left = _z3_expr(rel.lhs, env)
    right = _z3_expr(rel.rhs, env)
    if rel.operator in {"=", "=="}:
        return left == right
    if rel.operator == "!=":
        return left != right
    if rel.operator == "<":
        return left < right
    if rel.operator == "<=":
        return left <= right
    if rel.operator == ">":
        return left > right
    if rel.operator == ">=":
        return left >= right
    raise ValueError(f"unsupported operator: {rel.operator}")


def _lean_nat_expr(expr: sp.Expr) -> str:
    expr = sp.expand(expr)
    if expr.is_Integer:
        value = int(expr)
        if value < 0:
            raise ValueError("negative Nat literal")
        return str(value)
    if expr.is_Add:
        return "(" + " + ".join(_lean_nat_expr(arg) for arg in expr.args) + ")"
    if expr.is_Mul:
        return "(" + " * ".join(_lean_nat_expr(arg) for arg in expr.args) + ")"
    if expr.is_Pow:
        base, exponent = expr.args
        if not exponent.is_Integer or int(exponent) < 0 or int(exponent) > 16:
            raise ValueError("unsupported Lean Nat exponent")
        return f"({_lean_nat_expr(base)} ^ {int(exponent)})"
    raise ValueError("Lean adapter only accepts closed Nat arithmetic")


@dataclass
class LeanVerifier:
    timeout_seconds: int = 10

    @property
    def available(self) -> bool:
        return shutil.which("lean") is not None

    def verify_closed_equality(self, rel: ParsedRelation) -> dict[str, Any]:
        if rel.operator not in {"=", "=="} or rel.symbols:
            return {"ok": False, "status": "not_applicable", "reason": "requires closed equality"}
        if not self.available:
            return {"ok": False, "status": "unavailable"}

        try:
            lhs = _lean_nat_expr(rel.lhs)
            rhs = _lean_nat_expr(rel.rhs)
        except Exception as exc:
            return {"ok": False, "status": "not_applicable", "reason": repr(exc)}

        source = f"example : ({lhs} : Nat) = ({rhs} : Nat) := by decide\n"
        with tempfile.TemporaryDirectory(prefix="mathesis-lean-") as tmp:
            path = os.path.join(tmp, "Check.lean")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", tmp),
                "LANG": "C.UTF-8",
            }
            try:
                proc = subprocess.run(
                    ["lean", path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    cwd=tmp,
                )
            except Exception as exc:
                return {"ok": False, "status": "error", "reason": repr(exc)}
        return {
            "ok": proc.returncode == 0,
            "status": "verified" if proc.returncode == 0 else "rejected",
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "source": source,
        }


class SymbolicVerifier:
    def verify_relation(self, rel: ParsedRelation) -> dict[str, Any]:
        if rel.operator in {"=", "=="}:
            delta = sp.simplify(rel.lhs - rel.rhs)
            if delta == 0:
                return {"ok": True, "status": "verified", "method": "sympy_normal_form", "normal_form": "0"}
            if not rel.symbols:
                return {
                    "ok": False,
                    "status": "disproved",
                    "method": "sympy_exact",
                    "normal_form": str(delta),
                }
            return {"ok": False, "status": "unknown", "method": "sympy_normal_form", "normal_form": str(delta)}

        if not rel.symbols:
            value = bool(rel.to_sympy())
            return {
                "ok": value,
                "status": "verified" if value else "disproved",
                "method": "sympy_exact",
            }
        return {"ok": False, "status": "unknown", "method": "sympy"}


class SMTVerifier:
    def verify_relation(self, rel: ParsedRelation) -> dict[str, Any]:
        if z3 is None:
            return {"ok": False, "status": "unavailable", "method": "z3"}
        try:
            env: dict[str, Any] = {}
            formula = _z3_relation(rel, env)
            solver = z3.Solver()
            solver.add(z3.Not(formula))
            result = solver.check()
            if result == z3.unsat:
                return {"ok": True, "status": "verified", "method": "z3_unsat_negation"}
            if result == z3.sat:
                model = solver.model()
                counterexample = {str(decl): str(model[decl]) for decl in model.decls()}
                return {
                    "ok": False,
                    "status": "disproved",
                    "method": "z3_model",
                    "counterexample": counterexample,
                }
            return {"ok": False, "status": "unknown", "method": "z3", "reason": str(result)}
        except Exception as exc:
            return {"ok": False, "status": "unsupported", "method": "z3", "reason": repr(exc)}


class CompositeVerifier:
    """Proof-gated verifier.

    Creative components may propose anything; this class is the acceptance
    boundary. A statement is called VERIFIED only when at least two independent
    proof/check paths agree, except equation solution-set verification which
    uses exact solve + substitution + SMT completeness when supported.
    """

    def __init__(self, *, counterexample_radius: int = 8):
        self.symbolic = SymbolicVerifier()
        self.smt = SMTVerifier()
        self.lean = LeanVerifier()
        self.counterexample_radius = counterexample_radius

    def verify_relation(self, statement: str) -> VerificationCertificate:
        rel = parse_relation(statement)
        symbolic = self.symbolic.verify_relation(rel)
        smt = self.smt.verify_relation(rel)
        lean = self.lean.verify_closed_equality(rel)
        counterexample = find_counterexample(rel, radius=self.counterexample_radius)

        methods: list[str] = []
        if symbolic.get("ok"):
            methods.append(str(symbolic.get("method", "symbolic")))
        if smt.get("ok"):
            methods.append(str(smt.get("method", "smt")))
        if lean.get("ok"):
            methods.append("lean_kernel")

        if counterexample is not None:
            return VerificationCertificate(
                ok=False,
                status="disproved",
                statement=statement,
                methods=tuple(methods),
                counterexample=counterexample,
                details={"symbolic": symbolic, "smt": smt, "lean": lean},
            )

        if len(methods) >= 2:
            return VerificationCertificate(
                ok=True,
                status="verified",
                statement=statement,
                methods=tuple(methods),
                details={"symbolic": symbolic, "smt": smt, "lean": lean},
            )

        if symbolic.get("status") == "disproved" or smt.get("status") == "disproved":
            return VerificationCertificate(
                ok=False,
                status="disproved",
                statement=statement,
                methods=tuple(methods),
                counterexample=smt.get("counterexample"),
                details={"symbolic": symbolic, "smt": smt, "lean": lean},
            )

        return VerificationCertificate(
            ok=False,
            status="not_fully_verified",
            statement=statement,
            methods=tuple(methods),
            details={"symbolic": symbolic, "smt": smt, "lean": lean},
        )

    def evaluate(self, expression: str) -> tuple[Any, VerificationCertificate]:
        expr = parse_expr(expression)
        if expr.free_symbols:
            raise ValueError("evaluation requires a closed expression")
        result = exact_value(expr)

        if isinstance(result, dict) and {"numerator", "denominator"} <= set(result):
            right = f"({result['numerator']})/({result['denominator']})"
        else:
            right = str(result)
        certificate = self.verify_relation(f"{expression} = {right}")
        return result, certificate

    def solve_equation(self, statement: str, variable: str | None = None) -> tuple[list[Any], VerificationCertificate]:
        rel = parse_relation(statement)
        if rel.operator not in {"=", "=="}:
            raise ValueError("equation solver requires equality")

        symbols = rel.symbols
        if variable:
            target = next((sym for sym in symbols if sym.name == variable), sp.Symbol(variable, real=True))
        elif len(symbols) == 1:
            target = symbols[0]
        else:
            raise ValueError("specify exactly one variable to solve for")

        equation = sp.Eq(rel.lhs, rel.rhs)
        solutions = list(sp.solve(equation, target))
        substitution_ok = all(sp.simplify(rel.lhs.subs(target, sol) - rel.rhs.subs(target, sol)) == 0 for sol in solutions)

        completeness = {"ok": False, "status": "unsupported"}
        if z3 is not None and solutions and all(sol.is_Rational for sol in solutions):
            try:
                env: dict[str, Any] = {}
                formula = _z3_relation(rel, env)
                ztarget = env.get(target.name)\n                if ztarget is None:\n                    ztarget = z3.Real(target.name)
                allowed = z3.Or(*[
                    ztarget == z3.RealVal(f"{int(sol.p)}/{int(sol.q)}")
                    for sol in solutions
                ])
                solver = z3.Solver()
                solver.add(formula)
                solver.add(z3.Not(allowed))
                check = solver.check()
                completeness = {
                    "ok": check == z3.unsat,
                    "status": "verified" if check == z3.unsat else str(check),
                }
            except Exception as exc:
                completeness = {"ok": False, "status": "unsupported", "reason": repr(exc)}

        methods = ["sympy_exact_solve"]
        if substitution_ok:
            methods.append("exact_substitution")
        if completeness.get("ok"):
            methods.append("z3_solution_set_completeness")

        ok = bool(solutions) and substitution_ok and completeness.get("ok", False)
        cert = VerificationCertificate(
            ok=ok,
            status="verified" if ok else "checked_not_complete",
            statement=statement,
            methods=tuple(methods),
            details={
                "solutions": [str(sol) for sol in solutions],
                "substitution_ok": substitution_ok,
                "completeness": completeness,
            },
        )
        rendered = [exact_value(sol) for sol in solutions]
        return rendered, cert

    def diagnostics(self) -> dict[str, Any]:
        return {
            "sympy": sp.__version__,
            "z3_available": z3 is not None,
            "lean_available": self.lean.available,
        }
