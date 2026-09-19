from __future__ import annotations

import itertools
from typing import Any

import sympy as sp

from .safe_math import ParsedRelation


def _truth(rel: ParsedRelation, assignment: dict[sp.Symbol, int]) -> bool | None:
    lhs = sp.simplify(rel.lhs.subs(assignment))
    rhs = sp.simplify(rel.rhs.subs(assignment))
    if lhs.free_symbols or rhs.free_symbols:
        return None
    if rel.operator in {"=", "=="}:
        return bool(sp.simplify(lhs - rhs) == 0)
    if rel.operator == "!=":
        return bool(sp.simplify(lhs - rhs) != 0)
    if rel.operator == "<":
        return bool(lhs < rhs)
    if rel.operator == "<=":
        return bool(lhs <= rhs)
    if rel.operator == ">":
        return bool(lhs > rhs)
    if rel.operator == ">=":
        return bool(lhs >= rhs)
    return None


def find_counterexample(rel: ParsedRelation, radius: int = 8, max_points: int = 2000) -> dict[str, Any] | None:
    symbols = rel.symbols
    if not symbols:
        truth = _truth(rel, {})
        return None if truth else {"assignment": {}, "reason": "closed statement is false"}
    if len(symbols) > 3:
        return None

    radius = max(1, min(50, int(radius)))
    values = list(range(-radius, radius + 1))
    checked = 0
    for point in itertools.product(values, repeat=len(symbols)):
        assignment = dict(zip(symbols, point))
        try:
            truth = _truth(rel, assignment)
        except Exception:
            continue
        checked += 1
        if truth is False:
            return {
                "assignment": {sym.name: int(value) for sym, value in assignment.items()},
                "checked_points": checked,
            }
        if checked >= max_points:
            break
    return None
