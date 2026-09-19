from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import sympy as sp


_MAX_POWER = 32
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "factorial": sp.factorial,
    "gamma": sp.gamma,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "binomial": sp.binomial,
}


def normalize_math_text(text: str) -> str:
    value = str(text or "").strip()
    value = (
        value.replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
        .replace("π", "pi")
    )
    # Preserve whitelisted function calls like sin(x); implicit multiplication is
    # only inserted in unambiguous numeric/parenthesized positions.
    value = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", value)
    value = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", value)
    value = re.sub(r"(?<=\))(?=[A-Za-z(])", "*", value)
    return value


def _symbol(name: str, assumptions: dict[str, dict[str, bool]] | None = None) -> sp.Symbol:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid symbol name: {name!r}")
    if name in {"pi", "E", "I", "oo"} or name in _FUNCTIONS:
        raise ValueError("reserved mathematical name used as variable")
    kwargs: dict[str, bool] = {"real": True}
    if assumptions and name in assumptions:
        for key, value in assumptions[name].items():
            if key not in {
                "real", "integer", "positive", "nonnegative", "negative",
                "nonpositive", "rational", "finite", "nonzero",
            }:
                raise ValueError(f"unsupported assumption: {key}")
            kwargs[key] = bool(value)
        if kwargs.get("integer"):
            kwargs.setdefault("real", True)
    return sp.Symbol(name, **kwargs)


def _convert(
    node: ast.AST,
    symbols: dict[str, sp.Symbol],
    assumptions: dict[str, dict[str, bool]] | None,
) -> sp.Expr:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        if isinstance(node.value, float):
            return sp.Rational(str(node.value))
        return sp.Integer(node.value)

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi
        if node.id == "E":
            return sp.E
        if node.id == "I":
            return sp.I
        if node.id == "oo":
            return sp.oo
        if node.id not in symbols:
            symbols[node.id] = _symbol(node.id, assumptions)
        return symbols[node.id]

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _convert(node.operand, symbols, assumptions)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _convert(node.left, symbols, assumptions)
        right = _convert(node.right, symbols, assumptions)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if not right.is_Integer:
                # Symbolic roots are expressed through the sqrt whitelist.
                raise ValueError("only integer exponents are allowed; use sqrt() for square roots")
            exp = int(right)
            if abs(exp) > _MAX_POWER:
                raise ValueError(f"exponent magnitude exceeds {_MAX_POWER}")
            return left ** exp

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("only whitelisted mathematical functions are allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in mathematical functions")
        if len(node.args) < 1 or len(node.args) > 2:
            raise ValueError("mathematical function arity is outside the safe whitelist")
        fn = _FUNCTIONS[node.func.id]
        args = [_convert(arg, symbols, assumptions) for arg in node.args]
        return fn(*args)

    raise ValueError(f"unsupported mathematical syntax: {type(node).__name__}")


def parse_expr(
    text: str,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> sp.Expr:
    normalized = normalize_math_text(text)
    if len(normalized) > 4000:
        raise ValueError("expression too long")
    tree = ast.parse(normalized, mode="eval")
    symbols: dict[str, sp.Symbol] = {}
    return _convert(tree.body, symbols, assumptions)


@dataclass(frozen=True)
class ParsedRelation:
    lhs: sp.Expr
    rhs: sp.Expr
    operator: str

    @property
    def symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(sorted(self.lhs.free_symbols | self.rhs.free_symbols, key=lambda s: s.name))

    def to_sympy(self):
        if self.operator in {"=", "=="}:
            return sp.Eq(self.lhs, self.rhs)
        if self.operator == "!=":
            return sp.Ne(self.lhs, self.rhs)
        if self.operator == "<":
            return sp.Lt(self.lhs, self.rhs)
        if self.operator == "<=":
            return sp.Le(self.lhs, self.rhs)
        if self.operator == ">":
            return sp.Gt(self.lhs, self.rhs)
        if self.operator == ">=":
            return sp.Ge(self.lhs, self.rhs)
        raise ValueError(f"unsupported relation operator: {self.operator}")


_RELATION = re.compile(r"(?<![<>!=])(?P<op><=|>=|==|!=|=|<|>)(?![=])")


def parse_relation(
    text: str,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> ParsedRelation:
    normalized = normalize_math_text(text)
    match = _RELATION.search(normalized)
    if not match:
        raise ValueError("no relation operator found")
    left = normalized[: match.start()].strip()
    right = normalized[match.end() :].strip()
    if not left or not right:
        raise ValueError("relation requires expressions on both sides")
    return ParsedRelation(parse_expr(left, assumptions), parse_expr(right, assumptions), match.group("op"))


def exact_value(expr: sp.Expr) -> Any:
    value = sp.simplify(expr)
    if value.free_symbols:
        return str(value)
    if value.is_Integer:
        return int(value)
    if value.is_Rational:
        if int(value.q) == 1:
            return int(value.p)
        return {"numerator": int(value.p), "denominator": int(value.q)}
    return str(value)


def substitute_exact(expr: sp.Expr, assignment: dict[str, int]) -> sp.Expr:
    mapping = {sp.Symbol(k, real=True): sp.Integer(v) for k, v in assignment.items()}
    return sp.simplify(expr.subs(mapping))
