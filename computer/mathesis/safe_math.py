from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import sympy as sp


_MAX_POWER = 16
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


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
    value = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", value)
    value = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", value)
    value = re.sub(r"(?<=\))(?=[A-Za-z(])", "*", value)
    value = re.sub(r"(?<=[A-Za-z])(?=\()", "*", value)
    return value


def _symbol(name: str) -> sp.Symbol:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid symbol name: {name!r}")
    if name in {"pi", "E"}:
        raise ValueError("reserved mathematical constant used as variable")
    return sp.Symbol(name, real=True)


def _convert(node: ast.AST, symbols: dict[str, sp.Symbol]) -> sp.Expr:
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
        if node.id not in symbols:
            symbols[node.id] = _symbol(node.id)
        return symbols[node.id]

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _convert(node.operand, symbols)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _convert(node.left, symbols)
        right = _convert(node.right, symbols)
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
                raise ValueError("only integer exponents are allowed")
            exp = int(right)
            if abs(exp) > _MAX_POWER:
                raise ValueError(f"exponent magnitude exceeds {_MAX_POWER}")
            return left ** exp

    raise ValueError(f"unsupported mathematical syntax: {type(node).__name__}")


def parse_expr(text: str) -> sp.Expr:
    normalized = normalize_math_text(text)
    if len(normalized) > 2000:
        raise ValueError("expression too long")
    tree = ast.parse(normalized, mode="eval")
    symbols: dict[str, sp.Symbol] = {}
    return sp.simplify(_convert(tree.body, symbols))


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


def parse_relation(text: str) -> ParsedRelation:
    normalized = normalize_math_text(text)
    match = _RELATION.search(normalized)
    if not match:
        raise ValueError("no relation operator found")
    left = normalized[: match.start()].strip()
    right = normalized[match.end() :].strip()
    if not left or not right:
        raise ValueError("relation requires expressions on both sides")
    return ParsedRelation(parse_expr(left), parse_expr(right), match.group("op"))


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
