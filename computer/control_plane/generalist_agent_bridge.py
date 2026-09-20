from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Any

from generalist_lm.agent import GeneralistAgent

from . import generalist_provider

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric value required")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("finite numeric value required")
    return number


def _eval_arithmetic(expression: str) -> int | float:
    tree = ast.parse(str(expression), mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
                raise ValueError("exponent is too large")
            result = _ALLOWED_BINOPS[type(node.op)](left, right)
            if abs(float(result)) > 1e100:
                raise ValueError("arithmetic result is too large")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](walk(node.operand))
        raise ValueError("unsupported arithmetic expression")

    return walk(tree)


def _data_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    values_raw = arguments.get("values")
    if not isinstance(values_raw, list) or not values_raw or len(values_raw) > 10_000:
        raise ValueError("values must be a non-empty list of at most 10000 numbers")
    values = [_safe_number(value) for value in values_raw]
    operation = str(arguments.get("operation", "mean")).strip().lower()
    if operation == "mean":
        result = statistics.fmean(values)
    elif operation == "median":
        result = statistics.median(values)
    elif operation == "sum":
        result = sum(values)
    elif operation == "min":
        result = min(values)
    elif operation == "max":
        result = max(values)
    elif operation == "count":
        result = len(values)
    else:
        raise ValueError("unsupported data_stats operation")
    return {"operation": operation, "result": result, "count": len(values)}

def _table_rows(arguments: dict[str, Any], *, max_rows: int = 5000) -> list[dict[str, Any]]:
    rows = arguments.get("rows")
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ValueError(f"rows must be a list of at most {max_rows} objects")
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each table row must be an object")
        if len(row) > 128:
            raise ValueError("table row has too many columns")
        cleaned.append({str(key)[:200]: value for key, value in row.items()})
    return cleaned


def _table_profile(arguments: dict[str, Any]) -> dict[str, Any]:
    rows = _table_rows(arguments, max_rows=2000)
    columns = sorted({key for row in rows for key in row})[:128]
    profile: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        non_null = [value for value in values if value is not None]
        numeric = [
            _safe_number(value)
            for value in non_null
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        row: dict[str, Any] = {
            "count": len(non_null),
            "missing": len(values) - len(non_null),
            "numeric_count": len(numeric),
        }
        if numeric:
            row.update({
                "min": min(numeric),
                "max": max(numeric),
                "mean": statistics.fmean(numeric),
                "median": statistics.median(numeric),
            })
        profile[column] = row
    return {"row_count": len(rows), "columns": columns, "profile": profile}


def _aggregate_values(values: list[float], operation: str) -> int | float:
    if operation == "count":
        return len(values)
    if not values:
        raise ValueError("no numeric values available for aggregation")
    if operation == "sum":
        return sum(values)
    if operation == "mean":
        return statistics.fmean(values)
    if operation == "median":
        return statistics.median(values)
    if operation == "min":
        return min(values)
    if operation == "max":
        return max(values)
    raise ValueError("unsupported table aggregation operation")


def _table_aggregate(arguments: dict[str, Any]) -> dict[str, Any]:
    rows = _table_rows(arguments)
    operation = str(arguments.get("operation", "count")).strip().lower()
    if operation not in {"count", "sum", "mean", "median", "min", "max"}:
        raise ValueError("unsupported table aggregation operation")
    column_raw = arguments.get("column")
    column = str(column_raw) if column_raw is not None else None
    group_raw = arguments.get("group_by")
    group_by = str(group_raw) if group_raw is not None else None

    def selected(group_rows: list[dict[str, Any]]) -> list[float]:
        if operation == "count" and not column:
            return [1.0] * len(group_rows)
        if not column:
            raise ValueError("numeric aggregation requires a column")
        values: list[float] = []
        for row in group_rows:
            value = row.get(column)
            if value is None:
                continue
            values.append(_safe_number(value))
        return values

    if not group_by:
        values = selected(rows)
        result = len(rows) if operation == "count" and not column else _aggregate_values(values, operation)
        return {
            "operation": operation,
            "column": column,
            "group_by": None,
            "result": result,
            "row_count": len(rows),
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_by))
        groups.setdefault(key, []).append(row)
        if len(groups) > 200:
            raise ValueError("too many groups")

    result: dict[str, int | float] = {}
    for key in sorted(groups):
        group_rows = groups[key]
        values = selected(group_rows)
        result[key] = (
            len(group_rows)
            if operation == "count" and not column
            else _aggregate_values(values, operation)
        )
    return {
        "operation": operation,
        "column": column,
        "group_by": group_by,
        "groups": result,
        "row_count": len(rows),
    }



def tool_specs() -> dict[str, dict[str, Any]]:
    return {
        "calculator": {
            "description": "Evaluate bounded arithmetic only.",
            "schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        "data_stats": {
            "description": "Compute mean, median, sum, min, max or count over numeric values.",
            "schema": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "operation": {"type": "string"},
                },
                "required": ["values", "operation"],
            },
        },
        "table_profile": {
            "description": "Profile JSON table rows: columns, missing values and numeric summary statistics.",
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["rows"],
            },
        },
        "table_aggregate": {
            "description": "Aggregate JSON table rows with count, sum, mean, median, min or max, optionally grouped by a column.",
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}},
                    "operation": {"type": "string"},
                    "column": {"type": "string"},
                    "group_by": {"type": "string"},
                },
                "required": ["rows", "operation"],
            },
        },
        "file_read": {
            "description": "Read a text file inside the Airi-PC workspace.",
            "schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        "file_search": {
            "description": "Search text inside the Airi-PC workspace.",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        "project_analyze": {
            "description": "Inspect project languages, configs, tests and tree without modifying files.",
            "schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }


def execute_readonly_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "calculator":
        expression = str(arguments.get("expression", ""))
        if len(expression) > 500:
            raise ValueError("calculator expression is too long")
        return {"value": _eval_arithmetic(expression)}
    if name == "data_stats":
        return _data_stats(arguments)
    if name == "table_profile":
        return _table_profile(arguments)
    if name == "table_aggregate":
        return _table_aggregate(arguments)

    from coding import analyze, read, search

    if name == "file_read":
        return read(str(arguments.get("path", "")))
    if name == "file_search":
        return search(
            str(arguments.get("query", "")),
            str(arguments.get("path", ".")),
            limit=100,
        )
    if name == "project_analyze":
        return analyze(str(arguments.get("path", ".")))
    raise PermissionError(f"tool is not allowlisted: {name}")


def run_generalist_agent(
    messages: list[dict[str, str]],
    *,
    max_steps: int = 8,
    max_new_tokens: int = 512,
):
    status = generalist_provider.status()
    if not status.get("available"):
        raise RuntimeError(status.get("reason", "generalist provider unavailable"))
    backend = generalist_provider._backend()
    agent = GeneralistAgent(
        backend,
        tools=tool_specs(),
        executor=execute_readonly_tool,
        max_steps=max_steps,
    )
    return agent.run(messages, max_new_tokens=max_new_tokens)
