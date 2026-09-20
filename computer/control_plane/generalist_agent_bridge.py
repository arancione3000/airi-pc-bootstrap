from __future__ import annotations

import ast
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
    return float(value)


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
