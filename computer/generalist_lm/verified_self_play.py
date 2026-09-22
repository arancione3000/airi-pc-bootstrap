from __future__ import annotations

import ast
import json
import math
import operator
import random
from typing import Any

from .curriculum import ResearchRow

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expression: str) -> int | float:
    if not isinstance(expression, str) or not expression or len(expression) > 80:
        raise ValueError("invalid expression")
    tree = ast.parse(expression, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("non-numeric constant")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            result = _BINOPS[type(node.op)](left, right)
            if abs(float(result)) > 1_000_000:
                raise ValueError("result out of bounds")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    result = walk(tree)
    if not math.isfinite(float(result)):
        raise ValueError("non-finite result")
    return result


def _extract_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("self-play proposal must be an object")
    return value


def generate_verified_self_play_rows(
    runtime,
    readiness: dict[str, Any],
    *,
    cycle: int,
    max_tasks: int = 12,
) -> tuple[list[ResearchRow], dict[str, Any]]:
    """Let the model propose tasks, but trust only independent verification.

    The model never labels its own examples. It proposes an arithmetic
    expression plus claimed answer; a deterministic AST verifier recomputes the
    answer. Only verified proposals become tool-use curriculum rows.
    """
    if not bool((readiness or {}).get("enabled")):
        return [], {
            "enabled": False,
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
            "reason": str((readiness or {}).get("reason") or "readiness gate closed"),
        }

    rng = random.Random(int(cycle) * 104729 + 17)
    rows: list[ResearchRow] = []
    rejected = 0
    seen: set[str] = set()
    attempts = max(1, min(32, int(max_tasks)))
    for index in range(attempts):
        nonce = rng.randint(10, 9999)
        prompt = (
            "Create one small arithmetic verification task. "
            "Return JSON only with keys expression and result. "
            "Use integers and only +, -, *, //, %. "
            f"Nonce={nonce}; proposal={index}."
        )
        try:
            raw = runtime.generate(prompt, max_new_tokens=80)
            proposal = _extract_object(raw)
            expression = str(proposal.get("expression", "")).strip()
            claimed = proposal.get("result")
            verified = _safe_eval(expression)
            if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
                raise ValueError("claimed result is not numeric")
            if abs(float(verified) - float(claimed)) > 1e-9:
                raise ValueError("claimed result failed independent verification")
            if expression in seen:
                raise ValueError("duplicate verified proposal")
            seen.add(expression)
            arguments = json.dumps(
                {"expression": expression},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            rows.append(ResearchRow(
                "tools",
                [
                    {
                        "role": "user",
                        "content": (
                            f"Calculate {expression}. Use the calculator tool "
                            "instead of doing the arithmetic in prose."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"calculator","arguments":'
                            + arguments
                            + "}</tool_call>"
                        ),
                    },
                ],
            ))
        except Exception:
            rejected += 1

    return rows, {
        "enabled": True,
        "attempted": attempts,
        "accepted": len(rows),
        "rejected": rejected,
        "verifier": "deterministic_ast_arithmetic_v1",
        "model_labels_trusted": False,
        "training_rows": len(rows),
    }
