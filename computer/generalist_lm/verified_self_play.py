from __future__ import annotations

import ast
import json
import math
import operator
import random
import re
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


def _json_object(raw: str) -> dict[str, Any]:
    value = _extract_object(raw)
    return {str(key): val for key, val in value.items()}


def _verify_tool_solver(raw: str, expression: str) -> str:
    text = str(raw or "").strip()
    match = re.fullmatch(r'<tool_call>(\{.*\})</tool_call>', text, flags=re.DOTALL)
    if not match:
        raise ValueError("solver did not emit exactly one tool_call")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict) or payload.get("name") != "calculator":
        raise ValueError("solver chose the wrong tool")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or str(arguments.get("expression", "")).strip() != expression:
        raise ValueError("solver changed the verified expression")
    _safe_eval(expression)
    return text


def _verify_structured_solver(raw: str, key: str, value: int | str) -> str:
    parsed = json.loads(str(raw or "").strip())
    if parsed != {key: value}:
        raise ValueError("structured solver output failed exact schema verification")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _verify_coding_solver(raw: str, function_name: str, operation: str, constant: int) -> str:
    source = str(raw or "").strip()
    tree = ast.parse(source, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("coding solver must emit exactly one function")
    fn = tree.body[0]
    if fn.name != function_name or len(fn.args.args) != 1 or fn.args.args[0].arg != "x":
        raise ValueError("coding solver function signature mismatch")
    if fn.decorator_list or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        raise ValueError("coding solver body is not the constrained form")
    expr = fn.body[0].value
    if not isinstance(expr, ast.BinOp) or not isinstance(expr.left, ast.Name) or expr.left.id != "x":
        raise ValueError("coding solver return expression mismatch")
    if not isinstance(expr.right, ast.Constant) or isinstance(expr.right.value, bool):
        raise ValueError("coding solver constant mismatch")
    if int(expr.right.value) != int(constant):
        raise ValueError("coding solver used the wrong constant")
    expected_op = ast.Add if operation == "add" else ast.Mult
    if not isinstance(expr.op, expected_op):
        raise ValueError("coding solver used the wrong operation")
    # Hidden deterministic checks without executing arbitrary model code.
    for x in (-7, -1, 0, 2, 11):
        expected = x + constant if operation == "add" else x * constant
        observed = x + int(expr.right.value) if isinstance(expr.op, ast.Add) else x * int(expr.right.value)
        if observed != expected:
            raise ValueError("coding hidden test failed")
    return source


def generate_verified_multiagent_rows(
    proposer,
    solver,
    readiness: dict[str, Any],
    *,
    cycle: int,
    max_tasks: int = 9,
) -> tuple[list[ResearchRow], dict[str, Any]]:
    """Generalized proposer -> solver -> independent verifier self-play.

    The proposer never supplies trusted labels. The solver is queried
    separately, and deterministic code validates the task and solution before
    anything enters the curriculum.
    """
    if not bool((readiness or {}).get("enabled")):
        return [], {
            "enabled": False,
            "mode": "multiagent_verified",
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
            "reason": str((readiness or {}).get("reason") or "readiness gate closed"),
        }

    rng = random.Random(int(cycle) * 99991 + 31)
    attempts = max(1, min(24, int(max_tasks)))
    rows: list[ResearchRow] = []
    accepted_by_domain = {"tools": 0, "structured": 0, "coding": 0}
    rejected = 0

    for index in range(attempts):
        kind = ("tools", "structured", "coding")[index % 3]
        nonce = rng.randint(100, 99999)
        try:
            if kind == "tools":
                proposal_raw = proposer.generate(
                    "Propose one small arithmetic task as JSON only with key expression. "
                    "Use integers and only +, -, *, //, %. "
                    f"nonce={nonce}",
                    max_new_tokens=64,
                )
                proposal = _json_object(proposal_raw)
                expression = str(proposal.get("expression", "")).strip()
                expected = _safe_eval(expression)
                solver_raw = solver.generate(
                    f"Use calculator for {expression}. Return only the tool_call.",
                    max_new_tokens=96,
                )
                target = _verify_tool_solver(solver_raw, expression)
                rows.append(ResearchRow("tools", [
                    {"role": "user", "content": f"Use calculator for {expression}."},
                    {"role": "assistant", "content": target},
                ]))
                accepted_by_domain["tools"] += 1

            elif kind == "structured":
                proposal_raw = proposer.generate(
                    "Propose a tiny exact JSON task. Return JSON only with keys key and value. "
                    "key must be 1-16 lowercase ASCII letters; value must be an integer from -50 to 50. "
                    f"nonce={nonce}",
                    max_new_tokens=64,
                )
                proposal = _json_object(proposal_raw)
                key = str(proposal.get("key", ""))
                value = proposal.get("value")
                if not re.fullmatch(r"[a-z]{1,16}", key):
                    raise ValueError("invalid proposed JSON key")
                if isinstance(value, bool) or not isinstance(value, int) or not (-50 <= value <= 50):
                    raise ValueError("invalid proposed JSON value")
                solver_raw = solver.generate(
                    f'Return JSON only with key "{key}" and integer value {value}.',
                    max_new_tokens=64,
                )
                target = _verify_structured_solver(solver_raw, key, value)
                rows.append(ResearchRow("structured", [
                    {"role": "user", "content": f'Return JSON only with key "{key}" and integer value {value}.'},
                    {"role": "assistant", "content": target},
                ]))
                accepted_by_domain["structured"] += 1

            else:
                proposal_raw = proposer.generate(
                    "Propose one constrained Python task as JSON only with keys function, operation, constant. "
                    "function must match task_[a-z]{1,8}; operation is add or multiply; "
                    "constant is an integer from 1 to 12. "
                    f"nonce={nonce}",
                    max_new_tokens=80,
                )
                proposal = _json_object(proposal_raw)
                function_name = str(proposal.get("function", ""))
                operation = str(proposal.get("operation", ""))
                constant = proposal.get("constant")
                if not re.fullmatch(r"task_[a-z]{1,8}", function_name):
                    raise ValueError("invalid proposed function name")
                if operation not in {"add", "multiply"}:
                    raise ValueError("invalid proposed operation")
                if isinstance(constant, bool) or not isinstance(constant, int) or not (1 <= constant <= 12):
                    raise ValueError("invalid proposed constant")
                verb = "adds" if operation == "add" else "multiplies by"
                solver_raw = solver.generate(
                    f"Return only Python code defining {function_name}(x) that {verb} {constant}.",
                    max_new_tokens=128,
                )
                target = _verify_coding_solver(
                    solver_raw,
                    function_name,
                    operation,
                    constant,
                )
                rows.append(ResearchRow("coding", [
                    {"role": "user", "content": f"Return only Python code defining {function_name}(x) that {verb} {constant}."},
                    {"role": "assistant", "content": target},
                ]))
                accepted_by_domain["coding"] += 1
        except Exception:
            rejected += 1

    return rows, {
        "enabled": True,
        "mode": "multiagent_verified",
        "attempted": attempts,
        "accepted": len(rows),
        "rejected": rejected,
        "accepted_by_domain": accepted_by_domain,
        "proposer_and_solver_distinct": proposer is not solver,
        "model_labels_trusted": False,
        "verifiers": [
            "deterministic_ast_arithmetic_v1",
            "exact_json_schema_v1",
            "constrained_python_ast_hidden_tests_v1",
        ],
    }
