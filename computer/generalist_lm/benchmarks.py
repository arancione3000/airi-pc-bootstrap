from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Protocol

from .tool_protocol import parse_tool_call


class Backend(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 192) -> str: ...


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    domain: str
    prompt: str
    checker: Callable[[str], tuple[bool, str]]
    critical: bool = True


def _normalized(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact(expected: str):
    def check(text: str):
        ok = _normalized(text) == _normalized(expected)
        return ok, f"expected exact response {expected!r}"
    return check


def python_function(name: str, required_tokens: tuple[str, ...]):
    def check(text: str):
        code = str(text).strip()
        if code.startswith("python\n"):
            code = code[7:]
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"invalid Python: {exc.msg}"
        defs = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not any(node.name == name for node in defs):
            return False, f"missing function {name}"
        if not all(token in code for token in required_tokens):
            return False, "required implementation tokens missing"
        return True, "valid Python implementation"
    return check


def tool_checker(text: str):
    try:
        call = parse_tool_call(text, allowed_tools={"calculator"})
    except Exception as exc:
        return False, str(exc)
    return (call.arguments.get("expression") == "17*19", "expected calculator expression 17*19")


def default_suite() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            "language:instruction",
            "language",
            "Reply with exactly the word ORANGE and nothing else.",
            exact("ORANGE"),
        ),
        BenchmarkTask(
            "language:italian",
            "language",
            "Rispondi soltanto con la parola CIAO.",
            exact("CIAO"),
        ),
        BenchmarkTask(
            "code:add",
            "coding",
            "Return only Python code defining add(a, b) that returns a+b.",
            python_function("add", ("return", "+")),
        ),
        BenchmarkTask(
            "data:mean",
            "data",
            "Values are 2, 4, 6. What is their arithmetic mean? Reply with the number only.",
            exact("4"),
        ),
        BenchmarkTask(
            "reasoning:multiplication",
            "reasoning",
            "Compute 17*19. Reply with the integer only.",
            exact("323"),
        ),
        BenchmarkTask(
            "tools:calculator",
            "tools",
            "Use calculator for 17*19. Output exactly <tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"17*19\"}}</tool_call>",
            tool_checker,
        ),
        BenchmarkTask(
            "structured:json",
            "structured",
            "Return exactly this JSON object with no markdown: {\"ok\":true,\"items\":3}",
            exact("{\"ok\":true,\"items\":3}"),
        ),
    ]


def run_benchmark(backend: Backend, tasks: list[BenchmarkTask] | None = None) -> dict:
    tasks = list(tasks or default_suite())
    rows = []
    domains: dict[str, list[bool]] = {}
    for task in tasks:
        try:
            output = backend.generate(task.prompt, max_new_tokens=192)
            ok, detail = task.checker(output)
        except Exception as exc:
            output = ""
            ok, detail = False, f"backend error: {type(exc).__name__}: {exc}"
        row = {
            "name": task.name,
            "domain": task.domain,
            "critical": task.critical,
            "ok": bool(ok),
            "detail": detail,
            "output": str(output)[:2000],
        }
        rows.append(row)
        domains.setdefault(task.domain, []).append(bool(ok))

    domain_scores = {
        domain: sum(values) / len(values)
        for domain, values in domains.items()
    }
    critical_failures = [row["name"] for row in rows if row["critical"] and not row["ok"]]
    overall = sum(row["ok"] for row in rows) / max(1, len(rows))
    return {
        "ok": not critical_failures,
        "score": round(100.0 * overall, 6),
        "domain_scores": {k: round(v, 6) for k, v in domain_scores.items()},
        "critical_failures": critical_failures,
        "tasks": rows,
    }
