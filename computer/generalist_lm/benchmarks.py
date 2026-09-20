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


def expected_tool(name: str, expected_arguments: dict):
    def check(text: str):
        try:
            call = parse_tool_call(text, allowed_tools={name})
        except Exception as exc:
            return False, str(exc)
        ok = call.name == name and call.arguments == expected_arguments
        return ok, f"expected {name} with arguments {expected_arguments!r}"
    return check


def tool_checker(text: str):
    return expected_tool("calculator", {"expression": "17*19"})(text)


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


def qualification_suite() -> list[BenchmarkTask]:
    """Broader protected qualification surface, distinct from research curriculum."""
    core = default_suite()
    extras = [
        BenchmarkTask("language:italian-2", "language", "Rispondi esclusivamente con VERDE.", exact("VERDE"), critical=False),
        BenchmarkTask("language:extract", "language", "Text: alpha beta gamma. Reply only with the middle word.", exact("beta"), critical=False),
        BenchmarkTask("language:unknown", "language", "The secret code is not provided. Reply exactly UNKNOWN.", exact("UNKNOWN"), critical=False),
        BenchmarkTask("code:is-even", "coding", "Return only Python code defining is_even(n) using modulo 2.", python_function("is_even", ("%", "2")), critical=False),
        BenchmarkTask("code:negative", "coding", "Return only Python code defining negate(x) that returns -x.", python_function("negate", ("return", "-")), critical=False),
        BenchmarkTask("data:median", "data", "Median of 1, 2, 100? Reply with the number only.", exact("2"), critical=False),
        BenchmarkTask("data:sum", "data", "Sum 11, 13, 17. Reply with the number only.", exact("41"), critical=False),
        BenchmarkTask("reasoning:sequence", "reasoning", "Sequence 2,4,8,16. Next term only.", exact("32"), critical=False),
        BenchmarkTask("reasoning:logic", "reasoning", "All robins are birds. R is a robin. Is R a bird? Reply yes or no only.", exact("yes"), critical=False),
        BenchmarkTask("reasoning:subtract", "reasoning", "Compute 1000-375. Reply with the integer only.", exact("625"), critical=False),
        BenchmarkTask(
            "tools:data-stats",
            "tools",
            'Use data_stats to compute mean of 3,6,9. Output exactly <tool_call>{"name":"data_stats","arguments":{"values":[3,6,9],"operation":"mean"}}</tool_call>',
            expected_tool("data_stats", {"values": [3, 6, 9], "operation": "mean"}),
            critical=False,
        ),
        BenchmarkTask(
            "tools:calculator-2",
            "tools",
            'Use calculator for 23+19. Output exactly <tool_call>{"name":"calculator","arguments":{"expression":"23+19"}}</tool_call>',
            expected_tool("calculator", {"expression": "23+19"}),
            critical=False,
        ),
        BenchmarkTask("structured:array", "structured", 'Return exactly this JSON array: [1,2,3]', exact("[1,2,3]"), critical=False),
        BenchmarkTask("structured:boolean", "structured", 'Return exactly {"ready":false}', exact('{"ready":false}'), critical=False),
    ]
    return core + extras


def _run_backend(backend: Backend, prompt: str, *, max_new_tokens: int) -> str:
    chat = getattr(backend, "chat", None)
    if callable(chat):
        return chat([{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens)
    return backend.generate(prompt, max_new_tokens=max_new_tokens)


def run_benchmark(backend: Backend, tasks: list[BenchmarkTask] | None = None) -> dict:
    tasks = list(tasks or default_suite())
    rows = []
    domains: dict[str, list[bool]] = {}
    for task in tasks:
        try:
            output = _run_backend(backend, task.prompt, max_new_tokens=192)
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
