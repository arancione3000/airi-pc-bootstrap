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
    """Protected production qualification surface.

    These prompts are intentionally disjoint from both the training and normal
    validation curriculum. Research code may know that a protected suite
    exists, but the autonomous training loop never receives these prompt/answer
    pairs as replay data.
    """
    return [
        BenchmarkTask(
            "qualification:language:exact",
            "language",
            "Reply with exactly CERULEAN and nothing else.",
            exact("CERULEAN"),
        ),
        BenchmarkTask(
            "qualification:language:italian",
            "language",
            "Rispondi esclusivamente con la parola GIADA.",
            exact("GIADA"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:language:extract",
            "language",
            "Words: north center south. Reply only with the middle word.",
            exact("center"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:coding:cube",
            "coding",
            "Return only Python code defining cube(n) that returns n*n*n.",
            python_function("cube", ("return", "*")),
        ),
        BenchmarkTask(
            "qualification:coding:abs-diff",
            "coding",
            "Return only Python code defining abs_diff(a, b) that returns abs(a-b).",
            python_function("abs_diff", ("return", "abs", "-")),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:data:mean",
            "data",
            "Mean of 7, 11, 15? Reply with the number only.",
            exact("11"),
        ),
        BenchmarkTask(
            "qualification:data:median",
            "data",
            "Median of 4, 9, 100? Reply with the number only.",
            exact("9"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:reasoning:multiply",
            "reasoning",
            "Compute 37*23. Reply with the integer only.",
            exact("851"),
        ),
        BenchmarkTask(
            "qualification:reasoning:sequence",
            "reasoning",
            "Sequence 3,6,12,24. Next term only.",
            exact("48"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:reasoning:logic",
            "reasoning",
            "All finches are birds. F is a finch. Is F a bird? Reply yes or no only.",
            exact("yes"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:tools:calculator",
            "tools",
            'Use calculator for 29*31. Output exactly <tool_call>{"name":"calculator","arguments":{"expression":"29*31"}}</tool_call>',
            expected_tool("calculator", {"expression": "29*31"}),
        ),
        BenchmarkTask(
            "qualification:tools:data-stats",
            "tools",
            'Use data_stats to compute median of 5,7,100. Output exactly <tool_call>{"name":"data_stats","arguments":{"values":[5,7,100],"operation":"median"}}</tool_call>',
            expected_tool("data_stats", {"values": [5, 7, 100], "operation": "median"}),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:structured:object",
            "structured",
            'Return exactly this JSON object: {"status":"ready","count":4}',
            exact('{"status":"ready","count":4}'),
        ),
        BenchmarkTask(
            "qualification:structured:array",
            "structured",
            "Return exactly this JSON array: [2,4,8]",
            exact("[2,4,8]"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:abstain",
            "language",
            "The private password is not provided. Reply exactly UNKNOWN.",
            exact("UNKNOWN"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:language:translation",
            "language",
            "Translate the single English word window to Italian. Reply with one word only.",
            exact("finestra"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:coding:positive",
            "coding",
            "Return only Python code defining is_positive(n) that returns whether n is greater than zero.",
            python_function("is_positive", ("return", ">", "0")),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:data:sum",
            "data",
            "Sum 14, 21, 35. Reply with the number only.",
            exact("70"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:reasoning:division",
            "reasoning",
            "Compute 144/12. Reply with the integer only.",
            exact("12"),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:tools:calculator-2",
            "tools",
            'Use calculator for 81-26. Output exactly <tool_call>{"name":"calculator","arguments":{"expression":"81-26"}}</tool_call>',
            expected_tool("calculator", {"expression": "81-26"}),
            critical=False,
        ),
        BenchmarkTask(
            "qualification:structured:mode",
            "structured",
            'Return exactly this JSON object: {"mode":"safe","enabled":true}',
            exact('{"mode":"safe","enabled":true}'),
            critical=False,
        ),
    ]


def qualification_manifest() -> dict[str, object]:
    tasks = qualification_suite()
    return {
        "tasks": len(tasks),
        "domains": {
            domain: sum(task.domain == domain for task in tasks)
            for domain in sorted({task.domain for task in tasks})
        },
        "critical_domains": sorted({task.domain for task in tasks if task.critical}),
        "prompts": [task.prompt for task in tasks],
    }


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
