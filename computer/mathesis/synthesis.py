from __future__ import annotations

import ast
import math
import time
from typing import Any, Callable

from .types import ProgramCandidate


_ALLOWED_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
}


_TEMPLATES: dict[str, list[tuple[str, str, float]]] = {
    "gcd": [
        (
            "euclid_mod",
            "def solve(a, b):\n"
            "    a, b = abs(int(a)), abs(int(b))\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n",
            1.0,
        ),
        (
            "binaryish_subtractive",
            "def solve(a, b):\n"
            "    a, b = abs(int(a)), abs(int(b))\n"
            "    if a == 0: return b\n"
            "    if b == 0: return a\n"
            "    while a != b:\n"
            "        if a > b: a -= b\n"
            "        else: b -= a\n"
            "    return a\n",
            3.0,
        ),
    ],
    "fibonacci": [
        (
            "iterative",
            "def solve(n):\n"
            "    n = int(n)\n"
            "    if n < 0: raise ValueError('n must be nonnegative')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n",
            1.0,
        ),
        (
            "table",
            "def solve(n):\n"
            "    n = int(n)\n"
            "    if n < 0: raise ValueError('n must be nonnegative')\n"
            "    if n < 2: return n\n"
            "    values = [0, 1]\n"
            "    for i in range(2, n + 1): values.append(values[-1] + values[-2])\n"
            "    return values[n]\n",
            2.0,
        ),
    ],
    "factorial": [
        (
            "iterative",
            "def solve(n):\n"
            "    n = int(n)\n"
            "    if n < 0: raise ValueError('n must be nonnegative')\n"
            "    out = 1\n"
            "    for i in range(2, n + 1): out *= i\n"
            "    return out\n",
            1.0,
        ),
        (
            "descending",
            "def solve(n):\n"
            "    n = int(n)\n"
            "    if n < 0: raise ValueError('n must be nonnegative')\n"
            "    out = 1\n"
            "    while n > 1:\n"
            "        out *= n\n"
            "        n -= 1\n"
            "    return out\n",
            1.2,
        ),
    ],
    "is_prime": [
        (
            "sqrt_trial",
            "def solve(n):\n"
            "    n = int(n)\n"
            "    if n < 2: return False\n"
            "    if n % 2 == 0: return n == 2\n"
            "    d = 3\n"
            "    while d * d <= n:\n"
            "        if n % d == 0: return False\n"
            "        d += 2\n"
            "    return True\n",
            1.0,
        ),
        (
            "linear_trial",
            "def solve(n):\n"
            "    n = int(n)\n"
            "    if n < 2: return False\n"
            "    for d in range(2, n):\n"
            "        if n % d == 0: return False\n"
            "    return True\n",
            4.0,
        ),
    ],
    "sort": [
        (
            "insertion",
            "def solve(values):\n"
            "    out = list(values)\n"
            "    for i in range(1, len(out)):\n"
            "        key = out[i]\n"
            "        j = i - 1\n"
            "        while j >= 0 and out[j] > key:\n"
            "            out[j + 1] = out[j]\n"
            "            j -= 1\n"
            "        out[j + 1] = key\n"
            "    return out\n",
            2.0,
        ),
        (
            "selection",
            "def solve(values):\n"
            "    out = list(values)\n"
            "    for i in range(len(out)):\n"
            "        best = i\n"
            "        for j in range(i + 1, len(out)):\n"
            "            if out[j] < out[best]: best = j\n"
            "        out[i], out[best] = out[best], out[i]\n"
            "    return out\n",
            3.0,
        ),
    ],
}


def _guard_code(code: str) -> None:
    tree = ast.parse(code)
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.Delete,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"forbidden generated syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access is forbidden in generated programs")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            allowed = set(_ALLOWED_BUILTINS) | {"solve", "ValueError"}
            if node.func.id not in allowed:
                raise ValueError(f"call to {node.func.id!r} is not allowed")


def _load(code: str) -> Callable[..., Any]:
    _guard_code(code)
    namespace: dict[str, Any] = {}
    safe_builtins = dict(_ALLOWED_BUILTINS)
    safe_builtins["ValueError"] = ValueError
    exec(compile(code, "<mathesis-candidate>", "exec"), {"__builtins__": safe_builtins}, namespace)
    fn = namespace.get("solve")
    if not callable(fn):
        raise ValueError("candidate must define solve")
    return fn


def _is_prime_reference(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def _fib_reference(n: int) -> int:
    a, b = 0, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return a


def _cases(task: str):
    if task == "gcd":
        return [((a, b), math.gcd(a, b)) for a in range(-24, 25, 3) for b in range(-20, 21, 4)]
    if task == "fibonacci":
        return [((n,), _fib_reference(n)) for n in range(0, 31)]
    if task == "factorial":
        return [((n,), math.factorial(n)) for n in range(0, 14)]
    if task == "is_prime":
        return [((n,), _is_prime_reference(n)) for n in range(-5, 250)]
    if task == "sort":
        values = [
            [],
            [1],
            [2, 1],
            [3, -1, 3, 0],
            list(range(20, -1, -1)),
            [5, 5, 5, 1, 2, 1],
        ]
        return [((row,), sorted(row)) for row in values]
    raise ValueError(f"unsupported synthesis task: {task}")


class ProgramSynthesizer:
    def supported_tasks(self) -> tuple[str, ...]:
        return tuple(sorted(_TEMPLATES))

    def synthesize(self, task: str) -> ProgramCandidate:
        task = str(task or "").strip()
        if task not in _TEMPLATES:
            raise ValueError(f"unsupported synthesis task: {task}")

        cases = _cases(task)
        candidates: list[ProgramCandidate] = []
        for name, code, complexity_penalty in _TEMPLATES[task]:
            start = time.perf_counter()
            passed = 0
            errors: list[str] = []
            try:
                fn = _load(code)
                for args, expected in cases:
                    try:
                        actual = fn(*args)
                    except Exception as exc:
                        errors.append(repr(exc))
                        continue
                    if actual == expected:
                        passed += 1
                    else:
                        errors.append(f"{args!r}: {actual!r} != {expected!r}")
            except Exception as exc:
                errors.append(repr(exc))
            elapsed = max(0.0, time.perf_counter() - start)
            total = len(cases)
            verified = passed == total
            score = (passed / max(1, total)) * 100.0 - complexity_penalty - min(2.0, elapsed)
            candidates.append(
                ProgramCandidate(
                    name=name,
                    task=task,
                    code=code,
                    tests_passed=passed,
                    tests_total=total,
                    score=round(score, 6),
                    verified=verified,
                    details={
                        "complexity_penalty": complexity_penalty,
                        "elapsed_seconds": elapsed,
                        "errors": errors[:5],
                        "guard": "restricted_ast_and_builtins",
                    },
                )
            )

        verified = [candidate for candidate in candidates if candidate.verified]
        if not verified:
            raise RuntimeError(f"no verified program candidate for {task}")
        return max(verified, key=lambda candidate: candidate.score)
