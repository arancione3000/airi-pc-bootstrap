from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AIRI_ROOT", "/home/user/airi")).resolve()
MODEL = os.environ.get("AIRI_LOCAL_MODEL", "qwen2.5-coder:7b")
OLLAMA_URL = os.environ.get("AIRI_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MAX_ITERATIONS = max(1, min(int(os.environ.get("AIRI_AUTONOMOUS_ITERATIONS", "5")), 20))
TEST_COMMAND = os.environ.get("AIRI_AUTONOMOUS_TEST", "python3 -m pytest -q")

SYSTEM = """You are the local Airi-PC autonomous software engineer. Work only on the repository supplied in context. Return ONLY a unified diff patch. Never modify credentials, secrets, .git, .ssh, auth files, or files outside the repository. Prefer small reversible changes. The patch must be applicable with git apply --check."""


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _context() -> str:
    status = _run(["git", "status", "--short"])
    diff = _run(["git", "diff", "--", "."]) 
    log = _run(["git", "log", "-5", "--oneline"])
    return "\n".join([
        f"REPOSITORY={ROOT}",
        "\nGIT_STATUS:\n" + status.stdout[:12000],
        "\nGIT_DIFF:\n" + diff.stdout[:30000],
        "\nGIT_LOG:\n" + log.stdout[:4000],
    ])


def ask_local_model(goal: str, feedback: str = "") -> str:
    prompt = (
        f"GOAL:\n{goal}\n\nCONTEXT:\n{_context()}\n\n"
        f"PREVIOUS_FEEDBACK:\n{feedback[:16000]}\n\n"
        "Produce the smallest useful unified diff. Do not include markdown fences."
    )
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Local model unavailable at {OLLAMA_URL}: {exc}") from exc
    text = data.get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("Local model returned an empty patch")
    return text.replace("```diff", "").replace("```", "").strip()


def _safe_patch(patch: str) -> None:
    blocked = (".git/", ".ssh/", "auth/", "id_rsa", "credentials", "secret")
    for line in patch.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].split("\t", 1)[0]
            path = path[2:] if path.startswith(("a/", "b/")) else path
            if any(token in path.lower() for token in blocked):
                raise RuntimeError(f"Unsafe patch path: {path}")
            candidate = (ROOT / path).resolve()
            if ROOT not in candidate.parents and candidate != ROOT:
                raise RuntimeError(f"Patch escapes repository: {path}")


def apply_patch(patch: str) -> None:
    _safe_patch(patch)
    check = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", "-"], cwd=ROOT, text=True, input=patch, capture_output=True, check=False)
    if check.returncode != 0:
        raise RuntimeError("git apply --check failed: " + check.stderr[-8000:])
    applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=ROOT, text=True, input=patch, capture_output=True, check=False)
    if applied.returncode != 0:
        raise RuntimeError("git apply failed: " + applied.stderr[-8000:])


def run_tests() -> dict[str, Any]:
    p = subprocess.run(TEST_COMMAND, cwd=ROOT, shell=True, text=True, capture_output=True, timeout=180, check=False)
    return {"returncode": p.returncode, "stdout": p.stdout[-12000:], "stderr": p.stderr[-12000:]}


def autonomous_cycle(goal: str, iterations: int = MAX_ITERATIONS, commit: bool = True) -> dict[str, Any]:
    feedback = ""
    history: list[dict[str, Any]] = []
    for index in range(1, min(iterations, MAX_ITERATIONS) + 1):
        patch = ask_local_model(goal, feedback)
        apply_patch(patch)
        tests = run_tests()
        history.append({"iteration": index, "test": tests, "patch_bytes": len(patch.encode())})
        if tests["returncode"] == 0:
            if commit:
                message = "airi: autonomous local improvement"
                c = _run(["git", "add", "--", "."])
                if c.returncode != 0: raise RuntimeError(c.stderr)
                c = _run(["git", "commit", "-m", message])
                if c.returncode != 0: raise RuntimeError(c.stderr)
            head = _run(["git", "rev-parse", "HEAD"])
            return {"ok": True, "goal": goal, "iterations": index, "commit": head.stdout.strip(), "history": history}
        feedback = tests["stdout"] + "\n" + tests["stderr"]
    return {"ok": False, "goal": goal, "iterations": len(history), "history": history}


def provider_status() -> dict[str, Any]:
    try:
        body = json.dumps({"model": MODEL, "prompt": "ping", "stream": False}).encode()
        req = urllib.request.Request(OLLAMA_URL.replace("/api/chat", "/api/generate"), data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"available": True, "model": MODEL, "http_status": resp.status}
    except Exception as exc:
        return {"available": False, "model": MODEL, "error": str(exc)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("goal")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(autonomous_cycle(args.goal, args.iterations, not args.no_commit), indent=2))
