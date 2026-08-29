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
PROVIDER = os.environ.get("AIRI_MODEL_PROVIDER", "ollama")
MODEL = os.environ.get("OPENROUTER_MODEL", MODEL) if PROVIDER in {"openrouter", "openrouter_bridge"} and os.environ.get("OPENROUTER_MODEL") else MODEL
OLLAMA_URL = os.environ.get("AIRI_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OPENROUTER_BRIDGE_URL = os.environ.get("AIRI_OPENROUTER_BRIDGE_URL", "http://127.0.0.1:17893/v1/chat/completions")
OPENROUTER_BRIDGE_AUTH_FILE = Path(os.environ.get("AIRI_MODEL_GATEWAY_AUTH_FILE", "/tmp/airi-model-gateway.token"))
MAX_ITERATIONS = max(1, min(int(os.environ.get("AIRI_AUTONOMOUS_ITERATIONS", "5")), 20))
TEST_COMMAND = os.environ.get("AIRI_AUTONOMOUS_TEST", "python3 -m pytest -q")

SYSTEM = """You are the local Airi-PC autonomous software engineer. Work only on the repository supplied in context. Return ONLY a unified diff patch. Never modify credentials, secrets, .git, .ssh, auth files, or files outside the repository. Prefer small reversible changes. The patch must be applicable with git apply --check."""


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _context_at(root: Path) -> str:
    def run_local(command: list[str]):
        return subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30, check=False)
    status = run_local(["git", "status", "--short"])
    diff = run_local(["git", "diff", "--", "."])
    log = run_local(["git", "log", "-5", "--oneline"])
    return "\n".join([
        f"REPOSITORY={root}",
        "\nGIT_STATUS:\n" + status.stdout[:12000],
        "\nGIT_DIFF:\n" + diff.stdout[:30000],
        "\nGIT_LOG:\n" + log.stdout[:4000],
    ])


def _context() -> str:
    return _context_at(ROOT)


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


CHANGE_SYSTEM = """You are the Airi-PC autonomous coding engineer. Work only on the repository supplied in context. Return ONLY valid JSON in this exact shape: {\"changes\":[{\"path\":\"...\",\"operation\":\"patch|write\",\"old\":\"...\",\"new\":\"...\",\"content\":\"...\",\"test_command\":\"...\",\"scope\":[\"...\"]}]}. For patch operations provide exact old/new text from the inspected repository. For write operations provide complete file content. Never target credentials, secrets, .git, .ssh, auth files, or paths outside the repository. Return an empty changes array only when no source change is actually required."""


def _provider_request(messages: list[dict[str, str]], model: str) -> dict[str, Any]:
    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OpenRouter provider configured but OPENROUTER_API_KEY is missing")
        payload = {
            "model": os.environ.get("OPENROUTER_MODEL", model),
            "stream": False,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
            "include_reasoning": False,
        }
        req = urllib.request.Request(
            os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"),
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://127.0.0.1"),
                "X-Title": os.environ.get("OPENROUTER_X_TITLE", "Airi-PC"),
            },
        )
    else:
        payload = {"model": model, "stream": False, "messages": messages, "max_tokens": 4096, "temperature": 0.1, "include_reasoning": False}
        if PROVIDER == "openrouter_bridge":
            try:
                bridge_token = OPENROUTER_BRIDGE_AUTH_FILE.read_text().strip()
            except OSError as exc:
                raise RuntimeError("Model gateway unavailable: auth token missing") from exc
            req = urllib.request.Request(OPENROUTER_BRIDGE_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {bridge_token}"})
        else:
            payload = {"model": model, "stream": False, "messages": messages, "options": {"temperature": 0.1}}
            req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read(4000).decode(errors="replace")
        raise RuntimeError(f"Model provider HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Model provider unavailable: {exc}") from exc


def ask_local_model_changes(goal: str, context: str, feedback: str = "", root: str | Path | None = None, model: str | None = None) -> list[dict[str, Any]]:
    repo = Path(root).resolve() if root else ROOT
    prompt = (
        f"GOAL:\n{goal}\n\nREPOSITORY_CONTEXT:\n{context[:120000]}\n\n"
        f"PREVIOUS_FEEDBACK:\n{feedback[:16000]}\n\n"
        "Identify the smallest concrete source changes needed to satisfy the goal. "
        "Each change MUST include path, operation, test_command and scope. "
        "Use exact old text when operation=patch. Do not merely describe a plan."
    )
    data = _provider_request([{"role":"system","content":CHANGE_SYSTEM},{"role":"user","content":prompt}], model or MODEL)
    text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") or data.get("message", {}).get("content", "")).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Local model returned invalid changes JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("changes"), list):
        raise RuntimeError("Local model response missing changes array")
    return payload["changes"]


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
        payload = {"model": MODEL, "stream": False, "messages": [{"role": "user", "content": "Reply exactly AIRI_PROVIDER_OK"}], "max_tokens": 8, "temperature": 0}
        if PROVIDER == "openrouter_bridge":
            token = OPENROUTER_BRIDGE_AUTH_FILE.read_text().strip()
            if not token:
                raise RuntimeError("Model gateway unavailable: auth token missing")
            req = urllib.request.Request(OPENROUTER_BRIDGE_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        elif PROVIDER == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OpenRouter provider configured but OPENROUTER_API_KEY is missing")
            req = urllib.request.Request(os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"), data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://127.0.0.1"), "X-Title": os.environ.get("OPENROUTER_X_TITLE", "Airi-PC")})
        else:
            req = urllib.request.Request(OLLAMA_URL.replace("/api/chat", "/api/generate"), data=json.dumps({"model": MODEL, "prompt": "ping", "stream": False}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"available": True, "provider": PROVIDER, "model": MODEL, "http_status": resp.status}
    except Exception as exc:
        return {"available": False, "provider": PROVIDER, "model": MODEL, "error": str(exc)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("goal")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(autonomous_cycle(args.goal, args.iterations, not args.no_commit), indent=2))
