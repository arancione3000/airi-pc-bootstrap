from __future__ import annotations

"""Qualified local-reasoning compatibility surface.

ChatGPT remains the default reasoning authority. A local AIRI Generalist LM may
be used only when an explicitly enabled checkpoint has passed its qualification
benchmark. Model output never bypasses the existing patch/tool safety layers.
"""

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from . import generalist_provider

ROOT = Path(os.environ.get("AIRI_ROOT", "/home/user/airi")).resolve()
MAX_ITERATIONS = max(1, min(int(os.environ.get("AIRI_AUTONOMOUS_ITERATIONS", "5")), 20))
TEST_COMMAND = os.environ.get("AIRI_AUTONOMOUS_TEST", "python3 -m pytest -q")

_PROVIDER_AT_IMPORT = generalist_provider.status()
CHATGPT_ONLY = not bool(_PROVIDER_AT_IMPORT.get("available"))
REASONING_AUTHORITY = "chatgpt" if CHATGPT_ONLY else "airi-generalist"
PROVIDER = "disabled" if CHATGPT_ONLY else "airi-generalist"
MODEL = None if CHATGPT_ONLY else _PROVIDER_AT_IMPORT.get("model")


def _disabled(operation: str) -> RuntimeError:
    return RuntimeError(
        f"{operation} is disabled: ChatGPT is the sole reasoning authority "
        "unless a benchmark-qualified AIRI Generalist checkpoint is explicitly enabled."
    )


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _context_at(root: Path) -> str:
    def run_local(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30, check=False)
    status = run_local(["git", "status", "--short"])
    diff = run_local(["git", "diff", "--", "."])
    log = run_local(["git", "log", "-5", "--oneline"])
    return "\n".join([
        f"REPOSITORY={root}",
        "\nGIT_STATUS:\n" + status.stdout,
        "\nGIT_DIFF:\n" + diff.stdout[:30000],
        "\nGIT_LOG:\n" + log.stdout[:4000],
    ])


def _context() -> str:
    return _context_at(ROOT)


def _require_generalist() -> dict[str, Any]:
    row = generalist_provider.status()
    if not row.get("available"):
        raise _disabled("local model invocation")
    return row


def ask_local_model(goal: str, feedback: str = "") -> str:
    _require_generalist()
    messages = [
        {
            "role": "system",
            "content": (
                "You are AIRI Generalist, a local reasoning model. "
                "Be precise, state uncertainty, and never claim a tool action happened unless a tool result is provided."
            ),
        },
        {"role": "user", "content": str(goal)},
    ]
    if feedback:
        messages.append({"role": "user", "content": f"Feedback from previous attempt:\n{feedback}"})
    return generalist_provider.chat(messages, max_new_tokens=512)


GENERALIST_PROTECTED_PATHS = {
    "computer/generalist_lm/qualification.py",
    "computer/generalist_lm/production_promotion.py",
    "computer/generalist_lm/research_health.py",
    "computer/generalist_lm/tool_protocol.py",
    "computer/control_plane/generalist_provider.py",
    "computer/control_plane/generalist_agent_bridge.py",
    "computer/control_plane/local_agent.py",
    "computer/control_plane/model_router.py",
    "computer/control_plane/model_gateway.py",
    "computer/code_agent.py",
    "computer/coding.py",
    "config/AIRI_REASONING_POLICY.json",
    ".github/workflows/generalist-lm.yml",
    ".github/workflows/generalist-continuum.yml",
    ".github/workflows/generalist-watchdog.yml",
}
GENERALIST_PROTECTED_PREFIXES = (
    ".ai/generalist-lm/",
    ".ai/generalist-research/",
    "generalist-state/",
)


def _assert_generalist_editable_path(relative_path: str) -> None:
    rel = str(relative_path).replace("\\", "/").lstrip("./")
    if rel in GENERALIST_PROTECTED_PATHS:
        raise RuntimeError(f"Generalist model cannot modify protected self-governance path: {rel}")
    if any(rel.startswith(prefix) for prefix in GENERALIST_PROTECTED_PREFIXES):
        raise RuntimeError(f"Generalist model cannot modify protected state path: {rel}")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    raw = str(text).strip()
    if raw.startswith("~~~"):
        raw = raw.strip("~").strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("generalist model did not return the required JSON change array") from exc
    if not isinstance(value, list):
        raise RuntimeError("generalist model change response must be a JSON array")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("each proposed change must be an object")
        path = str(item.get("path", "")).strip()
        if not path:
            raise RuntimeError("each proposed change requires path")
        content = item.get("content")
        old = item.get("old")
        new = item.get("new")
        if isinstance(content, str) and old is None and new is None:
            rows.append({"path": path, "content": content})
            continue
        if isinstance(old, str) and isinstance(new, str) and content is None:
            if not old:
                raise RuntimeError("patch old text must not be empty")
            rows.append({"path": path, "old": old, "new": new})
            continue
        raise RuntimeError("each proposed change requires either string content or string old+new")
    return rows


def ask_local_model_changes(
    goal: str,
    context: str,
    feedback: str = "",
    root: str | Path | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    del model
    _require_generalist()
    base = Path(root or ROOT).resolve()
    prompt = (
        "Return ONLY a JSON array of proposed repository edits. "
        "Each item must be either {\"path\":\"relative/path\",\"old\":\"exact old text\",\"new\":\"replacement\"} "
        "for a focused patch, or {\"path\":\"relative/path\",\"content\":\"full new content\"} for a full replacement. "
        "Prefer focused old/new patches for existing large files. Do not use markdown. "
        "Do not propose .git, .ssh, credentials, secrets, or paths outside the repository.\n\n"
        f"GOAL:\n{goal}\n\nCONTEXT:\n{context[:60000]}"
    )
    if feedback:
        prompt += f"\n\nFEEDBACK:\n{feedback[:10000]}"
    text = generalist_provider.chat(
        [
            {"role": "system", "content": "You are a careful coding model proposing reviewable repository edits."},
            {"role": "user", "content": prompt},
        ],
        max_new_tokens=1600,
    )
    changes = _extract_json_array(text)
    blocked = (".git/", ".ssh/", "auth/", "id_rsa", "credentials", "secret")
    for row in changes:
        rel = row["path"].replace("\\", "/")
        if rel.startswith("/") or any(part == ".." for part in Path(rel).parts):
            raise RuntimeError(f"Unsafe model change path: {rel}")
        if any(token in rel.lower() for token in blocked):
            raise RuntimeError(f"Unsafe model change path: {rel}")
        _assert_generalist_editable_path(rel)
        candidate = (base / rel).resolve()
        if base not in candidate.parents and candidate != base:
            raise RuntimeError(f"Model change escapes repository: {rel}")
    return changes


def _provider_request(messages: list[dict[str, str]], model: str) -> dict[str, Any]:
    _require_generalist()
    if model not in {"airi-generalist", "local-causal-lm"}:
        raise RuntimeError("only the qualified AIRI Generalist provider is allowed locally")
    text = generalist_provider.chat(messages, max_new_tokens=512)
    return {"provider": "airi-generalist", "model": "local-causal-lm", "content": text}


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
    check = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=ROOT, text=True, input=patch, capture_output=True, check=False,
    )
    if check.returncode:
        raise RuntimeError("git apply --check failed: " + check.stderr[-8000:])
    applied = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=ROOT, text=True, input=patch, capture_output=True, check=False,
    )
    if applied.returncode:
        raise RuntimeError("git apply failed: " + applied.stderr[-8000:])


def run_tests() -> dict[str, Any]:
    command = shlex.split(TEST_COMMAND)
    if not command:
        raise RuntimeError("AIRI_AUTONOMOUS_TEST must contain a command")
    p = subprocess.run(
        command, cwd=ROOT, text=True,
        capture_output=True, timeout=180, check=False,
    )
    return {"returncode": p.returncode, "stdout": p.stdout[-12000:], "stderr": p.stderr[-12000:]}


def autonomous_cycle(goal: str, iterations: int = MAX_ITERATIONS, commit: bool = True) -> dict[str, Any]:
    del goal, iterations, commit
    raise RuntimeError(
        "autonomous local-model repository editing is not enabled by this provider bridge; "
        "reasoning output must still pass the Control Plane transaction and verification workflow"
    )


def provider_status() -> dict[str, Any]:
    row = generalist_provider.status()
    return {
        **row,
        "chatgpt_only": not bool(row.get("available")),
        "reasoning_authority": "airi-generalist" if row.get("available") else "chatgpt",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qualified AIRI Generalist local-model compatibility surface")
    parser.add_argument("goal")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args()
    if not provider_status().get("available"):
        raise SystemExit(_disabled("local-agent CLI"))
    print(ask_local_model(args.goal))
