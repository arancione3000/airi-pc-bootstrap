from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .github_access import commit as git_commit
from .github_access import push as git_push
from .github_access import status as git_status
from .live_telemetry import live_emit, live_flush, session_id
from .orchestrator import ControlPlane
from .screen_share import start_screen_share

ROOT = Path(os.environ.get("AIRI_ROOT") or os.environ.get("AIRIPC_WORKSPACE_ROOT") or ".").resolve()
ALLOWED_OPERATIONS = {
    "analyze", "read", "search", "write", "patch", "test", "build", "lint",
    "git_status", "git_diff", "verify", "context_pack", "snapshot", "restore",
    "environment_strategy",
}


def _load_spec(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if ROOT not in p.parents and p != ROOT:
        raise PermissionError("task spec escapes AIRI_ROOT")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("task spec must be an object")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("task spec requires non-empty steps")
    ids: set[str] = set()
    for raw in steps:
        if not isinstance(raw, dict):
            raise ValueError("every step must be an object")
        sid = str(raw.get("id") or "").strip()
        if not sid or sid in ids:
            raise ValueError(f"invalid/duplicate step id: {sid!r}")
        ids.add(sid)
        op = str(raw.get("operation") or "").strip()
        if op not in ALLOWED_OPERATIONS:
            raise ValueError(f"operation not allowed by remote runner: {op!r}")
    return data


def run_task(spec: dict[str, Any]) -> dict[str, Any]:
    goal = str(spec.get("goal") or "Airi-PC autonomous task")
    scope = [str(x) for x in (spec.get("scope") or ["."])]
    steps = list(spec["steps"])
    limits = dict(spec.get("limits") or {})
    cp = ControlPlane()
    start_screen_share()

    live_emit(
        "runtime",
        "Airi-PC session online",
        session_id(),
        "running",
        dedupe_key="remote-task:session:online",
    )
    live_emit(
        "runtime",
        "Airi autonomous task",
        f"{len(steps)} planned steps",
        "running",
        dedupe_key="remote-task:runner:start",
    )

    result = cp.autonomous_goal(
        goal,
        steps=steps,
        scope=scope,
        max_time=int(limits.get("max_time", 3300)),
        max_iterations=int(limits.get("max_iterations", max(40, len(steps) * 3))),
        max_retries=int(limits.get("max_retries", 1)),
        max_tool_calls=int(limits.get("max_tool_calls", max(80, len(steps) * 4))),
        max_parallel_tasks=1,
        resume=False,
    )
    ready = (result.get("result") or {}).get("status") == "READY"
    if not ready:
        task_id = (result.get("result") or {}).get("task_id") or result.get("task_id")
        task_snapshot = cp.tasks.read(task_id) if task_id else None
        live_emit(
            "runtime",
            "Airi autonomous task failed",
            str((result.get("result") or {}).get("reason") or (result.get("result") or {}).get("error") or "verification failed"),
            "failed",
            dedupe_key="remote-task:runner:failed",
        )
        live_flush(5.0)
        return {"ok": False, "execution": result, "task": task_snapshot}

    if spec.get("persist", True) is False:
        live_emit(
            "runtime",
            "Airi task completed",
            "Temporary session finished; repository unchanged",
            "completed",
            dedupe_key="remote-task:runner:temporary-done",
        )
        live_flush(5.0)
        return {"ok": True, "execution": result, "persisted": False}

    commit_paths = [str(x) for x in (spec.get("commit_paths") or scope)]
    message = str(spec.get("commit_message") or "feat: persist verified Airi-PC autonomous task")

    live_emit(
        "runtime",
        "Persisting verified result",
        f"{len(commit_paths)} scoped paths",
        "verifying",
        dedupe_key="remote-task:runner:persist",
    )
    before = git_status()
    sha = git_commit(message, commit_paths)
    pushed = git_push(str(spec.get("push_branch") or "main"))
    if not pushed.get("ok"):
        live_emit(
            "runtime",
            "Git push failed",
            pushed.get("stderr") or "push failed",
            "failed",
            dedupe_key="remote-task:runner:push-failed",
        )
        raise RuntimeError(pushed.get("stderr") or "git push failed")
    live_emit(
        "runtime",
        "Airi task persisted",
        sha[:12],
        "completed",
        dedupe_key="remote-task:runner:done",
    )
    live_flush(5.0)
    return {"ok": True, "execution": result, "before": before, "commit": sha, "push": pushed}


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 1:
        print("usage: python -m computer.control_plane.remote_task_runner <task-spec.json>", file=sys.stderr)
        return 64
    try:
        spec = _load_spec(args[0])
        result = run_task(spec)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0 if result.get("ok") else 2
    except Exception as exc:
        live_emit("runtime", "Airi runner crashed", str(exc), "failed", dedupe_key="remote-task:runner:crash")
        live_flush(3.0)
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
