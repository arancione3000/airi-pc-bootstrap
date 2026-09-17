from __future__ import annotations

import datetime as dt
import threading
import uuid
from typing import Any, Callable

from .store import load_json, now, redact, save_json

FILE = "mcp-tasks.json"
TERMINAL = {"completed", "failed", "cancelled"}


class MCPTaskStore:
    """Durable implementation of the MCP Tasks extension.

    Task records survive process restarts. A task still marked ``working`` can be
    resumed lazily on the next tasks/get call. Listeners receive complete public
    task snapshots so subscriptions/listen can emit notifications/tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.state = load_json(FILE, {"version": 1, "tasks": {}})
        self.state.setdefault("version", 1)
        self.state.setdefault("tasks", {})

    def _save(self) -> None:
        save_json(FILE, self.state)

    def add_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def _notify(self, task_id: str) -> None:
        try:
            snapshot = self.public(task_id)
        except KeyError:
            return
        listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                pass

    def create(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        stamp = now()
        row = {
            "taskId": task_id,
            "tool": tool,
            # Long-running autonomous goals should not carry credentials. We
            # persist a redacted copy so durable state never becomes a secret store.
            "arguments": redact(arguments),
            "status": "working",
            "statusMessage": "Airi-PC is processing this request.",
            "createdAt": stamp,
            "lastUpdatedAt": stamp,
            "ttlMs": 3_600_000,
            "pollIntervalMs": 1500,
            "result": None,
            "error": None,
            "cancelRequested": False,
        }
        with self._lock:
            self.state["tasks"][task_id] = row
            self._save()
        self._notify(task_id)
        return self.public(task_id, result_type="task")

    def _execute(self, task_id: str, dispatcher: Callable[[str, dict[str, Any]], Any]) -> None:
        with self._lock:
            row = self.state["tasks"].get(task_id)
            if not row or row.get("status") in TERMINAL or row.get("cancelRequested"):
                return
            tool, args = row["tool"], dict(row.get("arguments") or {})
        try:
            result = dispatcher(tool, args)
            with self._lock:
                row = self.state["tasks"].get(task_id)
                if not row:
                    return
                if row.get("cancelRequested"):
                    row["status"] = "cancelled"
                    row["statusMessage"] = "Cancelled."
                else:
                    row["status"] = "completed"
                    row["result"] = redact(result)
                    row["statusMessage"] = "Completed."
                row["lastUpdatedAt"] = now()
                self._save()
            self._notify(task_id)
        except Exception as exc:
            with self._lock:
                row = self.state["tasks"].get(task_id)
                if row:
                    row["status"] = "cancelled" if row.get("cancelRequested") else "failed"
                    row["error"] = {"code": -32603, "message": str(exc)}
                    row["statusMessage"] = "Cancelled." if row.get("cancelRequested") else str(exc)
                    row["lastUpdatedAt"] = now()
                    self._save()
            self._notify(task_id)
        finally:
            with self._lock:
                self._threads.pop(task_id, None)

    def ensure_running(self, task_id: str, dispatcher: Callable[[str, dict[str, Any]], Any]) -> None:
        with self._lock:
            row = self.state["tasks"].get(task_id)
            if not row:
                raise KeyError(task_id)
            if row.get("status") != "working" or row.get("cancelRequested"):
                return
            thread = self._threads.get(task_id)
            if thread and thread.is_alive():
                return
            thread = threading.Thread(
                target=self._execute,
                args=(task_id, dispatcher),
                daemon=True,
                name=f"airi-mcp-task-{task_id[:8]}",
            )
            self._threads[task_id] = thread
            thread.start()

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.state["tasks"].get(task_id)
            if not row:
                raise KeyError(task_id)
            if row.get("status") not in TERMINAL:
                row["cancelRequested"] = True
                # Cancellation is cooperative. Keep the current observable state
                # until the worker acknowledges it; a not-running persisted task
                # can be marked cancelled immediately.
                thread = self._threads.get(task_id)
                if not thread or not thread.is_alive():
                    row["status"] = "cancelled"
                    row["statusMessage"] = "Cancelled."
                else:
                    row["statusMessage"] = "Cancellation requested."
                row["lastUpdatedAt"] = now()
                self._save()
        self._notify(task_id)
        return {"resultType": "complete"}

    def update(self, task_id: str, input_responses: Any = None) -> dict[str, Any]:
        with self._lock:
            row = self.state["tasks"].get(task_id)
            if not row:
                raise KeyError(task_id)
            row["lastInputResponses"] = redact(input_responses or {})
            row["lastUpdatedAt"] = now()
            self._save()
        self._notify(task_id)
        return {"resultType": "complete"}

    def public(self, task_id: str, result_type: str = "complete") -> dict[str, Any]:
        with self._lock:
            row = self.state["tasks"].get(task_id)
            if not row:
                raise KeyError(task_id)
            out = {
                "resultType": result_type,
                "taskId": task_id,
                "status": row["status"],
                "statusMessage": row.get("statusMessage"),
                "createdAt": _iso(row.get("createdAt")),
                "lastUpdatedAt": _iso(row.get("lastUpdatedAt")),
                "ttlMs": row.get("ttlMs"),
                "pollIntervalMs": row.get("pollIntervalMs"),
            }
            if row.get("status") == "completed":
                result = row.get("result")
                if isinstance(result, dict) and ("content" in result or "structuredContent" in result):
                    out["result"] = result
                else:
                    out["result"] = {"content": [{"type": "text", "text": str(result)}], "structuredContent": result}
            elif row.get("status") == "failed":
                out["error"] = row.get("error") or {"code": -32603, "message": "task failed"}
            return redact(out)


def _iso(value: Any) -> str:
    try:
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
