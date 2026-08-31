from __future__ import annotations

import re
import threading
import uuid
from typing import Any, Callable, Iterable

from .store import load_json, now, redact, save_json

FILE = "reasoning.json"
PHASES = {
    "UNDERSTAND", "PLAN", "EXECUTE", "OBSERVE", "TEST",
    "DIAGNOSE", "REPLAN", "VERIFY", "DONE", "FAILED",
}
STEP_STATES = {
    "PENDING", "READY", "RUNNING", "BLOCKED", "FAILED",
    "COMPLETED", "SKIPPED",
}
DEFAULT_MAX_ATTEMPTS = 3


class ReasoningEngine:
    """Persistent state machine above the Control Plane.

    The engine owns reasoning-run state, planning metadata, observations,
    feedback, retry/replan decisions, evidence references, and completion.
    It never executes tools itself; execution is delegated to Control Plane
    callbacks supplied by the orchestrator.
    """

    def __init__(self, *, executor: Callable[..., dict[str, Any]] | None = None):
        self.executor = executor
        self._lock = threading.RLock()
        self.state = load_json(FILE, {"version": 1, "runs": {}, "active_run_id": None})
        if not isinstance(self.state, dict):
            self.state = {"version": 1, "runs": {}, "active_run_id": None}
        self.state.setdefault("version", 1)
        self.state.setdefault("runs", {})
        self.state.setdefault("active_run_id", None)

    def _persist(self) -> None:
        save_json(FILE, redact(self.state))

    @staticmethod
    def _normalize_step(step: Any, index: int) -> dict[str, Any]:
        if isinstance(step, str):
            step = {"title": step}
        if not isinstance(step, dict):
            raise ValueError("each reasoning step must be a string or object")
        row = dict(step)
        row.setdefault("id", f"step-{index + 1}")
        row.setdefault("title", row["id"])
        row.setdefault("description", row.get("title", row["id"]))
        row.setdefault("kind", row.get("operation", "analysis"))
        row.setdefault("status", "PENDING")
        row.setdefault("dependencies", list(row.get("depends_on", [])))
        row.setdefault("attempts", 0)
        row.setdefault("max_attempts", DEFAULT_MAX_ATTEMPTS)
        row.setdefault("result", None)
        row.setdefault("error", None)
        row.setdefault("evidence", [])
        row.setdefault("metadata", {})
        row.setdefault("operation", row.get("kind", "analyze"))
        row.setdefault("args", {})
        return row

    @staticmethod
    def _default_plan(goal: str, scope: Iterable[str] | None = None) -> list[dict[str, Any]]:
        project = list(scope or ["."])[0]
        return [
            {"id": "understand", "title": "understand goal", "kind": "analysis", "operation": "analyze", "args": {"path": project}},
            {"id": "plan", "title": "create execution plan", "kind": "planning", "operation": "context_pack", "args": {"query": goal}, "dependencies": ["understand"]},
            {"id": "execute", "title": "execute via Control Plane", "kind": "execution", "operation": "code_agent", "args": {"goal": goal, "project_path": project, "scope": list(scope or [project])}, "dependencies": ["plan"]},
            {"id": "test", "title": "test and verify", "kind": "verification", "operation": "verify", "args": {"requirements": [goal], "project_path": project}, "dependencies": ["execute"]},
        ]

    def start(self, goal: str, plan: Iterable[Any] | None = None, *, scope=None, metadata=None) -> dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        steps_raw = list(plan) if plan is not None else self._default_plan(goal, scope)
        steps = [self._normalize_step(s, i) for i, s in enumerate(steps_raw)]
        ids = [s["id"] for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate reasoning step id")
        known = set(ids)
        for step in steps:
            missing = [d for d in step["dependencies"] if d not in known]
            if missing:
                raise ValueError(f"unknown reasoning dependency: {missing}")
        run_id = uuid.uuid4().hex[:16]
        stamp = now()
        run = {
            "run_id": run_id,
            "goal": goal,
            "phase": "UNDERSTAND",
            "status": "RUNNING",
            "plan": steps,
            "current_step": None,
            "observations": [],
            "feedback": [],
            "errors": [],
            "evidence": [],
            "created_at": stamp,
            "updated_at": stamp,
            "metadata": dict(metadata or {}),
            "scope": list(scope or []),
            "history": [],
            "definition_of_done": self._done_checklist(),
        }
        with self._lock:
            self.state["runs"][run_id] = run
            self.state["active_run_id"] = run_id
            self._persist()
        self._set_phase(run_id, "PLAN")
        self.next_action(run_id)
        return self.status(run_id)

    @staticmethod
    def _done_checklist() -> dict[str, bool | None]:
        return {
            "requirements": False,
            "build": None,
            "syntax": None,
            "lint": None,
            "unit_tests": None,
            "integration_tests": None,
            "runtime": None,
            "gui": None,
            "interaction": None,
            "evidence": None,
            "regression": None,
            "security": None,
            "git_diff": None,
            "commit": None,
            "persistence": False,
        }

    def _run(self, run_id: str | None = None) -> dict[str, Any]:
        rid = run_id or self.state.get("active_run_id")
        if not rid or rid not in self.state["runs"]:
            raise KeyError(run_id or "active reasoning run")
        return self.state["runs"][rid]

    def status(self, run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            return redact(dict(self._run(run_id)))

    def _set_phase(self, run_id: str, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(phase)
        run = self._run(run_id)
        run["phase"] = phase
        run["updated_at"] = now()
        run["history"].append({"timestamp": run["updated_at"], "event": "phase", "phase": phase})
        self._persist()

    @staticmethod
    def _deps_completed(run: dict[str, Any], step: dict[str, Any]) -> bool:
        states = {s["id"]: s["status"] for s in run["plan"]}
        return all(states.get(dep) in {"COMPLETED", "SKIPPED"} for dep in step.get("dependencies", []))

    def next_action(self, run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            for step in run["plan"]:
                if step["status"] == "COMPLETED" or step["status"] == "SKIPPED":
                    continue
                if step["status"] in {"PENDING", "READY", "FAILED"} and self._deps_completed(run, step):
                    if step["status"] == "FAILED" and step["attempts"] >= step["max_attempts"]:
                        continue
                    if step["status"] != "FAILED":
                        step["status"] = "READY"
                    run["current_step"] = step["id"]
                    run["phase"] = "EXECUTE"
                    run["updated_at"] = now()
                    self._persist()
                    return {"ok": True, "action": "execute", "step": redact(dict(step)), "run_id": run["run_id"]}
            unfinished = [s for s in run["plan"] if s["status"] not in {"COMPLETED", "SKIPPED"}]
            if not unfinished:
                run["phase"] = "VERIFY"
                run["current_step"] = None
                run["updated_at"] = now()
                self._persist()
                return {"ok": True, "action": "verify", "run_id": run["run_id"]}
            blocked = [s for s in unfinished if s["status"] == "BLOCKED" or not self._deps_completed(run, s)]
            run["phase"] = "REPLAN" if blocked else "DIAGNOSE"
            run["updated_at"] = now()
            self._persist()
            return {"ok": True, "action": "replan", "run_id": run["run_id"], "blocked_steps": [s["id"] for s in blocked]}

    def mark_step(self, step_id: str, status: str, *, run_id: str | None = None, result=None, error=None, evidence=None, metadata=None) -> dict[str, Any]:
        if status not in STEP_STATES:
            raise ValueError(status)
        with self._lock:
            run = self._run(run_id)
            step = next((s for s in run["plan"] if s["id"] == step_id), None)
            if step is None:
                raise KeyError(step_id)
            if step["status"] == "COMPLETED" and status != "COMPLETED":
                return {"ok": False, "error": "completed_step_protected", "step": redact(dict(step))}
            if status == "RUNNING" and not self._deps_completed(run, step):
                raise ValueError("step dependencies are not completed")
            if status == "RUNNING":
                step["attempts"] += 1
            step["status"] = status
            if result is not None:
                step["result"] = redact(result)
            if error:
                step["error"] = redact(error)
                run["errors"].append({"step": step_id, "error": redact(error), "timestamp": now()})
            if evidence:
                refs = list(evidence) if isinstance(evidence, list) else [evidence]
                step["evidence"].extend(redact(x) for x in refs)
                run["evidence"].extend(redact(x) for x in refs)
            if metadata:
                step["metadata"].update(redact(metadata))
            if status == "FAILED":
                run["phase"] = "DIAGNOSE"
            elif status in {"COMPLETED", "SKIPPED"}:
                run["phase"] = "TEST" if any(s["status"] not in {"COMPLETED", "SKIPPED"} for s in run["plan"]) else "VERIFY"
            run["current_step"] = step_id if status not in {"COMPLETED", "SKIPPED"} else None
            run["updated_at"] = now()
            self._persist()
            return {"ok": True, "step": redact(dict(step)), "run_id": run["run_id"]}

    def observe(self, observation: Any, *, run_id: str | None = None, evidence=None, phase: str | None = None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            row = {"timestamp": now(), "observation": redact(observation), "step": run.get("current_step")}
            run["observations"].append(row)
            if evidence:
                refs = list(evidence) if isinstance(evidence, list) else [evidence]
                run["evidence"].extend(redact(x) for x in refs)
            if phase:
                if phase not in PHASES:
                    raise ValueError(phase)
                run["phase"] = phase
            elif run["phase"] == "EXECUTE":
                run["phase"] = "OBSERVE"
            run["updated_at"] = now()
            self._persist()
            return self.status(run["run_id"])

    @staticmethod
    def _classify(error: Any) -> str:
        text = str(error or "").lower()
        checks = {
            "resource_limit": ("oom", "out of memory", "exit 137", "killed"),
            "timeout": ("timeout", "timed out"),
            "permission": ("permission", "forbidden", "unauthorized", "scope"),
            "network": ("network", "connection", "http", "dns"),
            "dependency": ("not found", "missing", "dependency", "importerror"),
            "verification": ("verification", "test failed", "assertion"),
            "input": ("invalid", "schema"),
        }
        for category, needles in checks.items():
            if any(n in text for n in needles):
                return category
        return "tool_error"

    def feedback(self, *, operation: str, success: bool, result=None, error=None, tool=None, task=None, step=None, evidence=None, metadata=None, run_id=None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            row = {
                "operation": operation,
                "success": bool(success),
                "result": redact(result),
                "error": redact(error),
                "tool": tool,
                "task": task,
                "step": step or run.get("current_step"),
                "timestamp": now(),
                "metadata": redact(metadata or {}),
                "evidence": redact(list(evidence) if isinstance(evidence, list) else ([evidence] if evidence else [])),
            }
            run["feedback"].append(row)
            if not success and error:
                run["errors"].append({"step": row["step"], "error": redact(error), "classification": self._classify(error), "timestamp": row["timestamp"]})
                run["phase"] = "DIAGNOSE"
            elif success:
                run["phase"] = "OBSERVE"
            run["updated_at"] = now()
            self._persist()
            return row

    def replan(self, *, reason: str | None = None, run_id=None, strategy: str | None = None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            step = next((s for s in run["plan"] if s["id"] == run.get("current_step")), None)
            classification = self._classify(reason)
            if step and step["status"] == "FAILED" and step["attempts"] < step["max_attempts"]:
                step["status"] = "READY"
            elif step and step["attempts"] >= step["max_attempts"]:
                fallback_id = f"fallback-{uuid.uuid4().hex[:8]}"
                fallback_op = "analyze" if classification in {"resource_limit", "dependency"} else step.get("operation", "analyze")
                fallback = self._normalize_step({
                    "id": fallback_id,
                    "title": f"fallback after {classification}",
                    "kind": "fallback",
                    "operation": fallback_op,
                    "args": step.get("args", {}),
                    "dependencies": [d for d in step.get("dependencies", []) if d != step["id"]],
                    "max_attempts": 1,
                    "metadata": {"fallback_for": step["id"], "classification": classification, "strategy": strategy or "automatic"},
                }, len(run["plan"]))
                if not any(s["id"] == fallback_id for s in run["plan"]):
                    run["plan"].append(fallback)
                step["status"] = "SKIPPED"
            run["phase"] = "REPLAN"
            run["updated_at"] = now()
            run["history"].append({"timestamp": run["updated_at"], "event": "replan", "reason": redact(reason), "classification": classification, "strategy": strategy or "automatic"})
            self._persist()
            return self.next_action(run["run_id"])

    def finish(self, *, verified: bool = False, result=None, run_id=None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            incomplete = [s["id"] for s in run["plan"] if s["status"] not in {"COMPLETED", "SKIPPED"}]
            if not verified or incomplete:
                raise ValueError("reasoning run cannot finish before verification and completion")
            run["phase"] = "DONE"
            run["status"] = "COMPLETED"
            run["current_step"] = None
            run["metadata"]["verified"] = True
            run["metadata"]["result"] = redact(result)
            run["definition_of_done"]["requirements"] = True
            run["definition_of_done"]["persistence"] = True
            run["updated_at"] = now()
            self.state["active_run_id"] = None
            self._persist()
            return self.status(run["run_id"])

    def fail(self, reason: str, *, run_id=None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            run["phase"] = "FAILED"
            run["status"] = "FAILED"
            run["errors"].append({"timestamp": now(), "error": redact(reason), "classification": self._classify(reason)})
            run["updated_at"] = now()
            self.state["active_run_id"] = None
            self._persist()
            return self.status(run["run_id"])

    def mark_done_criterion(self, name: str, value: bool | None, *, run_id=None) -> dict[str, Any]:
        with self._lock:
            run = self._run(run_id)
            if name not in run["definition_of_done"]:
                raise KeyError(name)
            run["definition_of_done"][name] = value
            run["updated_at"] = now()
            self._persist()
            return self.status(run["run_id"])
