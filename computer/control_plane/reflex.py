from __future__ import annotations

import hashlib
import hmac
import json
import threading
from typing import Any, Callable

from .store import load_json, now, redact, save_json

FILE = "reflex.json"


class ReflexEngine:
    """Durable event/rule engine for event-driven autonomous continuation.

    Events are idempotent, persisted before dispatch, and processed serially so
    duplicate webhook deliveries or concurrent workers cannot run the same event
    twice. Failed actions remain retryable and eventually move to a dead letter
    state instead of disappearing.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self.state = load_json(FILE, {"version": 1, "rules": {}, "events": {}, "history": []})
        self.state.setdefault("version", 1)
        self.state.setdefault("rules", {})
        self.state.setdefault("events", {})
        self.state.setdefault("history", [])

    def _save(self) -> None:
        save_json(FILE, self.state)

    def add_rule(
        self,
        name: str,
        *,
        source: str = "*",
        event_type: str = "*",
        action: dict[str, Any] | None = None,
        enabled: bool = True,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        if not name or len(name) > 120:
            raise ValueError("rule name required (max 120 chars)")
        if not isinstance(action, dict) or not action.get("type"):
            raise ValueError("action.type is required")
        if action.get("type") not in {"autonomous_goal", "reasoning_goal", "record_experience"}:
            raise ValueError("unsupported reflex action type")
        row = {
            "name": name,
            "source": source or "*",
            "event_type": event_type or "*",
            "action": redact(action),
            "enabled": bool(enabled),
            "max_retries": max(0, min(int(max_retries), 20)),
            "created_at": now(),
            "updated_at": now(),
        }
        with self._lock:
            self.state["rules"][name] = row
            self._save()
        return redact(row)

    def remove_rule(self, name: str) -> dict[str, Any]:
        with self._lock:
            existed = self.state["rules"].pop(name, None) is not None
            self._save()
        return {"ok": existed, "name": name}

    @staticmethod
    def _event_key(source: str, event_type: str, payload: Any, event_id: str | None) -> str:
        if event_id:
            return str(event_id)[:200]
        raw = json.dumps(
            {"source": source, "event_type": event_type, "payload": redact(payload)},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def emit(self, source: str, event_type: str, payload: Any, event_id: str | None = None) -> dict[str, Any]:
        if not source or not event_type:
            raise ValueError("source and event_type are required")
        key = self._event_key(source, event_type, payload, event_id)
        with self._lock:
            if key in self.state["events"]:
                return {"ok": True, "duplicate": True, "event": redact(self.state["events"][key])}
            row = {
                "id": key,
                "source": str(source),
                "event_type": str(event_type),
                "payload": redact(payload),
                "status": "pending",
                "attempts": 0,
                "matched_rules": [],
                "results": [],
                "created_at": now(),
                "updated_at": now(),
            }
            self.state["events"][key] = row
            self._save()
        return {"ok": True, "duplicate": False, "event": redact(row)}

    @staticmethod
    def _matches(rule: dict[str, Any], event: dict[str, Any]) -> bool:
        return (
            rule.get("enabled", True)
            and rule.get("source", "*") in {"*", event.get("source")}
            and rule.get("event_type", "*") in {"*", event.get("event_type")}
        )

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = [e for e in self.state["events"].values() if e.get("status") in {"pending", "retrying"}]
            rows.sort(key=lambda x: x.get("created_at", 0))
            return redact(rows[: max(1, min(int(limit), 500))])

    def process(self, dispatcher: Callable[[dict[str, Any], dict[str, Any]], Any], limit: int = 20) -> dict[str, Any]:
        processed: list[dict[str, Any]] = []
        # One processor per runtime. This makes webhook bursts deterministic and
        # avoids two workers executing the same rule at the same time.
        if not self._process_lock.acquire(blocking=False):
            return {"ok": True, "processed": [], "busy": True, "status": self.status()}
        try:
            for snapshot in self.pending(limit):
                event_id = snapshot["id"]
                with self._lock:
                    event = self.state["events"].get(event_id)
                    if not event or event.get("status") not in {"pending", "retrying"}:
                        continue
                    matches = [dict(r) for r in self.state["rules"].values() if self._matches(r, event)]
                    event["matched_rules"] = [r["name"] for r in matches]
                    event["status"] = "running"
                    event["updated_at"] = now()
                    self._save()
                if not matches:
                    with self._lock:
                        event["status"] = "ignored"
                        event["updated_at"] = now()
                        self._save()
                    processed.append(redact(event))
                    continue
                # Successful rules are idempotently skipped on later retry cycles.
                # This prevents a transient failure in one rule from repeating side
                # effects that another rule already completed successfully.
                completed_rules = {
                    str(item.get("rule"))
                    for item in event.get("results", [])
                    if item.get("ok") is True and item.get("rule")
                }
                def failure_count(rule_name: str) -> int:
                    return sum(1 for item in event.get("results", []) if item.get("rule") == rule_name and item.get("ok") is False)
                pending_rules = [
                    rule for rule in matches
                    if rule["name"] not in completed_rules
                    and failure_count(rule["name"]) <= int(rule.get("max_retries", 3))
                ]
                for rule in pending_rules:
                    try:
                        result = dispatcher(dict(rule["action"]), redact(dict(event)))
                        with self._lock:
                            event["results"].append({"rule": rule["name"], "ok": True, "result": redact(result), "at": now()})
                    except Exception as exc:  # durable + retryable; never drop an event
                        with self._lock:
                            event["results"].append({"rule": rule["name"], "ok": False, "error": str(exc), "at": now()})
                with self._lock:
                    event["attempts"] = int(event.get("attempts", 0)) + 1
                    completed_rules = {
                        str(item.get("rule"))
                        for item in event.get("results", [])
                        if item.get("ok") is True and item.get("rule")
                    }
                    retryable = [
                        rule["name"] for rule in matches
                        if rule["name"] not in completed_rules
                        and failure_count(rule["name"]) <= int(rule.get("max_retries", 3))
                    ]
                    exhausted = [rule["name"] for rule in matches if rule["name"] not in completed_rules and rule["name"] not in retryable]
                    event["exhausted_rules"] = exhausted
                    if len(completed_rules) == len(matches):
                        event["status"] = "completed"
                    elif retryable:
                        event["status"] = "retrying"
                    else:
                        event["status"] = "dead_letter"
                    event["updated_at"] = now()
                    self.state["history"].append({"event_id": event_id, "status": event["status"], "at": now()})
                    self.state["history"] = self.state["history"][-1000:]
                    self._save()
                processed.append(redact(event))
        finally:
            self._process_lock.release()
        return {"ok": True, "processed": processed, "busy": False, "status": self.status()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for event in self.state["events"].values():
                key = event.get("status", "unknown")
                counts[key] = counts.get(key, 0) + 1
            return {"ok": True, "rules": len(self.state["rules"]), "events": len(self.state["events"]), "counts": counts}


def verify_github_signature(secret: str, body: bytes, signature: str | None) -> bool:
    if not secret or not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest("sha256=" + expected, signature)
