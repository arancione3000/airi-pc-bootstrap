from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "AIRI_TOOL_MANIFEST.json"
with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
    MANIFEST = json.load(fh)

TOOLS = list(MANIFEST["tools"])
TOOL_COUNT = int(MANIFEST["tool_count"])
if TOOL_COUNT != 83 or len(TOOLS) != 83 or len(set(TOOLS)) != 83:
    raise RuntimeError("AIRI_TOOL_MANIFEST_INVALID")

import server as _legacy
app: FastAPI = _legacy.app


def _obj(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return schema


SCHEMAS: dict[str, dict[str, Any]] = {
    "computer_find_text": _obj({"text": {"type": "string"}, "click": {"type": "boolean"}}, ["text"]),
    "computer_click_element": _obj({"text": {"type": "string"}}, ["text"]),
    "computer_click": _obj({"x": {"type": "integer"}, "y": {"type": "integer"}, "button": {"type": "string"}, "clicks": {"type": "integer"}}, ["x", "y"]),
    "computer_double_click": _obj({"x": {"type": "integer"}, "y": {"type": "integer"}, "button": {"type": "string"}}, ["x", "y"]),
    "computer_move": _obj({"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"]),
    "computer_drag": _obj({"x1": {"type": "integer"}, "y1": {"type": "integer"}, "x2": {"type": "integer"}, "y2": {"type": "integer"}, "duration": {"type": "number"}}, ["x1", "y1", "x2", "y2"]),
    "computer_scroll": _obj({"amount": {"type": "integer"}}, ["amount"]),
    "computer_key": _obj({"key": {"type": "string"}}, ["key"]),
    "computer_hotkey": _obj({"keys": {"type": "array", "items": {"type": "string"}}}, ["keys"]),
    "computer_type": _obj({"text": {"type": "string"}, "interval": {"type": "number"}}, ["text"]),
    "computer_wait": _obj({"seconds": {"type": "number"}}, ["seconds"]),
    "computer_browser_open": _obj({"url": {"type": "string"}, "wait_until": {"type": "string"}}, ["url"]),
    "computer_act_verify": _obj({"action": {"type": "string"}, "payload": {"type": "object"}, "confirmation": {"type": ["string", "null"]}}, ["action"]),
    "computer_cleanup_safe": _obj({"max_bytes": {"type": ["integer", "null"]}}),
    "computer_file_read": _obj({"path": {"type": "string"}}, ["path"]),
    "computer_file_search": _obj({"query": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
    "computer_file_write": _obj({"path": {"type": "string"}, "content": {"type": "string"}, "scope": {"type": "array", "items": {"type": "string"}}}, ["path", "content", "scope"]),
    "computer_file_patch": _obj({"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "replace_all": {"type": "boolean"}, "scope": {"type": "array", "items": {"type": "string"}}}, ["path", "old", "new", "scope"]),
    "computer_terminal_run": _obj({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}, "allow_shell": {"type": "boolean"}, "scope": {"type": "array", "items": {"type": "string"}}}, ["command", "scope"]),
    "computer_test_run": _obj({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}}),
    "computer_build_run": _obj({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}}),
    "computer_lint": _obj({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}}),
    "computer_git_status": _obj({"path": {"type": "string"}}),
    "computer_git_diff": _obj({"path": {"type": "string"}}),
    "computer_git_log": _obj({"path": {"type": "string"}, "n": {"type": "integer"}}),
    "computer_git_commit": _obj({"message": {"type": "string"}, "path": {"type": "string"}}, ["message"]),
    "computer_skill_load": _obj({"name": {"type": "string"}}, ["name"]),
    "computer_skill_create": _obj({"name": {"type": "string"}, "description": {"type": "string"}, "instructions": {"type": "string"}, "tools": {"type": "array", "items": {"type": "string"}}}, ["name", "description", "instructions"]),
    "computer_skill_update": _obj({"name": {"type": "string"}, "content": {"type": "string"}}, ["name", "content"]),
    "computer_skill_test": _obj({"name": {"type": "string"}}, ["name"]),
    "computer_skill_delete": _obj({"name": {"type": "string"}}, ["name"]),
    "computer_project_memory_update": _obj({"entry": {"type": "string"}}, ["entry"]),
    "computer_code_apply_fix": _obj({"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "test_command": {"type": "string"}}, ["path", "old", "new"]),
    "computer_code_verify_change": _obj({"path": {"type": "string"}, "test_command": {"type": "string"}}, ["path", "test_command"]),
    "computer_code_agent": _obj({"goal": {"type": "string"}, "project_path": {"type": "string"}, "max_attempts": {"type": "integer"}, "steps": {"type": "array"}, "scope": {"type": "array"}, "changes": {"type": "array"}, "test_command": {"type": "string"}}, ["goal"]),
    "computer_project_context": _obj({"path": {"type": "string"}}),
    "computer_task_start": _obj({"goal": {"type": "string"}, "steps": {"type": "array"}, "scope": {"type": "array"}}, ["goal", "steps"]),
    "computer_task_update": _obj({"index": {"type": "integer"}, "status": {"type": "string"}, "note": {"type": "string"}}, ["index", "status"]),
    "computer_task_finish": _obj({"outcome": {"type": "string"}, "note": {"type": "string"}}),
    "computer_scope_check": _obj({"paths": {"type": "array"}, "scope": {"type": "array"}}, ["paths", "scope"]),
    "computer_diff_summary": _obj({"path": {"type": "string"}, "scope": {"type": "array"}, "allow_test_changes": {"type": "boolean"}, "allow_security_changes": {"type": "boolean"}}),
    "computer_guardrails": _obj({"path": {"type": "string"}, "scope": {"type": "array"}, "allow_test_changes": {"type": "boolean"}, "allow_security_changes": {"type": "boolean"}}),
    "computer_snapshot": _obj({"paths": {"type": "array"}, "label": {"type": "string"}}, ["paths"]),
    "computer_restore_snapshot": _obj({"snapshot": {"type": "object"}}, ["snapshot"]),
    "computer_prepare_commit": _obj({"path": {"type": "string"}, "scope": {"type": "array"}}),
    "computer_code_commit": _obj({"message": {"type": "string"}, "project_path": {"type": "string"}, "scope": {"type": "array"}, "allow_test_changes": {"type": "boolean"}, "allow_security_changes": {"type": "boolean"}}, ["message"]),
    "computer_autonomous_cycle": _obj({"changes": {"type": "array"}, "project_path": {"type": "string"}, "test_command": {"type": "string"}, "scope": {"type": "array"}, "max_attempts": {"type": "integer"}}, ["changes"]),
    "computer_recovery_checkpoint": _obj({"goal": {"type": "string"}, "scope": {"type": "array"}, "step": {"type": "integer"}, "note": {"type": "string"}, "artifacts": {"type": "array"}, "status": {"type": "string"}}, ["goal"]),
    "computer_recovery_finish": _obj({"status": {"type": "string"}, "note": {"type": "string"}}),
    "computer_decision_record": _obj({"decision": {"type": "string"}, "reason": {"type": "string"}, "evidence": {"type": "array"}, "files": {"type": "array"}, "commit": {"type": "string"}, "result": {"type": "string"}}, ["decision"]),
    "computer_decisions": _obj({"limit": {"type": "integer"}}),
    "computer_persist": _obj({"message": {"type": "string"}, "branch": {"type": ["string", "null"]}, "push": {"type": "boolean"}, "scope": {"type": "array"}}, ["message"]),
    "computer_research": _obj({"topic": {"type": "string"}, "urls": {"type": "array"}, "max_sources": {"type": "integer"}}, ["topic"]),
    "computer_backup_prune": _obj({"max_entries": {"type": "integer"}, "max_age_days": {"type": "integer"}, "dry_run": {"type": "boolean"}}),
    "computer_scheduler_schedule": _obj({"name": {"type": "string"}, "action": {"type": "string"}, "interval_seconds": {"type": "integer"}, "run_now": {"type": "boolean"}}, ["name", "action", "interval_seconds"]),
    "computer_scheduler_cancel": _obj({"name": {"type": "string"}}, ["name"]),
    "computer_browser_auth_set": _obj({"profile": {"type": "string"}}, ["profile"]),
    "computer_browser_auth_save": _obj({"profile": {"type": ["string", "null"]}}),
    "computer_browser_auth_status": _obj({"profile": {"type": "string"}}, ["profile"]),
    "computer_browser_human_wait": _obj({"timeout": {"type": "integer"}, "poll": {"type": "number"}}),
}


def tool_specs() -> list[dict[str, Any]]:
    return [{"name": name, "description": f"Airi-PC canonical tool: {name}", "inputSchema": SCHEMAS.get(name, _obj())} for name in TOOLS]


def _remove_routes(paths: set[str]) -> None:
    app.router.routes[:] = [route for route in app.router.routes if not (isinstance(route, APIRoute) and route.path in paths)]


def _ready_payload() -> dict[str, Any]:
    checks = {"status": False, "gui": False, "browser": False, "mcp": False, "source_match": False}
    try:
        st = _legacy.status()
        checks["status"] = bool(st.get("ok"))
        checks["gui"] = bool(st.get("gui_available")) and st.get("resolution") == "1280x800" and os.environ.get("DISPLAY", ":99") == ":99"
    except Exception:
        pass
    try:
        bs = _legacy.browser().status()
        checks["browser"] = bool(bs.get("available")) and bool(bs.get("open")) and not bs.get("error")
    except Exception:
        pass
    checks["mcp"] = TOOL_COUNT == 83 and len(TOOLS) == 83 and len(set(TOOLS)) == 83 and all(spec["inputSchema"].get("type") == "object" for spec in tool_specs())
    expected_sha = os.environ.get("AIRI_EXPECTED_SHA", "").strip()
    actual_sha = os.environ.get("AIRI_BOOTSTRAP_SHA", "").strip()
    checks["source_match"] = bool(actual_sha) and (not expected_sha or actual_sha == expected_sha)
    ready = all(checks.values())
    return {"ready": ready, "checks": checks, "source_sha": actual_sha or None}


_remove_routes({"/ready", "/tools", "/mcp"})

@app.get("/ready")
def ready_contract() -> JSONResponse:
    payload = _ready_payload()
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)

@app.get("/tools")
def tools_contract() -> dict[str, Any]:
    specs = tool_specs()
    return {"name": "Airi Computer", "version": MANIFEST["runtime_version"], "tool_count": TOOL_COUNT, "tools": specs}

@app.post("/mcp")
def mcp_contract(req: dict[str, Any]) -> dict[str, Any]:
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "Airi Computer", "version": MANIFEST["runtime_version"]}}}
    if method in ("notifications/initialized", "ping"):
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tool_specs()}}
    if method == "tools/call":
        name = req.get("params", {}).get("name")
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Tool not found"}}
        return _legacy.mcp(req)
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
