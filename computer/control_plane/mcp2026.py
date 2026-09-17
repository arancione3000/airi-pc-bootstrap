from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import sys
import threading
import time
import urllib.parse
from collections import deque
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .judge import TrustedJudge
from .mcp_tasks import MCPTaskStore
from .reflex import ReflexEngine, verify_github_signature
from .resource_discovery import ResourceDiscovery

PROTOCOL = "2026-07-28"
LEGACY_PROTOCOL = "2025-06-18"
SERVER_INFO = {"name": "Airi Computer", "version": "2.1"}
SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
SUBSCRIPTION_ID_KEY = "io.modelcontextprotocol/subscriptionId"
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
CLIENT_CAPS_KEY = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
TASKS_EXT = "io.modelcontextprotocol/tasks"
HEADER_MISMATCH = -32020
UNSUPPORTED_PROTOCOL = -32022
TASK_CAPABILITY_MISSING = -32021
MISSING_CLIENT_CAPABILITY = -32021

REFLEX = ReflexEngine()
DISCOVERY = ResourceDiscovery()
JUDGE = TrustedJudge()
MCP_TASKS = MCPTaskStore()
_REFLEX_THREAD_LOCK = threading.Lock()
_REFLEX_THREAD: threading.Thread | None = None

NEW_TOOLS = [
    {"name": "computer_reflex_status", "description": "Inspect Airi Reflex durable event state", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "computer_reflex_rule_add", "description": "Add a durable event-to-action rule", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "source": {"type": "string"}, "event_type": {"type": "string"}, "action": {"type": "object"}, "max_retries": {"type": "integer"}}, "required": ["name", "action"]}},
    {"name": "computer_reflex_emit", "description": "Emit a durable event into Airi Reflex", "inputSchema": {"type": "object", "properties": {"source": {"type": "string"}, "event_type": {"type": "string"}, "payload": {}, "event_id": {"type": "string"}}, "required": ["source", "event_type"]}},
    {"name": "computer_reflex_process", "description": "Process pending Airi Reflex events", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "computer_resource_discover", "description": "Discover MCP servers, skills and agents through ARD registries", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "registries": {"type": "array", "items": {"type": "string"}}, "filter": {"type": "object"}, "limit": {"type": "integer"}}, "required": ["query"]}},
    {"name": "computer_resource_stage", "description": "Quarantine or approve-source-stage an ARD candidate without executing it", "inputSchema": {"type": "object", "properties": {"candidate": {"type": "object"}, "allowed_publishers": {"type": "array", "items": {"type": "string"}}}, "required": ["candidate"]}},
    {"name": "computer_resource_acquire_skill", "description": "Safely quarantine, validate and install a trusted Markdown skill discovered through ARD", "inputSchema": {"type": "object", "properties": {"candidate": {"type": "object"}, "allowed_publishers": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "integer"}}, "required": ["candidate"]}},
    {"name": "computer_resource_staged", "description": "List staged ARD resources", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "computer_judge", "description": "Run the independent fail-closed Airi Judge verifier in an isolated sandbox", "inputSchema": {"type": "object", "properties": {"goal": {"type": "string"}, "project_path": {"type": "string"}, "checks": {"type": "array"}, "forbidden_paths": {"type": "array"}}, "required": ["goal", "checks"]}},
]
NEW_TOOL_NAMES = {item["name"] for item in NEW_TOOLS}
TASK_CAPABLE_TOOLS = {"computer_autonomous_goal"}


class _TaskEventHub:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._seq = 0
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=1000)

    def publish(self, task: dict[str, Any]) -> None:
        with self._condition:
            self._seq += 1
            self._events.append((self._seq, dict(task)))
            self._condition.notify_all()

    def cursor(self) -> int:
        with self._condition:
            return self._seq

    def wait(self, after: int, timeout: float = 15.0) -> tuple[int, list[dict[str, Any]]]:
        with self._condition:
            if self._seq <= after:
                self._condition.wait(timeout=timeout)
            rows = [item for seq, item in self._events if seq > after]
            return self._seq, rows


TASK_EVENTS = _TaskEventHub()
MCP_TASKS.add_listener(TASK_EVENTS.publish)


def _meta(req: dict[str, Any]) -> dict[str, Any]:
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    return params.get("_meta") if isinstance(params.get("_meta"), dict) else {}


def _modern(req: dict[str, Any], request: Request) -> bool:
    header_version = request.headers.get("mcp-protocol-version")
    body_version = _meta(req).get(VERSION_KEY)
    return (
        req.get("method") == "server/discover"
        or body_version is not None
        or (header_version is not None and header_version != LEGACY_PROTOCOL)
    )


def _client_tasks_enabled(req: dict[str, Any]) -> bool:
    caps = _meta(req).get(CLIENT_CAPS_KEY) or {}
    extensions = caps.get("extensions") if isinstance(caps, dict) else {}
    return isinstance(extensions, dict) and TASKS_EXT in extensions


def _principal_name(req: dict[str, Any]) -> str | None:
    """Return the value mirrored in the required Mcp-Name HTTP header.

    MCP 2026-07-28 requires Mcp-Name for tools/call, prompts/get and
    resources/read. Extension methods such as tasks/get do not use this header.
    """
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    method = req.get("method")
    if method in {"tools/call", "prompts/get"}:
        return str(params.get("name")) if params.get("name") is not None else None
    if method == "resources/read":
        return str(params.get("uri")) if params.get("uri") is not None else None
    return None


def _decode_mcp_header_value(value: str) -> str:
    """Decode MCP's =?base64?...?= sentinel used for non-ASCII names."""
    if not value.startswith("=?base64?") or not value.endswith("?="):
        return value
    encoded = value[len("=?base64?"):-2]
    try:
        raw = base64.b64decode(encoded, validate=True)
        return raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("malformed base64 MCP header value") from exc


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("result"), dict):
        payload["result"].setdefault("resultType", "complete")
        payload["result"].setdefault("_meta", {})[SERVER_INFO_KEY] = SERVER_INFO
    return payload


def _rpc_result(rid: Any, result: dict[str, Any], *, modern: bool = True) -> JSONResponse:
    payload = {"jsonrpc": "2.0", "id": rid, "result": result}
    headers = {"MCP-Protocol-Version": PROTOCOL} if modern else None
    return JSONResponse(_stamp(payload) if modern else payload, headers=headers)


def _rpc_error(
    rid: Any,
    code: int,
    message: str,
    data: Any = None,
    *,
    status: int = 200,
    modern: bool = True,
) -> JSONResponse:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    headers = {"MCP-Protocol-Version": PROTOCOL} if modern else None
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": error}, status_code=status, headers=headers)


def _server_module():
    return sys.modules.get("server") or sys.modules.get("computer.server")


def _control_plane():
    module = _server_module()
    if module is None or not hasattr(module, "_control_plane_bootstrap"):
        raise RuntimeError("Airi control plane is not ready")
    return module._control_plane_bootstrap()


def _dispatch_reflex(action: dict[str, Any], event: dict[str, Any]) -> Any:
    cp = _control_plane()
    kind = action.get("type")
    template = str(action.get("goal") or action.get("goal_template") or "Handle event {event_type} from {source}: {payload}")
    goal = template.format(
        event_type=event.get("event_type"),
        source=event.get("source"),
        payload=json.dumps(event.get("payload"), ensure_ascii=False),
    )
    if kind == "autonomous_goal":
        return cp.autonomous_goal(
            goal,
            scope=action.get("scope"),
            max_time=int(action.get("max_time", 900)),
            resume=True,
        )
    if kind == "reasoning_goal":
        return cp.reasoning_goal(goal, scope=action.get("scope"), metadata=action.get("metadata"))
    if kind == "record_experience":
        return cp.record_experience(
            goal,
            json.dumps(event, ensure_ascii=False),
            str(action.get("solution", "event observed")),
            tags=["reflex", str(event.get("event_type"))],
        )
    raise ValueError("unsupported reflex action")


def _process_reflex_background() -> None:
    global _REFLEX_THREAD
    with _REFLEX_THREAD_LOCK:
        if _REFLEX_THREAD and _REFLEX_THREAD.is_alive():
            return

        def worker() -> None:
            try:
                REFLEX.process(_dispatch_reflex, limit=50)
            finally:
                global _REFLEX_THREAD
                with _REFLEX_THREAD_LOCK:
                    _REFLEX_THREAD = None

        _REFLEX_THREAD = threading.Thread(target=worker, daemon=True, name="airi-reflex")
        _REFLEX_THREAD.start()


def _trusted_publishers(args: dict[str, Any]) -> list[str]:
    explicit = args.get("allowed_publishers")
    if explicit is not None:
        return [str(item).strip() for item in explicit if str(item).strip()]
    return [item.strip() for item in os.environ.get("AIRI_ARD_TRUSTED_PUBLISHERS", "").split(",") if item.strip()]


def _call_new_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "computer_reflex_status":
        return REFLEX.status()
    if name == "computer_reflex_rule_add":
        return REFLEX.add_rule(
            args["name"],
            source=args.get("source", "*"),
            event_type=args.get("event_type", "*"),
            action=args["action"],
            max_retries=args.get("max_retries", 3),
        )
    if name == "computer_reflex_emit":
        row = REFLEX.emit(args["source"], args["event_type"], args.get("payload", {}), args.get("event_id"))
        if not row.get("duplicate"):
            _process_reflex_background()
        return row
    if name == "computer_reflex_process":
        return REFLEX.process(_dispatch_reflex, args.get("limit", 20))
    if name == "computer_resource_discover":
        return DISCOVERY.search(args["query"], args.get("registries"), args.get("filter"), args.get("limit", 8))
    if name == "computer_resource_stage":
        return DISCOVERY.stage(args["candidate"], allowed_publishers=_trusted_publishers(args))
    if name == "computer_resource_acquire_skill":
        result = DISCOVERY.acquire_skill(args["candidate"], allowed_publishers=_trusted_publishers(args), timeout=args.get("timeout", 15))
        if result.get("status") == "installed_markdown_skill":
            try:
                _control_plane().skills.refresh()
            except Exception:
                # Installation remains durable even if a registry refresh fails;
                # the next normal Control Plane bootstrap refreshes it again.
                pass
        return result
    if name == "computer_resource_staged":
        return DISCOVERY.list_staged()
    if name == "computer_judge":
        return JUDGE.evaluate(
            goal=args["goal"],
            project_path=args.get("project_path", "."),
            checks=args.get("checks"),
            forbidden_paths=args.get("forbidden_paths"),
        )
    raise KeyError(name)


def _dispatch_task_tool(name: str, args: dict[str, Any]) -> Any:
    cp = _control_plane()
    if name == "computer_autonomous_goal":
        return cp.autonomous_goal(
            args["goal"], args.get("steps"), args.get("scope"), args.get("max_time", 900),
            args.get("max_iterations", 25), args.get("max_retries", 3), args.get("max_tool_calls", 100),
            args.get("max_parallel_tasks", 1), args.get("resume", True),
        )
    if name == "computer_reasoning_goal":
        return cp.reasoning_goal(
            args["goal"], args.get("steps") or args.get("plan"), args.get("scope"), args.get("metadata"),
            args.get("max_time", 900), args.get("max_iterations", 25), args.get("max_retries", 3),
            args.get("max_tool_calls", 100), args.get("resume", True),
        )
    raise KeyError(name)


def _discover_result() -> dict[str, Any]:
    return {
        "resultType": "complete",
        "supportedVersions": [PROTOCOL],
        "capabilities": {
            "tools": {"listChanged": True},
            "extensions": {TASKS_EXT: {"get": {}, "update": {}, "cancel": {}, "notifications": {}}},
        },
        "instructions": "Airi-PC autonomous computer-use server with durable Tasks, Reflex events, ARD discovery and independent sandbox verification.",
        "ttlMs": 30_000,
        "cacheScope": "private",
    }


def _validate_modern(req: dict[str, Any], request: Request) -> JSONResponse | None:
    rid = req.get("id")
    meta = _meta(req)
    body_version = meta.get(VERSION_KEY)
    header_version = request.headers.get("mcp-protocol-version")

    # An unknown modern-era version in the transport header is itself enough
    # to identify a modern request and must not silently fall back to legacy.
    if header_version and header_version not in {PROTOCOL, LEGACY_PROTOCOL} and (body_version in {None, header_version}):
        return _rpc_error(
            rid,
            UNSUPPORTED_PROTOCOL,
            "Unsupported MCP protocol version",
            {"requested": header_version, "supported": [PROTOCOL]},
            status=400,
        )

    # Required per-request metadata is a JSON-RPC Invalid params error, while
    # unsupported versions have their own protocol error code.
    if not isinstance(body_version, str) or not body_version:
        return _rpc_error(rid, -32602, "Missing required request _meta protocolVersion", status=400)
    if body_version != PROTOCOL:
        return _rpc_error(
            rid,
            UNSUPPORTED_PROTOCOL,
            "Unsupported MCP protocol version",
            {"requested": body_version, "supported": [PROTOCOL]},
            status=400,
        )
    if CLIENT_CAPS_KEY not in meta or not isinstance(meta.get(CLIENT_CAPS_KEY), dict):
        return _rpc_error(rid, -32602, "Missing required request _meta clientCapabilities", status=400)

    # Streamable HTTP requires these routing headers on every modern request.
    if not header_version or header_version != body_version:
        return _rpc_error(rid, HEADER_MISMATCH, "MCP-Protocol-Version header is required and must match request _meta", status=400)
    header_method = request.headers.get("mcp-method")
    if not header_method or header_method != req.get("method"):
        return _rpc_error(rid, HEADER_MISMATCH, "Mcp-Method is required and must match the JSON-RPC method", status=400)

    principal = _principal_name(req)
    if principal is not None:
        header_name = request.headers.get("mcp-name")
        if not header_name:
            return _rpc_error(rid, HEADER_MISMATCH, "Mcp-Name is required for this method", status=400)
        try:
            decoded = _decode_mcp_header_value(header_name)
        except ValueError as exc:
            return _rpc_error(rid, HEADER_MISMATCH, str(exc), status=400)
        if decoded != principal:
            return _rpc_error(rid, HEADER_MISMATCH, "Mcp-Name header does not match the request body", status=400)
    return None


def _subscription_stream(rid: Any, notifications: dict[str, Any], *, allow_tasks: bool):
    requested_task_ids = [str(x) for x in notifications.get("taskIds", [])] if isinstance(notifications.get("taskIds"), list) else []
    standard = {key: value for key, value in notifications.items() if key in {"toolsListChanged", "promptsListChanged", "resourcesListChanged", "resourceSubscriptions"}}
    if allow_tasks and requested_task_ids:
        standard["taskIds"] = requested_task_ids
    subscription_id = rid

    async def generate():
        # Take the cursor before acknowledgement so a task transition occurring
        # while the client receives the ACK cannot be lost.
        cursor = TASK_EVENTS.cursor()
        ack = {
            "jsonrpc": "2.0",
            "method": "notifications/subscriptions/acknowledged",
            "params": {"notifications": standard, "_meta": {SUBSCRIPTION_ID_KEY: subscription_id}},
        }
        yield "data: " + json.dumps(ack, separators=(",", ":")) + "\n\n"
        while True:
            cursor, events = await asyncio.to_thread(TASK_EVENTS.wait, cursor, 15.0)
            emitted = False
            for task in events:
                if requested_task_ids and str(task.get("taskId")) in requested_task_ids:
                    params = dict(task)
                    params.pop("resultType", None)
                    params.setdefault("_meta", {})[SUBSCRIPTION_ID_KEY] = subscription_id
                    notice = {"jsonrpc": "2.0", "method": "notifications/tasks", "params": params}
                    emitted = True
                    yield "data: " + json.dumps(notice, separators=(",", ":"), default=str) + "\n\n"
            if not emitted:
                yield ": keepalive\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "MCP-Protocol-Version": PROTOCOL, "X-Accel-Buffering": "no"},
    )


def _origin_allowed(request: Request) -> bool:
    """Reject cross-site browser origins while preserving server-to-server MCP."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    try:
        parsed = urllib.parse.urlparse(origin)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    allowed = {
        "http://127.0.0.1:9010",
        "http://localhost:9010",
        "https://chatgpt.com",
    }
    allowed.update(x.strip().rstrip("/") for x in os.environ.get("AIRI_MCP_ALLOWED_ORIGINS", "").split(",") if x.strip())
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",", 1)[0].strip()
    if host:
        allowed.add(f"{proto}://{host}".rstrip("/"))
    return origin.rstrip("/") in allowed


def install(app: Any) -> None:
    """Install once. Legacy 2025 MCP remains intact; 2026 is additive/stateless."""
    if getattr(app.state, "airi_mcp2026_installed", False):
        return
    app.state.airi_mcp2026_installed = True

    @app.middleware("http")
    async def airi_modern_mcp(request: Request, call_next):
        if request.url.path == "/mcp" and not _origin_allowed(request):
            return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Origin not allowed"}}, status_code=403)

        if request.url.path == "/events/github" and request.method == "POST":
            body = await request.body()
            secret = os.environ.get("AIRI_GITHUB_WEBHOOK_SECRET", "")
            if not secret:
                return JSONResponse({"ok": False, "error": "github webhook secret not configured"}, status_code=503)
            if not verify_github_signature(secret, body, request.headers.get("x-hub-signature-256")):
                return JSONResponse({"ok": False, "error": "invalid signature"}, status_code=401)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)
            row = REFLEX.emit(
                "github",
                request.headers.get("x-github-event", "unknown"),
                payload,
                request.headers.get("x-github-delivery"),
            )
            if not row.get("duplicate"):
                _process_reflex_background()
            return JSONResponse(row)

        if request.url.path not in {"/mcp", "/tools", "/control-plane"}:
            return await call_next(request)

        req: dict[str, Any] = {}
        if request.url.path == "/mcp" and request.method == "POST":
            try:
                req = json.loads((await request.body()).decode("utf-8"))
            except Exception:
                req = {}
        modern = bool(req and _modern(req, request))

        if modern:
            invalid = _validate_modern(req, request)
            if invalid is not None:
                return invalid
            rid = req.get("id")
            method = req.get("method")
            if method == "server/discover":
                return _rpc_result(rid, _discover_result())
            if method == "subscriptions/listen":
                params = req.get("params") or {}
                notifications = params.get("notifications") or {}
                task_ids = notifications.get("taskIds") if isinstance(notifications, dict) else None
                if task_ids and not _client_tasks_enabled(req):
                    return _rpc_error(
                        rid,
                        TASK_CAPABILITY_MISSING,
                        "Missing required client capability",
                        {"requiredCapabilities": {"extensions": {TASKS_EXT: {}}}},
                    )
                return _subscription_stream(rid, notifications, allow_tasks=_client_tasks_enabled(req))
            if method in {"tasks/get", "tasks/update", "tasks/cancel"}:
                if not _client_tasks_enabled(req):
                    return _rpc_error(
                        rid,
                        TASK_CAPABILITY_MISSING,
                        "Missing required client capability",
                        {"requiredCapabilities": {"extensions": {TASKS_EXT: {}}}},
                    )
                params = req.get("params") or {}
                task_id = str(params.get("taskId") or "")
                try:
                    if method == "tasks/get":
                        MCP_TASKS.ensure_running(task_id, _dispatch_task_tool)
                        return _rpc_result(rid, MCP_TASKS.public(task_id))
                    if method == "tasks/cancel":
                        return _rpc_result(rid, MCP_TASKS.cancel(task_id))
                    return _rpc_result(rid, MCP_TASKS.update(task_id, params.get("inputResponses")))
                except KeyError:
                    return _rpc_error(rid, -32602, "Unknown taskId")
                except Exception as exc:
                    return _rpc_error(rid, -32603, str(exc))
            if method not in {"tools/list", "tools/call"}:
                return _rpc_error(rid, -32601, "Method not found", status=404)

        # New Airi tools are additive for both legacy and modern clients.
        if req.get("method") == "tools/call":
            params = req.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            if modern and name in TASK_CAPABLE_TOOLS and _client_tasks_enabled(req):
                try:
                    task = MCP_TASKS.create(name, args)
                    MCP_TASKS.ensure_running(task["taskId"], _dispatch_task_tool)
                    return _rpc_result(req.get("id"), task)
                except Exception as exc:
                    return _rpc_error(req.get("id"), -32000, str(exc))
            if name in NEW_TOOL_NAMES:
                try:
                    result = _call_new_tool(name, args)
                    return _rpc_result(
                        req.get("id"),
                        {"resultType": "complete", "content": [{"type": "text", "text": str(result)}], "structuredContent": result},
                        modern=modern,
                    )
                except Exception as exc:
                    return _rpc_error(req.get("id"), -32000, str(exc), modern=modern)

        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            return JSONResponse(content=None, status_code=response.status_code)

        if request.url.path == "/tools" and isinstance(data, dict):
            existing = list(data.get("tools") or [])
            # contract_server returns full tool descriptors; legacy server returns names.
            if existing and isinstance(existing[0], dict):
                known = {item.get("name") for item in existing if isinstance(item, dict)}
                existing.extend(item for item in NEW_TOOLS if item["name"] not in known)
            else:
                for name in sorted(NEW_TOOL_NAMES):
                    if name not in existing:
                        existing.append(name)
            data["tools"] = existing
            if "tool_count" in data:
                data["tool_count"] = len(existing)
        elif request.url.path == "/control-plane" and isinstance(data, dict):
            data["reflex"] = REFLEX.status()
            data["resource_discovery"] = {"staged": DISCOVERY.list_staged()["count"], "ard": "0.91"}
            data["judge"] = {"available": True, "mode": "isolated-sandbox-fail-closed"}
            data["mcp"] = {"legacy": LEGACY_PROTOCOL, "modern": PROTOCOL, "tasks_extension": True, "subscriptions": True}
        elif request.url.path == "/mcp" and isinstance(data, dict):
            if isinstance(data.get("result"), dict) and req.get("method") == "tools/list":
                listed = data["result"].setdefault("tools", [])
                known = {item.get("name") for item in listed if isinstance(item, dict)}
                listed.extend(item for item in NEW_TOOLS if item["name"] not in known)
                if modern:
                    data["result"].setdefault("ttlMs", 30_000)
                    data["result"].setdefault("cacheScope", "private")
            if modern:
                data = _stamp(data)

        headers = {key: value for key, value in response.headers.items() if key.lower() not in {"content-length", "content-type"}}
        if modern:
            headers["MCP-Protocol-Version"] = PROTOCOL
        return JSONResponse(data, status_code=response.status_code, headers=headers)
