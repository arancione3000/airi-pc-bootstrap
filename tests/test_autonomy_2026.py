from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DISPLAY", ":99")
sys.path.insert(0, str(ROOT / "computer"))

from control_plane import judge as judge_mod
from control_plane import mcp2026 as modern
from control_plane import mcp_tasks as task_mod
from control_plane import reflex as reflex_mod
from control_plane import resource_discovery as discovery_mod


def _fresh_reflex(monkeypatch):
    monkeypatch.setattr(reflex_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(reflex_mod, "save_json", lambda *_args, **_kwargs: None)
    return reflex_mod.ReflexEngine()


def test_reflex_is_idempotent_and_dispatches(monkeypatch):
    engine = _fresh_reflex(monkeypatch)
    engine.add_rule("github-comment", source="github", event_type="issue_comment", action={"type": "record_experience", "problem": "comment"})
    first = engine.emit("github", "issue_comment", {"x": 1}, "delivery-1")
    second = engine.emit("github", "issue_comment", {"x": 1}, "delivery-1")
    assert first["event"]["id"] == second["event"]["id"]
    assert second["duplicate"] is True
    seen = []
    out = engine.process(lambda action, event: seen.append((action, event["id"])) or {"ok": True})
    assert out["status"]["counts"].get("completed") == 1
    assert len(seen) == 1


def test_github_webhook_signature():
    secret = "unit-secret"
    body = b'{"action":"opened"}'
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert modern.verify_github_signature(secret, body, signature)
    assert not modern.verify_github_signature(secret, body + b"x", signature)


def test_ard_search_uses_spec_shape_and_separates_relevance_from_trust(monkeypatch):
    monkeypatch.setattr(discovery_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(discovery_mod, "save_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(discovery_mod.ResourceDiscovery, "_validate_remote", staticmethod(lambda _url: None))
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self, _n):
            return json.dumps({"results": [{"score": 97, "catalogEntry": {"identifier": "urn:air:publisher:skill:demo"}}]}).encode()

    def fake_open(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(discovery_mod, "_urlopen_no_redirect", fake_open)
    result = discovery_mod.ResourceDiscovery().search("browser helper", ["https://registry.example"], {"type": "skill"}, 5)
    assert captured["body"] == {
        "query": {"text": "browser helper", "filter": {"type": "skill"}},
        "federation": "auto",
        "pageSize": 5,
    }
    assert result["results"][0]["relevance_only"] is True
    assert "trust" in result["note"].lower()


def test_ard_trusted_markdown_skill_installs_without_executing_code(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery_mod, "ROOT", tmp_path)
    monkeypatch.setattr(discovery_mod, "AI", tmp_path / ".ai")
    monkeypatch.setattr(discovery_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(discovery_mod, "save_json", lambda *_args, **_kwargs: None)
    d = discovery_mod.ResourceDiscovery()
    candidate = {
        "displayName": "Demo Skill",
        "catalogEntry": {
            "identifier": "urn:air:trustedpub:skill:demo",
            "type": "text/markdown",
            "data": {"content": "# Demo Skill\n\n## Description\nSafe instructions only.\n"},
        },
    }
    result = d.acquire_skill(candidate, allowed_publishers=["trustedpub"])
    assert result["status"] == "installed_markdown_skill"
    installed = tmp_path / result["installed_path"]
    assert installed.read_text() == candidate["catalogEntry"]["data"]["content"]
    # A second identical acquisition must not overwrite with different content.
    result2 = d.acquire_skill(candidate, allowed_publishers=["trustedpub"])
    assert result2["status"] == "installed_markdown_skill"
    assert result2.get("note") == "identical skill already installed"


def test_skill_hunter_blocks_redirects():
    handler = discovery_mod._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://other.example") is None


def test_judge_runs_commands_in_isolated_copy(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("Airi-PC\n", encoding="utf-8")
    monkeypatch.setattr(judge_mod, "ROOT", tmp_path)
    judge = judge_mod.TrustedJudge(tmp_path)
    result = judge.evaluate(
        goal="verify safely",
        project_path=".",
        checks=[
            {"type": "file_contains", "path": "README.md", "text": "Airi-PC"},
            {"type": "command", "command": [sys.executable, "-c", "open('judge-sandbox-only.txt','w').write('x')"]},
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["sandboxed"] is True
    assert result["live_workspace_mutated_by_judge"] is False
    assert not (tmp_path / "judge-sandbox-only.txt").exists()


def test_judge_rejects_unapproved_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_mod, "ROOT", tmp_path)
    judge = judge_mod.TrustedJudge(tmp_path)
    result = judge.evaluate(goal="no shell", checks=[{"type": "command", "command": ["rm", "-rf", "."]}])
    assert result["verdict"] == "FAIL"
    assert "not allowed" in result["checks"][0]["error"]


def test_mcp_task_store_completes_and_publishes(monkeypatch):
    monkeypatch.setattr(task_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(task_mod, "save_json", lambda *_args, **_kwargs: None)
    store = task_mod.MCPTaskStore()
    events = []
    store.add_listener(events.append)
    task = store.create("computer_autonomous_goal", {"goal": "inspect"})
    store.ensure_running(task["taskId"], lambda name, args: {"name": name, "goal": args["goal"]})
    thread = store._threads[task["taskId"]]
    thread.join(timeout=2)
    final = store.public(task["taskId"])
    assert final["status"] == "completed"
    assert final["resultType"] == "complete"
    assert final["result"]["structuredContent"]["goal"] == "inspect"
    assert any(event["taskId"] == task["taskId"] for event in events)


def _meta(tasks=False):
    caps = {"extensions": {modern.TASKS_EXT: {}}} if tasks else {}
    return {modern.VERSION_KEY: modern.PROTOCOL, modern.CLIENT_CAPS_KEY: caps}


def test_mcp_2026_server_discover_and_required_routing_headers():
    app = FastAPI()

    @app.post("/mcp")
    async def legacy():
        return {"legacy": True}

    @app.get("/tools")
    async def tools():
        return {"tool_count": 1, "tools": [{"name": "legacy", "inputSchema": {"type": "object", "properties": {}}}]}

    modern.install(app)
    client = TestClient(app)
    req = {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": _meta()}}
    missing = client.post("/mcp", json=req, headers={"Mcp-Protocol-Version": modern.PROTOCOL})
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == modern.HEADER_MISMATCH
    ok = client.post(
        "/mcp", json=req,
        headers={"Mcp-Protocol-Version": modern.PROTOCOL, "Mcp-Method": "server/discover"},
    )
    result = ok.json()["result"]
    assert result["resultType"] == "complete"
    assert result["supportedVersions"] == [modern.PROTOCOL]
    assert result["_meta"][modern.SERVER_INFO_KEY]["name"] == "Airi Computer"


def test_mcp_2026_preserves_legacy_and_augments_tools():
    app = FastAPI()

    @app.post("/mcp")
    async def legacy():
        return {"jsonrpc": "2.0", "id": 1, "result": {"legacy": True}}

    @app.get("/tools")
    async def tools():
        return {"tool_count": 1, "tools": [{"name": "legacy", "inputSchema": {"type": "object", "properties": {}}}]}

    modern.install(app)
    client = TestClient(app)
    legacy = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert legacy.json()["result"]["legacy"] is True
    listing = client.get("/tools").json()
    names = [item["name"] for item in listing["tools"]]
    assert listing["tool_count"] == 1 + len(modern.NEW_TOOLS)
    assert modern.NEW_TOOL_NAMES.issubset(names)


def test_existing_auth_middleware_remains_outermost_and_blocks_modern_mcp():
    """Mirror server.py import order: MCP 2026 installs first, auth installs later."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/mcp")
    async def legacy():
        return {"jsonrpc": "2.0", "id": 1, "result": {"legacy": True}}

    @app.get("/tools")
    async def tools():
        return {"tool_count": 1, "tools": [{"name": "legacy", "inputSchema": {"type": "object", "properties": {}}}]}

    # This is the real ordering in computer/server.py: importing ControlPlane
    # installs the additive MCP 2026 layer, then server.py declares its auth
    # middleware afterwards. Starlette executes the later-added middleware first.
    modern.install(app)

    @app.middleware("http")
    async def existing_auth(request: Request, call_next):
        if request.url.path == "/mcp" and request.headers.get("authorization") != "Bearer allowed":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    client = TestClient(app)
    req = {
        "jsonrpc": "2.0",
        "id": 90,
        "method": "server/discover",
        "params": {"_meta": _meta(False)},
    }
    routing = {
        "Mcp-Protocol-Version": modern.PROTOCOL,
        "Mcp-Method": "server/discover",
    }
    denied = client.post("/mcp", json=req, headers=routing)
    assert denied.status_code == 401
    assert denied.json()["error"] == "unauthorized"

    allowed = client.post(
        "/mcp",
        json=req,
        headers={**routing, "Authorization": "Bearer allowed"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["result"]["supportedVersions"] == [modern.PROTOCOL]


def _modern_params(tasks: bool = True):
    extensions = {modern.TASKS_EXT: {}} if tasks else {}
    return {"_meta": {modern.VERSION_KEY: modern.PROTOCOL, modern.CLIENT_CAPS_KEY: {"extensions": extensions}}}


def _modern_headers(method: str, name: str | None = None):
    headers = {"MCP-Protocol-Version": modern.PROTOCOL, "Mcp-Method": method}
    if name is not None:
        headers["Mcp-Name"] = name
    return headers


def _contract_like_app():
    app = FastAPI()

    @app.get("/tools")
    def tools():
        return {
            "name": "legacy",
            "tool_count": 106,
            "base_tool_count": 83,
            "tools": [{"name": "old_tool", "inputSchema": {"type": "object", "properties": {}}}],
        }

    @app.get("/control-plane")
    def cp():
        return {"ok": True}

    @app.post("/mcp")
    def mcp(req: dict):
        if req.get("method") == "initialize":
            return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"protocolVersion": modern.LEGACY_PROTOCOL}}
        if req.get("method") == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "result": {"tools": [{"name": "old_tool", "inputSchema": {"type": "object", "properties": {}}}]},
            }
        return {"jsonrpc": "2.0", "id": req.get("id"), "result": {}}

    modern.install(app)
    return app


def test_modern_tools_list_is_augmented_and_cacheable():
    client = TestClient(_contract_like_app())
    response = client.post(
        "/mcp",
        headers=_modern_headers("tools/list"),
        json={"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": _modern_params()},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    names = {item["name"] for item in result["tools"]}
    assert "old_tool" in names
    assert modern.NEW_TOOL_NAMES.issubset(names)
    assert result["resultType"] == "complete"
    assert result["ttlMs"] > 0
    assert result["cacheScope"] == "private"


def test_modern_missing_meta_and_routing_mismatch_fail_closed():
    client = TestClient(_contract_like_app())
    missing = client.post(
        "/mcp",
        headers=_modern_headers("tools/list"),
        json={"jsonrpc": "2.0", "id": 12, "method": "tools/list", "params": {}},
    )
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == -32602

    mismatch = client.post(
        "/mcp",
        headers=_modern_headers("wrong"),
        json={"jsonrpc": "2.0", "id": 13, "method": "tools/list", "params": _modern_params()},
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["error"]["code"] == modern.HEADER_MISMATCH


def test_task_methods_do_not_require_mcp_name(monkeypatch):
    store = task_mod.MCPTaskStore()
    store.state = {"version": 1, "tasks": {}}
    monkeypatch.setattr(modern, "MCP_TASKS", store)
    task = store.create("computer_autonomous_goal", {"goal": "demo"})
    client = TestClient(_contract_like_app())
    request = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tasks/get",
        "params": {"taskId": task["taskId"], **_modern_params(True)},
    }
    response = client.post("/mcp", headers=_modern_headers("tasks/get"), json=request)
    assert response.status_code == 200


def test_modern_protocol_header_is_required():
    client = TestClient(_contract_like_app())
    request = {"jsonrpc": "2.0", "id": 15, "method": "tools/list", "params": _modern_params(False)}
    response = client.post("/mcp", headers={"Mcp-Method": "tools/list"}, json=request)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == modern.HEADER_MISMATCH


def test_mcp_name_base64_sentinel_is_decoded(monkeypatch):
    import base64

    monkeypatch.setattr(modern, "_call_new_tool", lambda _name, _args: {"ok": True})
    client = TestClient(_contract_like_app())
    tool_name = "computer_judge"
    encoded = "=?base64?" + base64.b64encode(tool_name.encode()).decode() + "?="
    request = {
        "jsonrpc": "2.0",
        "id": 16,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": {"goal": "x", "checks": []}, **_modern_params(False)},
    }
    response = client.post("/mcp", headers=_modern_headers("tools/call", encoded), json=request)
    assert response.status_code == 200


def test_unknown_modern_method_returns_http_404():
    client = TestClient(_contract_like_app())
    request = {"jsonrpc": "2.0", "id": 17, "method": "ping", "params": _modern_params(False)}
    response = client.post("/mcp", headers=_modern_headers("ping"), json=request)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == -32601


def test_cross_site_origin_is_rejected():
    client = TestClient(_contract_like_app())
    request = {"jsonrpc": "2.0", "id": 18, "method": "tools/list", "params": _modern_params(False)}
    response = client.post(
        "/mcp",
        headers={**_modern_headers("tools/list"), "Origin": "https://evil.example"},
        json=request,
    )
    assert response.status_code == 403


def test_signed_github_webhook_autoprocesses(monkeypatch):
    fresh = _fresh_reflex(monkeypatch)
    monkeypatch.setattr(modern, "REFLEX", fresh)
    monkeypatch.setattr(modern, "_dispatch_reflex", lambda _action, event: {"ok": True, "event": event["id"]})
    fresh.add_rule("github", source="github", event_type="issue_comment", action={"type": "record_experience"})
    monkeypatch.setenv("AIRI_GITHUB_WEBHOOK_SECRET", "ci-secret")
    client = TestClient(_contract_like_app())
    payload = json.dumps({"action": "created"}).encode()
    signature = "sha256=" + hmac.new(b"ci-secret", payload, hashlib.sha256).hexdigest()
    response = client.post(
        "/events/github",
        content=payload,
        headers={
            "content-type": "application/json",
            "x-hub-signature-256": signature,
            "x-github-event": "issue_comment",
            "x-github-delivery": "delivery-1",
        },
    )
    assert response.status_code == 200
    assert response.json()["duplicate"] is False
    for _ in range(100):
        if fresh.status()["counts"].get("completed") == 1:
            break
        import time
        time.sleep(0.01)
    assert fresh.status()["counts"].get("completed") == 1


def test_unknown_modern_protocol_never_falls_back_to_legacy():
    client = TestClient(_contract_like_app())
    request = {
        "jsonrpc": "2.0",
        "id": 19,
        "method": "tools/list",
        "params": {"_meta": {modern.VERSION_KEY: "2099-01-01", modern.CLIENT_CAPS_KEY: {}}},
    }
    response = client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": "2099-01-01", "Mcp-Method": "tools/list"},
        json=request,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == modern.UNSUPPORTED_PROTOCOL
    assert body["error"]["data"]["requested"] == "2099-01-01"


def test_reflex_retry_does_not_repeat_successful_rules(monkeypatch):
    engine = _fresh_reflex(monkeypatch)
    engine.add_rule("ok", source="x", event_type="y", action={"type": "record_experience", "solution": "ok"})
    engine.add_rule(
        "flaky",
        source="x",
        event_type="y",
        action={"type": "record_experience", "solution": "flaky"},
        max_retries=2,
    )
    engine.emit("x", "y", {}, "retry-once")
    calls = {"ok": 0, "flaky": 0}

    def dispatch(action, _event):
        key = action.get("solution")
        calls[key] += 1
        if key == "flaky" and calls[key] == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    first = engine.process(dispatch)
    assert first["status"]["counts"].get("retrying") == 1
    second_calls = []
    second = engine.process(lambda _action, event: second_calls.append(event["id"]) or {"ok": True})
    assert second["status"]["counts"].get("completed") == 1
    assert len(second_calls) == 1


def test_ard_rejects_loopback_registry(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="non-public"):
        discovery_mod.ResourceDiscovery._validate_remote("https://example.test")


def test_control_plane_import_auto_installs_mcp2026(tmp_path):
    import subprocess
    code = f'''\nimport sys, types\nfrom pathlib import Path\nfrom fastapi import FastAPI\nsys.path.insert(0, {str(ROOT / "computer")!r})\nfake=types.ModuleType("server")\nfake.app=FastAPI()\nsys.modules["server"]=fake\nimport control_plane\nassert getattr(fake.app.state,"airi_mcp2026_installed",False) is True\n'''
    env = dict(os.environ)
    env["AIRIPC_WORKSPACE_ROOT"] = str(tmp_path / "workspace")
    completed = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
