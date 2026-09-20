from __future__ import annotations

"""OpenAI-compatible local gateway for a qualified AIRI Generalist checkpoint."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from typing import Any

from . import generalist_provider

HOST = os.environ.get("AIRI_MODEL_GATEWAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("AIRI_MODEL_GATEWAY_PORT", "17893"))


def gateway_token() -> str:
    return os.environ.get("AIRI_MODEL_GATEWAY_TOKEN", "").strip()


def gateway_config() -> dict[str, Any]:
    status = generalist_provider.status()
    return {
        "provider": "airi-generalist",
        "configured": bool(status.get("available")),
        "model": status.get("model"),
        "chatgpt_only": not bool(status.get("available")),
        "qualified": bool(status.get("qualified")),
        "enabled": bool(status.get("enabled")),
    }


def authorized_header(header: str) -> bool:
    token = gateway_token()
    if not token:
        return False
    return str(header or "") == f"Bearer {token}"


def handle_chat(payload: dict[str, Any], authorization: str) -> tuple[int, dict[str, Any]]:
    if not authorized_header(authorization):
        return 401, {"error": "unauthorized"}
    status = generalist_provider.status()
    if not status.get("available"):
        return 503, {
            "error": "generalist_provider_unavailable",
            "reason": status.get("reason"),
        }
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return 400, {"error": "messages_required"}
    max_tokens = payload.get("max_tokens", 256)
    try:
        max_tokens = max(1, min(int(max_tokens), 2048))
        text = generalist_provider.chat(messages, max_new_tokens=max_tokens)
    except (ValueError, RuntimeError) as exc:
        return 400, {"error": "generation_failed", "reason": str(exc)}
    model = status.get("model") or "local-causal-lm"
    return 200, {
        "id": "airi-generalist-local",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            return _json_response(self, 200, {"ok": True, **gateway_config()})
        if self.path == "/status":
            config = gateway_config()
            return _json_response(self, 200, {**config, "reachable": True})
        return _json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            return _json_response(self, 404, {"error": "not_found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            return _json_response(self, 400, {"error": "invalid_request"})
        code, response = handle_chat(payload, self.headers.get("Authorization", ""))
        return _json_response(self, code, response)


def serve() -> None:
    server = ThreadingHTTPServer((HOST, PORT), GatewayHandler)
    server.serve_forever()


if __name__ == "__main__":
    serve()
