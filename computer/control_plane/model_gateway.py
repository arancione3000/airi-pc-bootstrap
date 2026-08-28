from __future__ import annotations

import hmac
import json
import os
import secrets
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = int(os.environ.get("AIRI_MODEL_GATEWAY_PORT", "17893"))
MODEL = os.environ.get("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-fin:free")
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
AUTH_FILE = Path(os.environ.get("AIRI_MODEL_GATEWAY_AUTH_FILE", "/tmp/airi-model-gateway.token"))
TIMEOUT = float(os.environ.get("AIRI_MODEL_GATEWAY_TIMEOUT", "90"))


def _api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "")


def _token() -> str:
    configured = os.environ.get("AIRI_MODEL_GATEWAY_TOKEN", "").strip()
    if configured:
        return configured
    if AUTH_FILE.exists():
        return AUTH_FILE.read_text().strip()
    value = secrets.token_urlsafe(32)
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(value + "\n")
    os.chmod(AUTH_FILE, 0o600)
    return value


def gateway_token() -> str:
    return _token()


def gateway_config() -> dict[str, Any]:
    return {"provider": "openrouter", "configured": bool(_api_key()), "model": MODEL}


def _authorized_value(value: str) -> bool:
    token = _token()
    return bool(token) and hmac.compare_digest(value, token)


def authorized_header(raw: str) -> bool:
    return raw.startswith("Bearer ") and _authorized_value(raw[7:])


def _provider_request(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("provider_unavailable")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://127.0.0.1"),
        "X-Title": os.environ.get("OPENROUTER_X_TITLE", "Airi-PC"),
    }
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError("provider_auth_error") from exc
        if exc.code == 429:
            raise RuntimeError("provider_rate_limited") from exc
        raise RuntimeError("provider_http_error") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("provider_unavailable") from exc
    except TimeoutError as exc:
        raise RuntimeError("provider_timeout") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("malformed_provider_response") from exc
    if not isinstance(data, dict):
        raise RuntimeError("malformed_provider_response")
    return data


def handle_chat(payload: dict[str, Any], authorization: str) -> tuple[int, dict[str, Any]]:
    if not authorized_header(authorization):
        return 401, {"error": "unauthorized"}
    if not _api_key():
        return 503, {"error": "provider_unavailable"}
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return 400, {"error": "invalid_request"}
    payload = dict(payload)
    payload.setdefault("model", MODEL)
    try:
        return 200, _provider_request(payload)
    except RuntimeError as exc:
        name = str(exc)
        code = 504 if name == "provider_timeout" else (429 if name == "provider_rate_limited" else 502)
        return code, {"error": name}


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
            return _json_response(self, 200, {**gateway_config(), "reachable": gateway_config()["configured"]})
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
    ThreadingHTTPServer((HOST, PORT), GatewayHandler).serve_forever()
