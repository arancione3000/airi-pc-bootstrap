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
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
AUTH_FILE = Path(os.environ.get("AIRI_MODEL_GATEWAY_AUTH_FILE", "/tmp/airi-model-gateway.token"))
TIMEOUT = float(os.environ.get("AIRI_MODEL_GATEWAY_TIMEOUT", "90"))


def _token() -> str:
    if AUTH_FILE.exists():
        return AUTH_FILE.read_text().strip()
    value = secrets.token_urlsafe(32)
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(value + "\n")
    os.chmod(AUTH_FILE, 0o600)
    return value

GATEWAY_TOKEN = _token()


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    raw = handler.headers.get("Authorization", "")
    return raw.startswith("Bearer ") and hmac.compare_digest(raw[7:], GATEWAY_TOKEN)


def _provider_request(payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            return _json_response(self, 200, {"ok": True, "provider": "openrouter", "configured": bool(API_KEY), "model": MODEL})
        if self.path == "/status":
            return _json_response(self, 200, {"provider": "openrouter", "configured": bool(API_KEY), "model": MODEL, "reachable": bool(API_KEY)})
        return _json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            return _json_response(self, 404, {"error": "not_found"})
        if not _authorized(self):
            return _json_response(self, 401, {"error": "unauthorized"})
        if not API_KEY:
            return _json_response(self, 503, {"error": "provider_unavailable"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                raise ValueError("invalid_request")
            body.setdefault("model", MODEL)
            return _json_response(self, 200, _provider_request(body))
        except ValueError:
            return _json_response(self, 400, {"error": "invalid_request"})
        except RuntimeError as exc:
            return _json_response(self, 504 if str(exc) == "provider_timeout" else 502, {"error": str(exc)})


def serve() -> None:
    server = ThreadingHTTPServer((HOST, PORT), GatewayHandler)
    server.serve_forever()
