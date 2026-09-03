from __future__ import annotations

"""Disabled compatibility shim.

The Airi-PC bootstrap intentionally contains no model provider. ChatGPT is the
sole reasoning authority. This module exists only so legacy imports fail clearly.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from typing import Any

HOST = os.environ.get("AIRI_MODEL_GATEWAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("AIRI_MODEL_GATEWAY_PORT", "17893"))

def gateway_token() -> str: return "disabled"

def gateway_config() -> dict[str, Any]:
    return {"provider": "disabled", "configured": False, "model": None, "chatgpt_only": True}

def authorized_header(_: str) -> bool: return False

def handle_chat(_: dict[str, Any], __: str) -> tuple[int, dict[str, Any]]:
    return 410, {"error": "model_gateway_disabled", "reason": "ChatGPT is the sole reasoning authority."}

def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body=json.dumps(payload,ensure_ascii=False).encode()
    handler.send_response(code); handler.send_header("Content-Type","application/json"); handler.send_header("Content-Length",str(len(body))); handler.end_headers(); handler.wfile.write(body)

class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self,*_args: Any)->None: return
    def do_GET(self)->None:
        if self.path == "/health": return _json_response(self,200,{"ok":True,**gateway_config()})
        if self.path == "/status": return _json_response(self,200,{**gateway_config(),"reachable":False})
        return _json_response(self,404,{"error":"not_found"})
    def do_POST(self)->None:
        if self.path != "/v1/chat/completions": return _json_response(self,404,{"error":"not_found"})
        try:
            length=int(self.headers.get("Content-Length","0")); json.loads(self.rfile.read(length))
        except (ValueError,json.JSONDecodeError): return _json_response(self,400,{"error":"invalid_request"})
        code,response=handle_chat({},self.headers.get("Authorization","")); return _json_response(self,code,response)

def serve()->None:
    raise RuntimeError("Airi-PC model gateway is disabled; ChatGPT is the sole reasoning authority")
