from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from computer.control_plane.model_gateway import gateway_config, handle_chat

app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True, **gateway_config()}


@app.get("/status")
async def status():
    config = gateway_config()
    return {**config, "reachable": config["configured"]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    code, response = handle_chat(payload, request.headers.get("authorization", ""))
    return JSONResponse(response, status_code=code)
