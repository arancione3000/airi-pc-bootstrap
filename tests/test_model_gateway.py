from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from control_plane import model_gateway


def test_gateway_rejects_missing_token(monkeypatch):
    monkeypatch.delenv("AIRI_MODEL_GATEWAY_TOKEN", raising=False)
    code, payload = model_gateway.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}]},
        "",
    )
    assert code == 401
    assert payload["error"] == "unauthorized"


def test_gateway_rejects_unqualified_provider(monkeypatch):
    monkeypatch.setenv("AIRI_MODEL_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(
        model_gateway.generalist_provider,
        "status",
        lambda: {"available": False, "reason": "not qualified", "qualified": False, "enabled": True, "model": None},
    )
    code, payload = model_gateway.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}]},
        "Bearer secret",
    )
    assert code == 503
    assert payload["error"] == "generalist_provider_unavailable"


def test_gateway_returns_openai_style_completion_for_qualified_provider(monkeypatch):
    monkeypatch.setenv("AIRI_MODEL_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(
        model_gateway.generalist_provider,
        "status",
        lambda: {
            "available": True,
            "reason": "qualified",
            "qualified": True,
            "enabled": True,
            "model": "local-causal-lm",
        },
    )
    monkeypatch.setattr(
        model_gateway.generalist_provider,
        "chat",
        lambda messages, max_new_tokens=256: "hello from airi",
    )
    code, payload = model_gateway.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        "Bearer secret",
    )
    assert code == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "local-causal-lm"
    assert payload["choices"][0]["message"]["content"] == "hello from airi"


def test_gateway_limits_requested_generation(monkeypatch):
    monkeypatch.setenv("AIRI_MODEL_GATEWAY_TOKEN", "secret")
    monkeypatch.setattr(
        model_gateway.generalist_provider,
        "status",
        lambda: {
            "available": True,
            "reason": "qualified",
            "qualified": True,
            "enabled": True,
            "model": "local-causal-lm",
        },
    )
    seen = {}
    def fake_chat(messages, max_new_tokens=256):
        seen["max_new_tokens"] = max_new_tokens
        return "ok"
    monkeypatch.setattr(model_gateway.generalist_provider, "chat", fake_chat)
    code, _ = model_gateway.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 999999},
        "Bearer secret",
    )
    assert code == 200
    assert seen["max_new_tokens"] == 2048
