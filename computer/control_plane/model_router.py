from __future__ import annotations

import os

from . import generalist_provider
from .store import load_json, save_json

FILE = "model-routing.json"
DEFAULTS = {
    "simple": "chatgpt",
    "coding": "chatgpt",
    "data": "chatgpt",
    "research": "chatgpt",
    "vision": "chatgpt",
    "review": "chatgpt",
}
CHATGPT_PROVIDER = {
    "name": "chatgpt",
    "capabilities": ["simple", "coding", "data", "research", "vision", "review"],
    "available": True,
    "cost_class": "reasoning-authority",
}
GENERALIST_CAPABILITIES = ["simple", "coding", "data", "review"]


def _prefer_generalist() -> bool:
    return os.environ.get("AIRI_GENERALIST_PREFER", "0").strip().lower() in {"1", "true", "yes", "on"}


class ModelRouter:
    """Route to a qualified local generalist model only when explicitly enabled."""

    def __init__(self):
        self.state = load_json(FILE, {})
        local = generalist_provider.status()
        providers = {"chatgpt": dict(CHATGPT_PROVIDER)}
        if local.get("available"):
            providers["airi-generalist"] = {
                "name": "airi-generalist",
                "capabilities": list(GENERALIST_CAPABILITIES),
                "available": True,
                "cost_class": "local-qualified",
                "qualification": local.get("qualification"),
            }
        self.state["version"] = 3
        self.state["providers"] = providers
        self.state["routes"] = DEFAULTS.copy()
        self.state["routing_authority"] = (
            "airi-generalist-eligible"
            if "airi-generalist" in providers
            else "chatgpt"
        )
        save_json(FILE, self.state)

    def register_provider(self, name, capabilities=None, available=False, cost_class="unknown"):
        del available, cost_class
        if name == "chatgpt":
            self.state["providers"]["chatgpt"] = dict(CHATGPT_PROVIDER)
            save_json(FILE, self.state)
            return self.state["providers"]["chatgpt"]
        if name == "airi-generalist":
            local = generalist_provider.status()
            if local.get("available"):
                row = {
                    "name": "airi-generalist",
                    "capabilities": list(capabilities or GENERALIST_CAPABILITIES),
                    "available": True,
                    "cost_class": "local-qualified",
                    "qualification": local.get("qualification"),
                }
                self.state["providers"]["airi-generalist"] = row
                save_json(FILE, self.state)
                return row
        return {
            "name": str(name),
            "capabilities": list(capabilities or []),
            "available": False,
            "cost_class": "disabled",
            "disabled": True,
            "reason": "provider is not a qualified allowlisted reasoning backend",
        }

    def choose(self, task_type="simple", complexity="medium", needs_vision=False, prefer_speed=False):
        del complexity, prefer_speed
        kind = "vision" if needs_vision else str(task_type)
        candidates = ["chatgpt"]
        local = self.state["providers"].get("airi-generalist")
        if local and kind in local.get("capabilities", []):
            candidates.append("airi-generalist")
        selected = (
            "airi-generalist"
            if _prefer_generalist() and "airi-generalist" in candidates
            else "chatgpt"
        )
        return {
            "route": kind,
            "selected": selected,
            "candidates": candidates,
            "reasoning_authority": selected,
        }

    def status(self):
        return self.state
