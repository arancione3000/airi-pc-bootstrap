from .orchestrator import ControlPlane
from .reasoning_engine import ReasoningEngine

__all__ = ["ControlPlane", "ReasoningEngine"]

# Airi-PC 2.1 additions are installed as a compatibility middleware.  This keeps
# the legacy 2025 MCP surface intact while adding the 2026 stateless era and the
# Reflex / ARD / Judge tools to the same FastAPI application.
try:
    import sys
    from .mcp2026 import install as _install_mcp2026
    _server = sys.modules.get("server") or sys.modules.get("computer.server")
    _app = getattr(_server, "app", None) if _server else None
    if _app is not None:
        _install_mcp2026(_app)
except Exception:
    # Never make the legacy runtime unimportable because an optional modern
    # compatibility layer failed to initialize. Runtime self-tests expose it.
    pass
