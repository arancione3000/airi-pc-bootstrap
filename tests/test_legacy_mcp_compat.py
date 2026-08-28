from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))
os.environ.setdefault("DISPLAY", ":99")

import contract_server

def test_legacy_mcp_non_control_plane_call_does_not_shadow_action():
    result = contract_server._legacy.mcp({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "computer_status", "arguments": {}},
    })
    assert "error" not in result
    assert result["result"]["structuredContent"]["ok"] is True
