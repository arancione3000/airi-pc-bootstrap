from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution.cli import parser


def test_cli_exposes_autonomous_pipeline_commands():
    p = parser()
    for argv in (
        ["bootstrap-liar"],
        ["factcheck", "a sufficiently long claim"],
        ["queue-add", "a sufficiently long claim"],
        ["report"],
        ["audit"],
        ["drift"],
        ["edge-quantize"],
        ["predict-edge", "a sufficiently long claim"],
        ["maintenance"],
        ["autopilot"],
        ["export"],
    ):
        args = p.parse_args(argv)
        assert args.cmd == argv[0]


def test_server_exposes_native_evolution_mcp_tools():
    text = (ROOT / "computer" / "server.py").read_text(encoding="utf-8")
    expected = [
        "computer_evolution_status",
        "computer_evolution_audit",
        "computer_evolution_bootstrap_liar",
        "computer_evolution_factcheck",
        "computer_evolution_predict",
        "computer_evolution_predict_edge",
        "computer_evolution_drift",
        "computer_evolution_edge_quantize",
        "computer_evolution_report",
        "computer_evolution_export",
        "computer_evolution_queue_add",
        "computer_evolution_queue_list",
        "computer_evolution_queue_verify",
        "computer_evolution_start",
        "computer_evolution_stop",
        "computer_evolution_maintenance",
        "computer_evolution_autopilot",
    ]
    for name in expected:
        assert name in text
    assert "from evolution import runtime as evolution_runtime" in text


def test_scheduler_allows_only_bounded_evolution_maintenance():
    text = (ROOT / "computer" / "advanced.py").read_text(encoding="utf-8")
    assert "'evolution_maintenance'" in text
    assert "evolution_runtime.maintenance(mode='safe')" in text
