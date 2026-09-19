from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution import lab, lab_runtime, lab_worker


def _patch_lab(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "airi"
    state = workspace / ".ai" / "evolution-lab" / "shadow-router"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(lab, "ROOT", workspace)
    monkeypatch.setattr(lab, "LAB_STATE", state)
    monkeypatch.setattr(lab, "RAW", state / "raw" / "observations.jsonl")
    monkeypatch.setattr(lab, "DATA", state / "data" / "verified.jsonl")
    monkeypatch.setattr(lab, "META", state / "lab-meta.json")
    monkeypatch.setattr(lab_runtime, "STATE", state)
    monkeypatch.setattr(lab_runtime, "PID", state / "worker.pid")
    monkeypatch.setattr(lab_runtime, "LOG", state / "worker.log")
    monkeypatch.setattr(lab_runtime, "AUTOPILOT_DISABLED", state / "autopilot.disabled")
    return workspace, state


def test_shadow_trace_sanitizes_values_and_keeps_only_structural_shape(monkeypatch, tmp_path: Path):
    _patch_lab(monkeypatch, tmp_path)
    feature = lab.route_feature(
        "Open https://secret.example account=user@example.com password=hunter2 at /home/user/private/file.txt 123456",
        "browser_open",
        "computer_browser_open",
        {"url": "https://secret.example/private", "token": "SUPERSECRET", "timeout": 30},
    )
    assert "hunter2" not in feature
    assert "SUPERSECRET" not in feature
    assert "secret.example" not in feature
    assert "user@example.com" not in feature
    assert "/home/user/private" not in feature
    assert "123456" not in feature
    assert "url:str" not in feature
    assert "token:str" not in feature
    assert "timeout:int" not in feature
    assert "arg_shape count:3;int:1,str:2" in feature
    assert "url:1" in feature and "email:1" in feature and "path:1" in feature


def test_shadow_observations_rebuild_into_isolated_training_dataset(monkeypatch, tmp_path: Path):
    _, state = _patch_lab(monkeypatch, tmp_path)
    for i in range(6):
        lab.record_execution(
            goal=f"Read project documentation section {i}",
            operation="read",
            tool="computer_file_read",
            args={"path": f"/private/path/{i}.txt"},
            success=True,
            latency_ms=20,
            task_id=f"t{i}",
            node_id="n1",
            candidates=["computer_file_read", "computer_browser_text"],
        )
    for i in range(6):
        lab.record_execution(
            goal=f"Open unavailable web resource {i}",
            operation="browser_open",
            tool="computer_browser_open",
            args={"url": f"https://private.example/{i}"},
            success=False,
            latency_ms=700,
            error="network connection failed",
            task_id=f"f{i}",
            node_id="n1",
            candidates=["computer_browser_open"],
        )
    result = lab.rebuild_dataset()
    assert result["observations"] == 12
    assert result["success"] == 6
    assert result["failure"] == 6
    raw = (state / "raw" / "observations.jsonl").read_text(encoding="utf-8")
    assert "private.example" not in raw
    assert "/private/path/" not in raw
    rows = [json.loads(line) for line in (state / "data" / "verified.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert {row["label"] for row in rows} == {0, 1}


def test_lab_write_boundary_rejects_state_outside_workspace(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "airi"
    outside = tmp_path / "outside"
    workspace.mkdir()
    monkeypatch.setattr(lab, "ROOT", workspace)
    monkeypatch.setattr(lab, "LAB_STATE", outside)
    monkeypatch.setattr(lab, "RAW", outside / "raw.jsonl")
    with pytest.raises(RuntimeError, match="escaped Airi workspace"):
        lab.record_execution(
            goal="safe test",
            operation="read",
            tool="computer_file_read",
            args={},
            success=True,
        )


def test_lab_runtime_audit_detects_symlink_escape(monkeypatch, tmp_path: Path):
    _, state = _patch_lab(monkeypatch, tmp_path)
    state.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (state / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    result = lab_runtime.audit()
    assert result["ok"] is False
    assert any("symlink escapes" in message for message in result["errors"])


def test_worker_policy_blocks_network_subprocess_and_external_writes(tmp_path: Path):
    root = tmp_path / "lab"
    root.mkdir()
    assert lab_worker.sandbox_violation("socket.connect", (), root)
    assert lab_worker.sandbox_violation("socket.getaddrinfo", (), root)
    assert lab_worker.sandbox_violation("subprocess.Popen", (), root)
    assert lab_worker.sandbox_violation("open", (tmp_path / "outside.txt", "w", 0), root)
    assert lab_worker.sandbox_violation("os.remove", (tmp_path / "outside.txt",), root)
    assert lab_worker.sandbox_violation("open", (root / "inside.txt", "w", 0), root) is None
    assert lab_worker.sandbox_violation("open", (tmp_path / "outside.txt", "r", 0), root) is None


def test_worker_environment_does_not_inherit_credentials(monkeypatch, tmp_path: Path):
    _patch_lab(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    env = lab_runtime._safe_env()
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == "2"


def test_shadow_scoring_is_advisory_and_never_selects_tool(monkeypatch, tmp_path: Path):
    _, state = _patch_lab(monkeypatch, tmp_path)
    champion = state / "champion"
    champion.mkdir(parents=True)
    (champion / "model.pt").write_bytes(b"stub")
    (state / "lab-meta.json").write_text(
        json.dumps({"feature_schema": lab.FEATURE_SCHEMA}),
        encoding="utf-8",
    )

    def fake_predict(_state, text, vocab_size=8192):
        p = 0.9 if "computer_file_read" in text else 0.2
        return {
            "real_probability": p,
            "fake_probability": 1.0 - p,
            "confidence": max(p, 1.0 - p),
            "label": "likely_real" if p >= 0.5 else "likely_fake",
        }

    monkeypatch.setattr(lab, "predict_text", fake_predict)
    result = lab.score_candidates(
        "Read the project file",
        "read",
        ["computer_browser_text", "computer_file_read"],
        {"path": "/secret/project.txt"},
    )
    assert result["shadow_only"] is True
    assert result["candidates"][0]["tool"] == "computer_file_read"
    assert "selected" not in result
    assert "never used to execute" in result["warning"]


def test_maintenance_starts_only_after_new_observation_threshold(monkeypatch):
    monkeypatch.setattr(lab_runtime, "status", lambda: {
        "worker_running": False,
        "pending_observations": lab_runtime.DEFAULT_TRIGGER - 1,
    })
    called = []
    monkeypatch.setattr(lab_runtime, "start", lambda **kwargs: called.append(True) or {"ok": True})
    result = lab_runtime.maintenance()
    assert result["training_started"] is None
    assert called == []

    monkeypatch.setattr(lab_runtime, "status", lambda: {
        "worker_running": False,
        "pending_observations": lab_runtime.DEFAULT_TRIGGER,
    })
    result = lab_runtime.maintenance()
    assert result["training_started"]["ok"] is True
    assert called == [True]


def test_control_plane_records_shadow_observations_without_using_them_for_route():
    text = (ROOT / "computer" / "control_plane" / "orchestrator.py").read_text(encoding="utf-8")
    assert "_lab_record_execution" in text
    execute_start = text.index("    def execute(")
    execute_end = text.index("    def _classify_failure", execute_start)
    execute = text[execute_start:execute_end]
    assert "route=self.route(candidates)" in execute
    assert execute.index("route=self.route(candidates)") < execute.index("_lab_record_execution")


def test_server_exposes_shadow_lab_observation_tools():
    text = (ROOT / "computer" / "server.py").read_text(encoding="utf-8")
    for name in (
        "computer_evolution_lab_status",
        "computer_evolution_lab_audit",
        "computer_evolution_lab_score",
        "computer_evolution_lab_start",
        "computer_evolution_lab_stop",
        "computer_evolution_lab_maintenance",
        "computer_evolution_lab_autopilot",
    ):
        assert name in text


def test_scheduler_has_default_persistent_shadow_lab_job():
    text = (ROOT / "computer" / "advanced.py").read_text(encoding="utf-8")
    assert "evolution-lab-shadow-router" in text
    assert "evolution_lab_maintenance" in text
    assert "autopilot.disabled" in text


def test_insufficient_cycle_marks_observations_seen_and_waits_for_new_data(monkeypatch, tmp_path: Path):
    _, state = _patch_lab(monkeypatch, tmp_path)
    for i in range(10):
        lab.record_execution(
            goal=f"Repeated route sample {i}",
            operation="read",
            tool="computer_file_read",
            args={"path": f"/private/{i}.txt"},
            success=True,
        )
    result = lab.run_cycle(population=4, generations=1, candidate_epochs=1, finalist_epochs=1)
    assert result["trained"] is False
    meta = json.loads((state / "lab-meta.json").read_text(encoding="utf-8"))
    assert meta["last_cycle_raw_count"] == 10
    st = lab.status()
    assert st["pending_observations"] == 0
