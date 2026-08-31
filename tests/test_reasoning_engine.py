from pathlib import Path
import json
import sys

import pytest

COMPUTER = Path(__file__).resolve().parents[1] / "computer"
if str(COMPUTER) not in sys.path:
    sys.path.insert(0, str(COMPUTER))


def _engine(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRIPC_WORKSPACE_ROOT", str(tmp_path))
    import control_plane.store as store
    store.ROOT = Path(tmp_path).resolve()
    store.AI = store.ROOT / ".ai"
    store.CP = store.AI / "control_plane"
    from control_plane.reasoning_engine import ReasoningEngine
    return ReasoningEngine()


def test_goal_creation_and_planning(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("inspect project", [{"id": "a", "title": "inspect", "operation": "analyze"}])
    assert run["goal"] == "inspect project"
    assert run["phase"] == "EXECUTE"
    assert run["current_step"] == "a"


def test_dependency_resolution(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("dep", [
        {"id": "a", "dependencies": []},
        {"id": "b", "dependencies": ["a"]},
    ])
    assert engine.next_action(run["run_id"])["step"]["id"] == "a"
    with pytest.raises(ValueError):
        engine.mark_step("b", "RUNNING", run_id=run["run_id"])
    engine.mark_step("a", "RUNNING", run_id=run["run_id"])
    engine.mark_step("a", "COMPLETED", run_id=run["run_id"], result={"ok": True})
    assert engine.next_action(run["run_id"])["step"]["id"] == "b"


def test_observe_feedback_and_replan(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("repair", [{"id": "a", "max_attempts": 1}])
    rid = run["run_id"]
    engine.mark_step("a", "RUNNING", run_id=rid)
    engine.mark_step("a", "FAILED", run_id=rid, error="test timeout")
    engine.observe({"screenshot": "evidence/gui-1.png"}, run_id=rid, evidence={"path": "evidence/gui-1.png"})
    engine.feedback(operation="test", success=False, error="test timeout", tool="computer_test_run", run_id=rid)
    action = engine.replan(reason="test timeout", run_id=rid)
    assert action["action"] in {"execute", "replan", "verify"}
    assert engine.status(rid)["errors"]


def test_persistence(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("persist me", [{"id": "a"}])
    state_path = tmp_path / ".ai" / "control_plane" / "reasoning.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert run["run_id"] in data["runs"]
    from control_plane.reasoning_engine import ReasoningEngine
    restored = ReasoningEngine()
    assert restored.status(run["run_id"])["goal"] == "persist me"


def test_completed_step_protection(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("protect", [{"id": "a"}])
    rid = run["run_id"]
    engine.mark_step("a", "RUNNING", run_id=rid)
    engine.mark_step("a", "COMPLETED", run_id=rid)
    result = engine.mark_step("a", "FAILED", run_id=rid, error="late error")
    assert result["error"] == "completed_step_protected"


def test_invalid_input(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        engine.start("", [])
    with pytest.raises(ValueError):
        engine.start("x", [{"id": "a", "dependencies": ["missing"]}])


def test_retry_limit_and_recovery(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("recover", [{"id": "a", "max_attempts": 1}])
    rid = run["run_id"]
    engine.mark_step("a", "RUNNING", run_id=rid)
    engine.mark_step("a", "FAILED", run_id=rid, error="dependency missing")
    action = engine.replan(reason="dependency missing", run_id=rid, strategy="fallback")
    status = engine.status(rid)
    assert any(step["metadata"].get("fallback_for") == "a" for step in status["plan"])
    assert action["action"] in {"execute", "replan", "verify"}


def test_finish_requires_verification(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    run = engine.start("finish", [{"id": "a"}])
    rid = run["run_id"]
    with pytest.raises(ValueError):
        engine.finish(verified=True, run_id=rid)
    engine.mark_step("a", "RUNNING", run_id=rid)
    engine.mark_step("a", "COMPLETED", run_id=rid)
    done = engine.finish(verified=True, result={"ok": True}, run_id=rid)
    assert done["phase"] == "DONE"
    assert done["status"] == "COMPLETED"


def test_reasoning_goal_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRIPC_WORKSPACE_ROOT", str(tmp_path))
    import control_plane.store as store
    store.ROOT = Path(tmp_path).resolve()
    store.AI = store.ROOT / ".ai"
    store.CP = store.AI / "control_plane"
    from control_plane.orchestrator import ControlPlane
    cp = ControlPlane()
    plan = [{"id": "inspect", "title": "inspect workspace", "operation": "analyze", "args": {"path": "."}}]
    result = cp.reasoning_goal("inspect workspace", steps=plan, scope=[str(tmp_path)], max_time=30, max_iterations=5, max_retries=1, max_tool_calls=5)
    assert result["run_id"]
    assert result["phase"] in {"DONE", "FAILED"}
    if result["phase"] == "DONE":
        assert result["metadata"]["verified"] is True
        state_path = Path(tmp_path) / ".ai" / "control_plane" / "reasoning.json"
        assert state_path.exists()
