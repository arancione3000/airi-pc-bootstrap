from __future__ import annotations

import pytest

from control_plane import reflex as reflex_mod


def _fresh_reflex(monkeypatch):
    monkeypatch.setattr(reflex_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(reflex_mod, "save_json", lambda *_args, **_kwargs: None)
    return reflex_mod.ReflexEngine()


def test_reflex_respects_per_rule_retry_limits(monkeypatch):
    engine = _fresh_reflex(monkeypatch)
    engine.add_rule("never-retry", source="x", event_type="y", action={"type": "record_experience", "solution": "zero"}, max_retries=0)
    engine.add_rule("retry-three", source="x", event_type="y", action={"type": "record_experience", "solution": "three"}, max_retries=3)
    event_id = engine.emit("x", "y", {}, "mixed-retries")["event"]["id"]
    calls = {"zero": 0, "three": 0}

    def dispatch(action, _event):
        calls[action["solution"]] += 1
        raise RuntimeError("fail")

    for _ in range(4):
        engine.process(dispatch)
    event = engine.state["events"][event_id]
    assert event["status"] == "dead_letter"
    assert calls == {"zero": 1, "three": 4}
    assert "never-retry" in event["exhausted_rules"]


def test_project_index_rejects_paths_outside_workspace(monkeypatch, tmp_path):
    from control_plane import project_index as project_index_mod
    import coding

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    monkeypatch.setattr(project_index_mod, "ROOT", workspace)
    monkeypatch.setattr(project_index_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(project_index_mod, "save_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(coding, "ROOT", workspace)
    with pytest.raises(ValueError, match="outside Airi-PC workspace"):
        project_index_mod.ProjectIndex().refresh(["../outside.txt"])


def test_transaction_engine_rejects_closed_mutation_and_restores_file_type(monkeypatch, tmp_path):
    from control_plane import transaction_engine as tx_mod
    import coding

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(tx_mod, "ROOT", workspace)
    monkeypatch.setattr(coding, "ROOT", workspace)
    monkeypatch.setattr(tx_mod, "load_json", lambda _name, default: default)
    monkeypatch.setattr(tx_mod, "save_json", lambda *_args, **_kwargs: None)
    target = workspace / "data.txt"
    target.write_text("before")
    engine = tx_mod.TransactionEngine()
    tx = engine.begin(["data.txt"], "regression")
    target.unlink(); target.mkdir(); (target / "nested").write_text("after")
    rolled = engine.rollback(tx["id"])
    assert rolled["status"] == "rolled_back"
    assert target.is_file() and target.read_text() == "before"
    with pytest.raises(ValueError, match="cannot add step"):
        engine.step(tx["id"], "late")
    with pytest.raises(ValueError, match="cannot commit"):
        engine.commit(tx["id"])


def test_maintenance_recovery_validates_level_and_healthy_auto_is_noop(monkeypatch):
    from control_plane import maintenance as maintenance_mod

    manager = maintenance_mod.MaintenanceManager()
    healthy = {"overall_ok": True, "checks": {}, "timestamp": 0}
    monkeypatch.setattr(manager, "run", lambda: healthy)
    assert manager.recover("auto")["noop"] is True
    invalid = manager.recover(6)
    assert invalid == {"ok": False, "error": "level must be auto or 1..5"}
