from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.model import GeneralistLMConfig
from generalist_lm.qualification import QUALIFICATION_VERSION, checkpoint_digest, qualification_status
from generalist_lm.runtime import GeneralistRuntime


def make_qualified_checkpoint(path: Path) -> Path:
    pytest.importorskip("torch")
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()
    runtime = GeneralistRuntime.fresh(cfg)
    runtime.save_checkpoint(path, metadata={"test_checkpoint": True})
    digest = checkpoint_digest(path)
    (path / "benchmark.json").write_text(json.dumps({
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-qualification-v2",
        "checkpoint_digest": digest,
        "qualified": True,
        "minimum_score": 85.0,
        "report": {
            "ok": True,
            "score": 100.0,
            "critical_failures": [],
            "domain_scores": {
                "language": 1.0,
                "coding": 1.0,
                "data": 1.0,
                "reasoning": 1.0,
                "tools": 1.0,
                "structured": 1.0,
            },
        },
    }, indent=2), encoding="utf-8")
    return path


def test_qualification_is_bound_to_exact_checkpoint(tmp_path: Path):
    make_qualified_checkpoint(tmp_path)
    before = qualification_status(tmp_path)
    assert before["qualified"] is True
    assert before["integrity_ok"] is True

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["dropout"] = 0.125
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    after = qualification_status(tmp_path)
    assert after["qualified"] is False
    assert after["integrity_ok"] is False


def test_generalist_provider_requires_flag_and_qualification(tmp_path: Path, monkeypatch):
    from control_plane import generalist_provider

    make_qualified_checkpoint(tmp_path)
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path))
    monkeypatch.delenv("AIRI_GENERALIST_ENABLE", raising=False)
    assert generalist_provider.status()["available"] is False

    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")
    row = generalist_provider.status()
    assert row["available"] is True
    assert row["qualified"] is True
    assert row["checkpoint_complete"] is True


def test_model_router_prefers_generalist_only_when_explicitly_requested(tmp_path: Path, monkeypatch):
    import control_plane.store as store
    from control_plane.model_router import ModelRouter

    make_qualified_checkpoint(tmp_path / "model")
    monkeypatch.setattr(store, "CP", tmp_path / "control-plane")
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path / "model"))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")

    router = ModelRouter()
    assert "airi-generalist" in router.status()["providers"]
    assert router.choose(task_type="coding")["selected"] == "chatgpt"

    monkeypatch.setenv("AIRI_GENERALIST_PREFER", "1")
    assert router.choose(task_type="coding")["selected"] == "airi-generalist"
    assert router.choose(task_type="simple")["selected"] == "airi-generalist"
    assert router.choose(task_type="vision", needs_vision=True)["selected"] == "chatgpt"


def test_unqualified_or_unknown_provider_cannot_be_registered(tmp_path: Path, monkeypatch):
    import control_plane.store as store
    from control_plane.model_router import ModelRouter

    monkeypatch.setattr(store, "CP", tmp_path / "control-plane")
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path / "missing"))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")
    router = ModelRouter()
    assert set(router.status()["providers"]) == {"chatgpt"}
    denied = router.register_provider("airi-generalist", ["coding"], available=True)
    assert denied["available"] is False
    unknown = router.register_provider("arbitrary-remote-model", ["coding"], available=True)
    assert unknown["available"] is False


def test_local_agent_invocation_uses_only_qualified_provider(tmp_path: Path, monkeypatch):
    from control_plane import generalist_provider, local_agent

    make_qualified_checkpoint(tmp_path)
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")
    monkeypatch.setattr(generalist_provider, "chat", lambda messages, max_new_tokens=256: "LOCAL_OK")

    assert local_agent.ask_local_model("say ok") == "LOCAL_OK"
    result = local_agent._provider_request(
        [{"role": "user", "content": "say ok"}],
        "airi-generalist",
    )
    assert result["provider"] == "airi-generalist"
    assert result["content"] == "LOCAL_OK"

    with pytest.raises(RuntimeError):
        local_agent._provider_request(
            [{"role": "user", "content": "say ok"}],
            "unapproved-provider",
        )


def test_model_change_parser_rejects_escape_and_sensitive_paths(tmp_path: Path, monkeypatch):
    from control_plane import generalist_provider, local_agent

    make_qualified_checkpoint(tmp_path / "model")
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path / "model"))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")

    monkeypatch.setattr(
        generalist_provider,
        "chat",
        lambda messages, max_new_tokens=1600: '[{"path":"../escape.txt","content":"bad"}]',
    )
    with pytest.raises(RuntimeError, match="Unsafe model change path|escapes"):
        local_agent.ask_local_model_changes("edit", "context", root=tmp_path / "repo")

    monkeypatch.setattr(
        generalist_provider,
        "chat",
        lambda messages, max_new_tokens=1600: '[{"path":".ssh/id_rsa","content":"bad"}]',
    )
    with pytest.raises(RuntimeError, match="Unsafe model change path"):
        local_agent.ask_local_model_changes("edit", "context", root=tmp_path / "repo")


def test_provider_reports_missing_torch_as_unavailable(tmp_path: Path, monkeypatch):
    from control_plane import generalist_provider

    make_qualified_checkpoint(tmp_path)
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")
    monkeypatch.setattr(generalist_provider.importlib.util, "find_spec", lambda name: None if name == "torch" else __import__("importlib").util.find_spec(name))
    row = generalist_provider.status()
    assert row["available"] is False
    assert row["runtime_dependency"] is False


def test_transformers_attestation_is_bound_to_exact_local_files(tmp_path: Path):
    from generalist_lm.qualification import (
        QUALIFICATION_VERSION,
        transformers_model_digest,
        transformers_qualification_status,
    )

    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (model_dir / "weights.bin").write_bytes(b"weights-v1")
    attestation = model_dir / "custom-attestation.json"
    digest = transformers_model_digest(model_dir, exclude_path=attestation)
    attestation.write_text(json.dumps({
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-transformers-qualification-v2",
        "backend_type": "transformers",
        "model_digest": digest,
        "qualified": True,
        "report": {"ok": True, "score": 100.0, "critical_failures": []},
    }), encoding="utf-8")

    before = transformers_qualification_status(model_dir, attestation_path=attestation)
    assert before["qualified"] is True
    assert before["integrity_ok"] is True

    (model_dir / "weights.bin").write_bytes(b"weights-v2")
    after = transformers_qualification_status(model_dir, attestation_path=attestation)
    assert after["qualified"] is False
    assert after["integrity_ok"] is False


def test_transformers_provider_requires_exact_qualification_and_dependencies(tmp_path: Path, monkeypatch):
    from control_plane import generalist_provider

    model_dir = tmp_path / "hf"
    model_dir.mkdir()
    attestation = tmp_path / "attestation.json"
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")
    monkeypatch.setenv("AIRI_GENERALIST_TRANSFORMERS_MODEL", str(model_dir))
    monkeypatch.setenv("AIRI_GENERALIST_TRANSFORMERS_ATTESTATION", str(attestation))
    monkeypatch.setattr(
        generalist_provider,
        "transformers_qualification_status",
        lambda model_dir, attestation_path=None: {"qualified": True, "integrity_ok": True},
    )
    real_find = generalist_provider.importlib.util.find_spec
    monkeypatch.setattr(
        generalist_provider.importlib.util,
        "find_spec",
        lambda name: object() if name in {"torch", "transformers"} else real_find(name),
    )

    row = generalist_provider.status()
    assert row["available"] is True
    assert row["backend"] == "transformers"
    assert row["runtime_dependency"] is True


def test_bootstrap_dependency_probe_keeps_generalist_runtime_optional():
    start = (ROOT / "computer" / "start.sh").read_text(encoding="utf-8")
    requirements = (ROOT / "computer" / "requirements.txt").read_text(encoding="utf-8")
    assert "import fastapi,uvicorn,pyautogui,pytesseract,PIL,playwright" in start
    assert 'if [ "${AIRI_GENERALIST_ENABLE:-0}" = "1" ]; then' in start
    assert "import torch,transformers" in start
    assert "generalist_lm/requirements.txt" in start
    assert "generalist_lm/requirements.txt" not in requirements
    assert "control_plane.model_gateway" in start


def test_readonly_generalist_tools_compute_without_code_execution():
    from control_plane.generalist_agent_bridge import execute_readonly_tool

    calc = execute_readonly_tool("calculator", {"expression": "17*19"})
    assert calc["value"] == 323

    stats = execute_readonly_tool(
        "data_stats",
        {"values": [2, 4, 6, 8], "operation": "mean"},
    )
    assert stats == {"operation": "mean", "result": 5.0, "count": 4}

    with pytest.raises(ValueError):
        execute_readonly_tool(
            "calculator",
            {"expression": "__import__('os').system('echo bad')"},
        )
    with pytest.raises(PermissionError):
        execute_readonly_tool("shell", {"command": "echo bad"})


def test_generalist_agent_bridge_executes_only_readonly_allowlist(monkeypatch):
    from control_plane import generalist_agent_bridge, generalist_provider

    class Backend:
        def __init__(self):
            self.calls = 0
        def chat(self, messages, *, max_new_tokens=256):
            self.calls += 1
            if self.calls == 1:
                return '<tool_call>{"name":"data_stats","arguments":{"values":[1,2,3],"operation":"sum"}}</tool_call>'
            assert any(m.get("role") == "tool" for m in messages)
            return "6"

    monkeypatch.setattr(generalist_provider, "status", lambda: {"available": True})
    monkeypatch.setattr(generalist_provider, "_backend", lambda: Backend())
    run = generalist_agent_bridge.run_generalist_agent(
        [{"role": "user", "content": "Sum 1,2,3"}],
        max_steps=3,
    )
    assert run.ok is True
    assert run.answer == "6"
    assert run.tool_calls == [{
        "name": "data_stats",
        "arguments": {"values": [1, 2, 3], "operation": "sum"},
    }]


def test_generalist_router_exposes_data_but_not_unimplemented_research(tmp_path: Path, monkeypatch):
    import control_plane.store as store
    from control_plane.model_router import ModelRouter

    make_qualified_checkpoint(tmp_path / "model")
    monkeypatch.setattr(store, "CP", tmp_path / "control-plane")
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path / "model"))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")
    monkeypatch.setenv("AIRI_GENERALIST_PREFER", "1")

    router = ModelRouter()
    assert router.choose(task_type="data")["selected"] == "airi-generalist"
    assert router.choose(task_type="research")["selected"] == "chatgpt"
    assert router.choose(task_type="vision", needs_vision=True)["selected"] == "chatgpt"


def test_code_agent_can_use_qualified_generalist_only_through_scoped_cycle(monkeypatch):
    import code_agent
    from control_plane import local_agent

    monkeypatch.setenv("AIRI_GENERALIST_AUTOCODE", "1")
    monkeypatch.setattr(local_agent, "provider_status", lambda: {"available": True})
    monkeypatch.setattr(
        local_agent,
        "ask_local_model_changes",
        lambda goal, context, feedback="", root=None, model=None: [
            {"path": "demo.py", "content": "def answer():\n    return 42\n"}
        ],
    )
    monkeypatch.setattr(code_agent, "load_project_context", lambda path=".": {"loaded": True, "content": "ctx"})
    monkeypatch.setattr(code_agent, "analyze", lambda path=".": {"project": path})
    monkeypatch.setattr(code_agent, "git_status", lambda path=".": {"available": True})
    monkeypatch.setattr(code_agent, "checkpoint", lambda *args, **kwargs: {"ok": True})
    seen = {}
    def fake_cycle(changes, project_path=".", test_command="", declared_scope=None, max_attempts=5):
        seen["changes"] = changes
        seen["scope"] = declared_scope
        return {"ok": True, "rolled_back": False}
    monkeypatch.setattr(code_agent, "autonomous_change_cycle", fake_cycle)

    result = code_agent.agent(
        "implement answer",
        project_path=".",
        scope=["demo.py"],
        test_command="pytest -q",
    )
    assert result["ok"] is True
    assert result["generated_by_generalist"] is True
    assert seen["scope"] == ["demo.py"]
    assert seen["changes"][0]["path"] == "demo.py"


def test_generalist_autocode_refuses_missing_scope(monkeypatch):
    import code_agent
    monkeypatch.setenv("AIRI_GENERALIST_AUTOCODE", "1")
    with pytest.raises(PermissionError, match="explicit declared scope"):
        code_agent.agent("edit something", project_path=".", scope=None)


def test_model_change_parser_accepts_focused_patch_and_rejects_ambiguous_edit():
    from control_plane import local_agent

    rows = local_agent._extract_json_array(
        '[{"path":"demo.py","old":"return 1","new":"return 2"}]'
    )
    assert rows == [{"path": "demo.py", "old": "return 1", "new": "return 2"}]

    with pytest.raises(RuntimeError):
        local_agent._extract_json_array(
            '[{"path":"demo.py","content":"x","old":"a","new":"b"}]'
        )


def test_readonly_generalist_table_profile_and_groupby():
    from control_plane.generalist_agent_bridge import execute_readonly_tool

    rows = [
        {"team": "a", "score": 10, "cost": 2.0},
        {"team": "a", "score": 20, "cost": None},
        {"team": "b", "score": 30, "cost": 4.0},
    ]
    profile = execute_readonly_tool("table_profile", {"rows": rows})
    assert profile["row_count"] == 3
    assert profile["profile"]["score"]["mean"] == 20.0
    assert profile["profile"]["cost"]["missing"] == 1
    assert profile["profile"]["cost"]["numeric_count"] == 2

    grouped = execute_readonly_tool(
        "table_aggregate",
        {
            "rows": rows,
            "operation": "mean",
            "column": "score",
            "group_by": "team",
        },
    )
    assert grouped["groups"] == {"a": 15.0, "b": 30.0}

    counted = execute_readonly_tool(
        "table_aggregate",
        {"rows": rows, "operation": "count", "group_by": "team"},
    )
    assert counted["groups"] == {"a": 2, "b": 1}


def test_readonly_generalist_table_tools_reject_non_numeric_and_excessive_groups():
    from control_plane.generalist_agent_bridge import execute_readonly_tool

    with pytest.raises(ValueError, match="numeric value"):
        execute_readonly_tool(
            "table_aggregate",
            {
                "rows": [{"group": "a", "value": "not-a-number"}],
                "operation": "mean",
                "column": "value",
            },
        )

    rows = [{"group": f"g{i}", "value": i} for i in range(201)]
    with pytest.raises(ValueError, match="too many groups"):
        execute_readonly_tool(
            "table_aggregate",
            {
                "rows": rows,
                "operation": "sum",
                "column": "value",
                "group_by": "group",
            },
        )


def test_generalist_autocode_requires_explicit_test_command(monkeypatch):
    import code_agent

    monkeypatch.setenv("AIRI_GENERALIST_AUTOCODE", "1")
    with pytest.raises(PermissionError, match="explicit test command"):
        code_agent.agent(
            "edit safely",
            project_path=".",
            scope=["demo.py"],
            test_command="",
        )


def test_generalist_registration_cannot_escalate_unimplemented_capabilities(tmp_path: Path, monkeypatch):
    import control_plane.store as store
    from control_plane.model_router import ModelRouter

    make_qualified_checkpoint(tmp_path / "model")
    monkeypatch.setattr(store, "CP", tmp_path / "control-plane")
    monkeypatch.setenv("AIRI_GENERALIST_STATE", str(tmp_path / "model"))
    monkeypatch.setenv("AIRI_GENERALIST_ENABLE", "1")

    router = ModelRouter()
    row = router.register_provider(
        "airi-generalist",
        ["coding", "vision", "research", "data"],
        available=True,
    )
    assert row["available"] is True
    assert set(row["capabilities"]) == {"coding", "data"}
    assert "vision" not in row["capabilities"]
    assert "research" not in row["capabilities"]
