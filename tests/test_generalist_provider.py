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
        "attested_by": "airi-generalist-qualification-v1",
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
