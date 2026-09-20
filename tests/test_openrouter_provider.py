import sys
import pytest

sys.path.insert(0, "computer")

from control_plane import local_agent
from control_plane.model_router import ModelRouter


def test_default_surface_keeps_chatgpt_as_reasoning_authority(monkeypatch):
    monkeypatch.delenv("AIRI_GENERALIST_ENABLE", raising=False)
    monkeypatch.delenv("AIRI_GENERALIST_PREFER", raising=False)
    status = local_agent.provider_status()
    assert local_agent.CHATGPT_ONLY is True
    assert local_agent.REASONING_AUTHORITY == "chatgpt"
    assert status["available"] is False
    assert status["reasoning_authority"] == "chatgpt"


def test_local_agent_calls_fail_closed_without_qualified_checkpoint(monkeypatch):
    monkeypatch.delenv("AIRI_GENERALIST_ENABLE", raising=False)
    with pytest.raises(RuntimeError, match="reasoning authority|qualified AIRI Generalist"):
        local_agent._provider_request([{"role": "user", "content": "x"}], "airi-generalist")
    with pytest.raises(RuntimeError, match="reasoning authority|qualified AIRI Generalist"):
        local_agent.ask_local_model("goal")
    with pytest.raises(RuntimeError, match="reasoning authority|qualified AIRI Generalist"):
        local_agent.ask_local_model_changes("goal", "context")


def test_openrouter_cannot_be_registered_operationally(tmp_path, monkeypatch):
    import control_plane.store as store
    monkeypatch.setattr(store, "CP", tmp_path / "control-plane")
    monkeypatch.setenv("AIRI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    monkeypatch.delenv("AIRI_GENERALIST_ENABLE", raising=False)
    router = ModelRouter()
    assert set(router.state["providers"]) == {"chatgpt"}
    assert router.choose(task_type="coding")["selected"] == "chatgpt"
    result = router.register_provider("openrouter", ["coding"], available=True)
    assert result["available"] is False
    assert "openrouter" not in router.state["providers"]
