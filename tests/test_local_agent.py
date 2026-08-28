from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_agent", ROOT / "computer" / "control_plane" / "local_agent.py")
local_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_agent)


def test_safe_patch_accepts_repo_file(monkeypatch):
    monkeypatch.setattr(local_agent, "ROOT", ROOT.resolve())
    local_agent._safe_patch("--- a/README.md\n+++ b/README.md\n@@\n")


def test_safe_patch_rejects_sensitive_path(monkeypatch):
    monkeypatch.setattr(local_agent, "ROOT", ROOT.resolve())
    try:
        local_agent._safe_patch("--- a/.ssh/id_rsa\n+++ b/.ssh/id_rsa\n@@\n")
    except RuntimeError:
        return
    raise AssertionError("sensitive patch path was accepted")
