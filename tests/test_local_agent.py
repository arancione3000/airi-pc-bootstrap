from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from control_plane import local_agent


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
