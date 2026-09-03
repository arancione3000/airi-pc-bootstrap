from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from startup import git_source_sha, resolve_expected_sha, resolve_runtime_sha, runtime_needs_restart


def test_start_script_uses_configurable_display_everywhere():
    text = (ROOT / "computer" / "start.sh").read_text(encoding="utf-8")
    assert 'DISPLAY_NUM="${DISPLAY_NUM:-99}"' in text
    assert 'export DISPLAY=":$DISPLAY_NUM"' in text
    assert "Xvfb :99" not in text
    assert "DISPLAY=:99" not in text
    assert "[X]vfb :99" not in text
    assert 'Xvfb "$DISPLAY"' in text
    assert 'DISPLAY="$DISPLAY" xdpyinfo' in text
    assert 'env DISPLAY="$DISPLAY" openbox' in text
    assert 'env DISPLAY="$DISPLAY" xterm' in text


def test_runtime_sha_prefers_live_git_over_stale_marker(monkeypatch, tmp_path):
    monkeypatch.setattr("startup.git_source_sha", lambda root: "live-sha")
    marker = tmp_path / ".runtime_source_sha"
    marker.write_text("stale-sha\n", encoding="utf-8")
    env = {"AIRI_BOOTSTRAP_SHA": "also-stale"}
    assert resolve_runtime_sha(tmp_path, marker, env) == "live-sha"
    assert resolve_expected_sha(tmp_path, marker, env) == "live-sha"


def test_expected_sha_explicit_override_is_honored(monkeypatch, tmp_path):
    monkeypatch.setattr("startup.git_source_sha", lambda root: "git-sha")
    assert resolve_expected_sha(tmp_path, tmp_path / "missing", {"AIRI_EXPECTED_SHA": "required-sha"}) == "required-sha"


def test_stale_runtime_is_rejected_and_matching_runtime_is_reused():
    stale = {"ready": True, "source_sha": "old", "checks": {"source_match": False}}
    current = {"ready": True, "source_sha": "new", "checks": {"source_match": True}}
    not_ready = {"ready": False, "source_sha": "new", "checks": {"source_match": True}}
    assert runtime_needs_restart(stale, "new") is True
    assert runtime_needs_restart(not_ready, "new") is True
    assert runtime_needs_restart(current, "new") is False
    assert runtime_needs_restart(current, "") is False


def test_startup_script_contains_stale_runtime_guard():
    text = (ROOT / "computer" / "start.sh").read_text(encoding="utf-8")
    assert "runtime_ready_matches" in text
    assert "runtime_needs_restart" in text
    assert "Never reuse a server merely because /status answers" in text


def test_git_source_sha_returns_empty_for_non_repo(tmp_path):
    assert git_source_sha(tmp_path) == ""
