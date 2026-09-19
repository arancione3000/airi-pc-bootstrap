from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution import daemon, lab, lab_runtime, state_sync


def _git(args, cwd: Path | None = None):
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=True,
    )


def _patch_state(monkeypatch, workspace: Path):
    state = workspace / ".ai" / "evolution-lab" / "shadow-router"
    monkeypatch.setattr(lab, "ROOT", workspace)
    monkeypatch.setattr(lab, "LAB_STATE", state)
    monkeypatch.setattr(lab, "RAW", state / "raw" / "observations.jsonl")
    monkeypatch.setattr(lab, "DATA", state / "data" / "verified.jsonl")
    monkeypatch.setattr(lab, "META", state / "lab-meta.json")

    monkeypatch.setattr(lab_runtime, "STATE", state)
    monkeypatch.setattr(lab_runtime, "PID", state / "worker.pid")
    monkeypatch.setattr(lab_runtime, "LOG", state / "worker.log")
    monkeypatch.setattr(lab_runtime, "AUTOPILOT_DISABLED", state / "autopilot.disabled")
    monkeypatch.setattr(lab_runtime, "DAEMON_PID", state / "daemon.pid")
    monkeypatch.setattr(lab_runtime, "DAEMON_STATUS", state / "daemon-status.json")
    monkeypatch.setattr(lab_runtime, "SYNC_META", state / "sync-meta.json")

    sync_root = workspace / ".ai" / "evolution-git"
    monkeypatch.setattr(state_sync, "SOURCE_ROOT", workspace)
    monkeypatch.setattr(state_sync, "SYNC_ROOT", sync_root)
    monkeypatch.setattr(state_sync, "CHECKOUT", sync_root / "checkout")
    monkeypatch.setattr(state_sync, "SYNC_META", state / "sync-meta.json")

    monkeypatch.setattr(daemon, "STATE", state)
    monkeypatch.setattr(daemon, "LOCK", state / "daemon.lock")
    monkeypatch.setattr(daemon, "PID", state / "daemon.pid")
    monkeypatch.setattr(daemon, "STATUS", state / "daemon-status.json")
    return state


def _init_remote(tmp_path: Path):
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    _git(["init", "--bare", str(remote)])
    source.mkdir()
    _git(["init"], source)
    _git(["config", "user.name", "Test"], source)
    _git(["config", "user.email", "test@example.com"], source)
    (source / "README.md").write_text("source\n", encoding="utf-8")
    _git(["add", "README.md"], source)
    _git(["commit", "-m", "base"], source)
    _git(["branch", "-M", "main"], source)
    _git(["remote", "add", "origin", str(remote)], source)
    _git(["push", "-u", "origin", "main"], source)
    _git(["checkout", "-b", "airi-evolution-state"], source)
    state_dir = source / "evolution-state" / "shadow-router"
    state_dir.mkdir(parents=True)
    (state_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "feature_schema": 2,
                "exported_at": 0,
                "state_digest": "",
                "files": {},
                "privacy": {
                    "raw_observations_exported": False,
                    "goal_text_exported": False,
                    "argument_values_exported": False,
                    "dataset_audit": {"ok": True, "records": 0, "errors": []},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(["add", "evolution-state"], source)
    _git(["commit", "-m", "seed state"], source)
    _git(["push", "-u", "origin", "airi-evolution-state"], source)
    _git(["checkout", "main"], source)
    return source, remote


def test_route_feature_never_contains_goal_or_argument_values():
    goal = (
        "Please open https://secret.example/private for alice@example.com "
        "password=hunter2 file /home/alice/very-secret.txt order 123456"
    )
    feature = lab.route_feature(
        goal,
        "browser_open",
        "computer_browser_open",
        {"password": "hunter2", "url": "https://secret.example/private", "count": 123456},
        ["computer_browser_open", "computer_browser_text"],
    )
    for secret in (
        "secret.example",
        "alice@example.com",
        "hunter2",
        "/home/alice",
        "123456",
        "password",
        "url:str",
        "count:int",
    ):
        assert secret not in feature
    assert feature.startswith("goal_shape ")
    assert "operation browser_open" in feature
    assert "tool computer_browser_open" in feature
    assert "arg_shape count:3" in feature
    assert "computer_browser_text" in feature


def test_legacy_observations_are_purged_before_training_export(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "airi"
    workspace.mkdir()
    _patch_state(monkeypatch, workspace)
    lab.RAW.parent.mkdir(parents=True)
    lab.RAW.write_text(
        json.dumps(
            {
                "id": "legacy",
                "feature": "goal TOP SECRET old feature",
                "label": 1,
                "feature_schema": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lab.record_execution(
        goal="TOP SECRET should never survive",
        operation="read",
        tool="computer_file_read",
        args={"path": "/private/file.txt"},
        success=True,
        candidates=["computer_file_read"],
    )
    result = lab.rebuild_dataset()
    assert result["observations"] == 1
    assert "TOP SECRET" not in lab.DATA.read_text(encoding="utf-8")
    assert "TOP SECRET" not in lab.RAW.read_text(encoding="utf-8")
    assert '"feature_schema": 1' not in lab.RAW.read_text(encoding="utf-8")


def test_state_sync_pushes_and_verifies_remote_commit(monkeypatch, tmp_path: Path):
    source, remote = _init_remote(tmp_path)
    _patch_state(monkeypatch, source)
    monkeypatch.setenv("AIRI_EVOLUTION_GIT_URL", str(remote))

    for i in range(8):
        lab.record_execution(
            goal=("short task" if i % 2 else "a longer task with several words and two lines\nsecond line"),
            operation="read" if i < 4 else "search",
            tool="computer_file_read" if i < 4 else "computer_file_search",
            args={"value": i, "flag": bool(i % 2)},
            success=i % 3 != 0,
            candidates=["computer_file_read", "computer_file_search"],
        )
    lab.rebuild_dataset()

    result = state_sync.sync_to_git(force_heartbeat=True)
    assert result["ok"] is True
    assert result["synced"] is True
    assert result["commit"] == result["remote_sha"]

    verify = tmp_path / "verify"
    _git(["clone", "--branch", "airi-evolution-state", str(remote), str(verify)])
    exported = verify / "evolution-state" / "shadow-router"
    assert (exported / "manifest.json").exists()
    assert (exported / "data" / "verified.jsonl").exists()
    assert not (exported / "raw").exists()
    audit = state_sync.privacy_audit_dataset(exported / "data" / "verified.jsonl")
    assert audit["ok"] is True
    data_text = (exported / "data" / "verified.jsonl").read_text(encoding="utf-8")
    assert "short task" not in data_text
    assert "second line" not in data_text


def test_restore_recovers_safe_training_state_from_remote(monkeypatch, tmp_path: Path):
    source, remote = _init_remote(tmp_path)
    state = _patch_state(monkeypatch, source)
    monkeypatch.setenv("AIRI_EVOLUTION_GIT_URL", str(remote))

    for i in range(6):
        lab.record_execution(
            goal=f"restore sample {i}",
            operation="read",
            tool="computer_file_read",
            args={"n": i},
            success=bool(i % 2),
            candidates=["computer_file_read"],
        )
    lab.rebuild_dataset()
    pushed = state_sync.sync_to_git(force_heartbeat=True)
    assert pushed["ok"] and pushed["synced"]

    if state.exists():
        shutil.rmtree(state)
    restored = state_sync.restore_from_git(force=True)
    assert restored["ok"] is True
    assert restored["restored"] is True
    assert restored["records"] == 6
    assert lab.DATA.exists()
    assert state_sync.privacy_audit_dataset(lab.DATA)["ok"] is True


def test_daemon_offline_cycle_budget_resets_only_for_new_dataset(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "airi"
    workspace.mkdir()
    _patch_state(monkeypatch, workspace)

    status = {
        "worker_running": False,
        "training_records": 50,
        "success_records": 25,
        "failure_records": 25,
        "unique_route_features": 20,
        "pending_observations": 0,
        "champion_compatible": True,
    }
    monkeypatch.setattr(lab_runtime, "status", lambda: dict(status))
    monkeypatch.setattr(lab_runtime, "maintenance", lambda: {"ok": True, "training_started": None})
    starts = []
    monkeypatch.setattr(lab_runtime, "start", lambda **kwargs: starts.append(True) or {"ok": True, "started": True})
    monkeypatch.setattr(state_sync, "dataset_digest", lambda: "dataset-A")
    monkeypatch.setattr(state_sync, "sync_to_git", lambda **kwargs: {"ok": True, "synced": False})
    monkeypatch.setattr(state_sync, "_source_sha", lambda: "same-source")

    d = daemon.EvolutionDaemon()
    d.last_offline_cycle_at = 0
    monkeypatch.setattr(daemon, "OFFLINE_EVOLUTION_SECONDS", 1)
    first = d.run_once()
    assert first["offline_cycles_on_dataset"] == 1
    assert starts == [True]

    status["worker_running"] = False
    d.previous_worker_running = False
    d.last_offline_cycle_at = 0
    daemon.STATUS.write_text(
        json.dumps(
            {
                "dataset_digest": "dataset-A",
                "offline_cycles_on_dataset": daemon.MAX_OFFLINE_CYCLES_PER_DATASET,
                "last_offline_cycle_at": 0,
            }
        ),
        encoding="utf-8",
    )
    second = d.run_once()
    assert second["offline_cycle"] is None

    monkeypatch.setattr(state_sync, "dataset_digest", lambda: "dataset-B")
    d.last_offline_cycle_at = 0
    third = d.run_once()
    assert third["offline_cycles_on_dataset"] == 1


def test_runtime_restart_does_not_target_evolution_daemon():
    text = (ROOT / "computer" / "start.sh").read_text(encoding="utf-8")
    start = text.index("stop_server() {")
    end = text.index("\n}\nif [", start)
    stop_block = text[start:end]
    assert "evolution.daemon" not in stop_block
    assert "EVOLUTION_DAEMON_PID_FILE" in text
    assert '"$PYTHON_BIN" -m evolution.daemon' in text


def test_systemd_evolution_service_is_independent_and_restartable():
    text = (
        ROOT
        / "airi-os"
        / "config"
        / "includes.chroot"
        / "etc"
        / "systemd"
        / "system"
        / "airi-evolution-lab.service"
    ).read_text(encoding="utf-8")
    assert "Restart=always" in text
    assert "ExecStart=/opt/airi-pc/venv-current/bin/python -m evolution.daemon" in text
    assert "PartOf=airi-pc.service" not in text


def test_privacy_audit_rejects_unexpected_evidence_payload(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "airi"
    workspace.mkdir()
    _patch_state(monkeypatch, workspace)
    lab.DATA.parent.mkdir(parents=True)
    lab.DATA.write_text(
        json.dumps(
            {
                "id": "x",
                "feature_schema": lab.FEATURE_SCHEMA,
                "text": "goal_shape chars:le32,words:le5,lines:le1,url:0,email:0,path:0,digits:le0 operation read tool computer_file_read arg_shape count:0;none candidates computer_file_read",
                "text_id": "x",
                "label": 1,
                "source": "airi-shadow-route-observation",
                "evidence": json.dumps({"operation": "read", "tool": "computer_file_read", "secret": "https://leak.example"}),
                "added_at": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit = state_sync.privacy_audit_dataset(lab.DATA)
    assert audit["ok"] is False
    assert any("unexpected evidence fields" in error for error in audit["errors"])
