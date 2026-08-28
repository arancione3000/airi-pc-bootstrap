from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'computer'))
from control_plane.supervisor import Supervisor


def test_supervisor_single_instance_lock(tmp_path):
    a=Supervisor(str(tmp_path)); b=Supervisor(str(tmp_path))
    try:
        assert a.acquire_single_instance() is True
        assert b.acquire_single_instance() is False
        assert a.pid_path.read_text().strip().isdigit()
    finally:
        b.release_single_instance(); a.release_single_instance()


def test_supervisor_snapshot_probe_timeout(monkeypatch, tmp_path):
    s=Supervisor(str(tmp_path))
    def timeout(*args, **kwargs):
        import subprocess
        raise subprocess.TimeoutExpired(cmd='curl', timeout=1)
    monkeypatch.setattr(s, '_run', timeout)
    assert s.http_probe('http://127.0.0.1:9010/ready')['error']=='probe_timeout'


def test_recovery_does_not_accept_popen_without_ready(monkeypatch, tmp_path):
    s=Supervisor(str(tmp_path))
    monkeypatch.setattr(s, 'snapshot', lambda: {'timestamp': time.time(), 'ready': {'ok': False}, 'status': {'ok': False}, 'processes': []})
    script=tmp_path/'computer'/'start.sh'; script.parent.mkdir(parents=True); script.write_text('#!/bin/sh\nsleep 60\n'); script.chmod(0o755)
    class Proc:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): raise TimeoutError()
        def kill(self): pass
    monkeypatch.setattr('control_plane.supervisor.subprocess.Popen', lambda *a, **k: Proc())
    monkeypatch.setattr(s, '_wait_ready', lambda deadline: {'timestamp': time.time(), 'ready': {'ok': False}, 'status': {'ok': False}, 'processes': []})
    monkeypatch.setattr('control_plane.supervisor.time.sleep', lambda *_: None)
    result=s.recover()
    assert result['ok'] is False
    assert result['error']=='recovery_verification_failed'


def test_run_once_healthy_does_not_restart(tmp_path):
    s=Supervisor(str(tmp_path))
    s.snapshot=lambda: {'timestamp': time.time(), 'ready': {'ok': True}, 'status': {'ok': True}, 'processes': []}
    called=[]; s.recover=lambda: called.append(True)
    result=s.run_once()
    assert result['ok'] is True and result['recovered'] is False and called==[]
