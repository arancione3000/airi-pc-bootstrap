from __future__ import annotations
import json, os, signal, subprocess, time
from pathlib import Path


def _fake_supervisor(base: Path):
    script=base/'scripts'/'airi-supervisor'; script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("""#!/bin/sh
mkdir -p \"$AIRI_STATE_DIR\" \"$BASE/.ai/control_plane\"
echo $$ > \"$AIRI_STATE_DIR/supervisor.pid\"
printf '%s' '{\"timestamp\":'\"$(date +%s)\"',\"ok\":true}' > \"$BASE/.ai/control_plane/supervisor-run.json\"
exec -a computer/control_plane/supervisor.py sleep 30
""")
    script.chmod(0o755)


def test_watchdog_single_instance(tmp_path):
    base=tmp_path; (base/'.ai/state').mkdir(parents=True)
    _fake_supervisor(base)
    env={**os.environ,'AIRI_BASE':str(base),'BASE':str(base),'AIRI_STATE_DIR':str(base/'.ai/state'),'AIRI_WATCHDOG_INTERVAL':'0.1','AIRI_SUPERVISOR_STALE_SECONDS':'20'}
    wd=Path(__file__).resolve().parents[1]/'scripts/airi-watchdog'
    first=subprocess.Popen([str(wd)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        time.sleep(0.5)
        second=subprocess.run([str(wd)],env=env,capture_output=True,text=True,timeout=3)
        assert second.returncode==20
        assert first.poll() is None
        assert (base/'.ai/state/watchdog.pid').read_text().strip()==str(first.pid)
    finally:
        os.killpg(first.pid, signal.SIGTERM)
        try: first.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(first.pid, signal.SIGKILL); first.wait(timeout=3)


def test_watchdog_stale_pid_starts_supervisor(tmp_path):
    base=tmp_path; (base/'.ai/state').mkdir(parents=True)
    _fake_supervisor(base)
    (base/'.ai/state/supervisor.pid').write_text('999999\n')
    env={**os.environ,'AIRI_BASE':str(base),'BASE':str(base),'AIRI_STATE_DIR':str(base/'.ai/state'),'AIRI_WATCHDOG_INTERVAL':'0.1','AIRI_SUPERVISOR_STALE_SECONDS':'20','AIRI_WATCHDOG_MAX_RETRIES':'1'}
    wd=Path(__file__).resolve().parents[1]/'scripts/airi-watchdog'
    proc=subprocess.Popen([str(wd)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        deadline=time.time()+5
        while time.time()<deadline:
            hb=base/'.ai/control_plane/supervisor-run.json'; pid=base/'.ai/state/supervisor.pid'
            if hb.exists() and pid.exists() and pid.read_text().strip().isdigit() and int(pid.read_text()) != 999999:
                break
            time.sleep(0.1)
        assert (base/'.ai/control_plane/supervisor-run.json').exists()
        assert int((base/'.ai/state/supervisor.pid').read_text()) != 999999
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=3)
        try:
            os.kill(int((base/'.ai/state/supervisor.pid').read_text()),9)
        except Exception:
            pass
