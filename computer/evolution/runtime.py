from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .data import append_verified, class_counts, load_records

ROOT = Path(os.environ.get("AIRI_ROOT") or Path(__file__).resolve().parents[2]).resolve()
STATE = Path(os.environ.get("AIRI_EVOLUTION_STATE") or ROOT / ".ai" / "evolution").resolve()
DATA = STATE / "data" / "verified.jsonl"
STATUS = STATE / "status.json"
PID = STATE / "evolution.pid"
LOG = STATE / "evolution.log"
DEFAULT_TRIGGER = int(os.environ.get("AIRI_EVOLUTION_TRIGGER_SAMPLES", "20"))


def _json_read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def torch_status() -> dict[str, Any]:
    code = "import json; import torch; print(json.dumps({'version':torch.__version__,'cuda':torch.cuda.is_available(),'cuda_devices':torch.cuda.device_count()}))"
    p = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
    if p.returncode != 0:
        return {"available": False, "error": (p.stderr or p.stdout).strip()[-1000:]}
    try:
        return {"available": True, **json.loads(p.stdout.strip().splitlines()[-1])}
    except Exception:
        return {"available": True, "raw": p.stdout.strip()[-1000:]}


def setup() -> dict[str, Any]:
    current = torch_status()
    if current.get("available"):
        return {"ok": True, "installed": False, "torch": current}
    req = Path(__file__).with_name("requirements.txt")
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
    after = torch_status()
    return {
        "ok": p.returncode == 0 and after.get("available", False),
        "installed": p.returncode == 0,
        "torch": after,
        "returncode": p.returncode,
        "stdout": p.stdout[-3000:],
        "stderr": p.stderr[-3000:],
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _last_trained_count() -> int:
    prov = _json_read(STATE / "champion" / "provenance.json", {})
    return int(prov.get("dataset_records", 0) or 0)


def status() -> dict[str, Any]:
    records = load_records(DATA)
    counts = class_counts(records)
    pid = int(PID.read_text().strip()) if PID.exists() and PID.read_text().strip().isdigit() else None
    running = bool(pid and _pid_alive(pid))
    if pid and not running:
        PID.unlink(missing_ok=True)
    trained_count = _last_trained_count()
    pending = max(0, len(records) - trained_count)
    champion_metrics = _json_read(STATE / "champion" / "metrics.json", None)
    return {
        "ok": True,
        "state_dir": str(STATE),
        "dataset_records": len(records),
        "class_counts": counts,
        "last_champion_dataset_records": trained_count,
        "pending_verified_samples": pending,
        "evolution_due": pending >= DEFAULT_TRIGGER,
        "running": running,
        "pid": pid if running else None,
        "champion": champion_metrics,
        "last_run_status": _json_read(STATUS, None),
        "torch": torch_status(),
    }


def ingest(record: dict, *, auto_evolve: bool = True, mode: str = "safe", trigger_samples: int = DEFAULT_TRIGGER) -> dict[str, Any]:
    row = append_verified(DATA, record)
    st = status()
    started = None
    if auto_evolve and not row.get("duplicate") and st["pending_verified_samples"] >= max(1, int(trigger_samples)) and min(st["class_counts"].values()) >= 4 and st["dataset_records"] >= 40:
        started = start(mode=mode, auto_setup=True)
    return {"ok": True, "record": row, "status": st, "evolution_started": started}


def start(*, mode: str = "safe", auto_setup: bool = True, population: int | None = None, generations: int | None = None, candidate_epochs: int | None = None) -> dict[str, Any]:
    st = status()
    if st["running"]:
        return {"ok": True, "started": False, "reason": "already_running", "pid": st["pid"]}
    if st["dataset_records"] < 40 or min(st["class_counts"].values()) < 4:
        return {"ok": False, "started": False, "reason": "insufficient_verified_data", "status": st}
    ts = torch_status()
    if not ts.get("available") and auto_setup:
        installed = setup()
        if not installed.get("ok"):
            return {"ok": False, "started": False, "reason": "torch_setup_failed", "setup": installed}
    elif not ts.get("available"):
        return {"ok": False, "started": False, "reason": "torch_unavailable", "torch": ts}

    STATE.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "evolution.cli", "_run", "--mode", mode]
    for flag, value in (("--population", population), ("--generations", generations), ("--candidate-epochs", candidate_epochs)):
        if value is not None:
            cmd.extend([flag, str(value)])
    log = LOG.open("ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=str(ROOT / "computer"), stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "AIRI_ROOT": str(ROOT), "PYTHONPATH": str(ROOT / "computer") + os.pathsep + os.environ.get("PYTHONPATH", "")})
    PID.write_text(str(proc.pid), encoding="utf-8")
    _json_write(STATUS, {"state": "running", "pid": proc.pid, "mode": mode, "started_at": time.time(), "command": cmd})
    return {"ok": True, "started": True, "pid": proc.pid, "mode": mode, "log": str(LOG)}


def stop() -> dict[str, Any]:
    if not PID.exists() or not PID.read_text().strip().isdigit():
        return {"ok": True, "stopped": False, "reason": "not_running"}
    pid = int(PID.read_text().strip())
    if not _pid_alive(pid):
        PID.unlink(missing_ok=True)
        return {"ok": True, "stopped": False, "reason": "stale_pid"}
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    PID.unlink(missing_ok=True)
    _json_write(STATUS, {"state": "stopped", "pid": pid, "stopped_at": time.time()})
    return {"ok": True, "stopped": True, "pid": pid}


def history(limit: int = 10) -> list[dict]:
    path = STATE / "history.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows[-max(1, min(200, int(limit))):]


def predict(text: str) -> dict[str, Any]:
    if not torch_status().get("available"):
        return {"ok": False, "error": "torch_unavailable", "hint": "run airi-evolve setup"}
    from .engine import predict_text
    return {"ok": True, **predict_text(STATE, text)}
