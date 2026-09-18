from __future__ import annotations

import json
import os
import signal
import shutil
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
DRIFT_TRIGGER = int(os.environ.get("AIRI_EVOLUTION_DRIFT_TRIGGER_SAMPLES", "10"))


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
    if shutil.which("nvidia-smi"):
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)]
        install_profile = "default-gpu-capable"
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "torch>=2.7,<3", "--index-url", "https://download.pytorch.org/whl/cpu"]
        install_profile = "cpu-only"
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
    after = torch_status()
    return {
        "ok": p.returncode == 0 and after.get("available", False),
        "installed": p.returncode == 0,
        "install_profile": install_profile,
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



def bootstrap_liar(*, auto_evolve: bool = True, mode: str = "safe", enable_autopilot: bool = True) -> dict[str, Any]:
    from .liar import bootstrap
    result = bootstrap(STATE)
    started = None
    if result.get("ok") and auto_evolve:
        st = status()
        if st["dataset_records"] >= 40 and min(st["class_counts"].values()) >= 4 and not st["running"]:
            started = start(mode=mode, auto_setup=True)
    autopilot_result = autopilot(True) if result.get("ok") and enable_autopilot else None
    return {**result, "evolution_started": started, "autopilot": autopilot_result}


def queue_add(claim: str, metadata: dict | None = None) -> dict[str, Any]:
    from .pipeline import queue_claim
    return queue_claim(STATE, claim, metadata)


def queue_items(status_filter: str | None = None, limit: int = 100) -> list[dict]:
    from .pipeline import queue_list
    return queue_list(STATE, status=status_filter, limit=limit)


def queue_verify(qid: str, urls: list[str], *, min_sources: int = 2, auto_evolve: bool = True, mode: str = "safe", trigger_samples: int = DEFAULT_TRIGGER) -> dict[str, Any]:
    from .pipeline import verify_queued_claim
    result = verify_queued_claim(STATE, qid, urls, min_sources=min_sources, auto_ingest=True)
    started = None
    st = status()
    ingest_row = result.get("ingest") or {}
    if auto_evolve and result.get("status") == "verified" and not ingest_row.get("duplicate") and st["pending_verified_samples"] >= max(1, int(trigger_samples)) and st["dataset_records"] >= 40 and min(st["class_counts"].values()) >= 4 and not st["running"]:
        started = start(mode=mode, auto_setup=True)
    return {**result, "evolution_started": started, "evolution_status": st}


def pipeline_status() -> dict[str, Any]:
    from .pipeline import pipeline_status as read_pipeline
    return {**read_pipeline(STATE), "evolution": status()}


def report(history_limit: int = 20) -> dict[str, Any]:
    from .artifacts import report as build_report
    return build_report(STATE, history_limit=history_limit)


def export(out_path: str | None = None, include_torchscript: bool = False) -> dict[str, Any]:
    from .artifacts import export_bundle
    allowed = (STATE / "exports").resolve()
    target = None
    if out_path:
        raw = Path(out_path)
        target = raw if raw.is_absolute() else allowed / raw
        target = target.resolve()
        if target != allowed and allowed not in target.parents:
            return {"ok": False, "error": "export path must stay inside the evolution exports directory", "exports_dir": str(allowed)}
    return export_bundle(STATE, target, include_torchscript=include_torchscript)



def factcheck(claim: str, *, max_sources: int = 8, min_sources: int = 2, auto_evolve: bool = True, mode: str = "safe") -> dict[str, Any]:
    from advanced import research
    queued = queue_add(claim, {"origin": "airi_factcheck"})
    collected = []
    research_runs = []
    queries = [f'"{claim}" fact check', f'{claim} factcheck true false']
    for query in queries:
        try:
            rr = research(query, None, max(2, min(10, int(max_sources))))
            research_runs.append(rr)
            for source in rr.get("sources", []):
                url = source.get("url")
                if url and url not in collected:
                    collected.append(url)
        except Exception as exc:
            research_runs.append({"ok": False, "topic": query, "error": repr(exc), "sources": []})
    result = queue_verify(queued["id"], collected, min_sources=min_sources, auto_evolve=auto_evolve, mode=mode)
    model_prediction = None
    try:
        model_prediction = predict(claim)
    except Exception as exc:
        model_prediction = {"ok": False, "error": repr(exc)}
    return {
        "ok": result.get("status") == "verified",
        "claim": claim,
        "queue_id": queued["id"],
        "research": research_runs,
        "candidate_urls": collected,
        "verification": result,
        "model_prediction": model_prediction,
    }


def drift(window: int = 100, min_window: int = DRIFT_TRIGGER) -> dict[str, Any]:
    from .monitoring import drift_report
    return drift_report(STATE, window=window, min_window=min_window)


def edge_quantize(max_samples: int = 64) -> dict[str, Any]:
    if not torch_status().get("available"):
        return {"ok": False, "error": "torch_unavailable", "hint": "run airi-evolve setup"}
    from .edge import quantize_champion
    return quantize_champion(STATE, max_samples=max_samples)


def predict_edge(text: str) -> dict[str, Any]:
    if not torch_status().get("available"):
        return {"ok": False, "error": "torch_unavailable", "hint": "run airi-evolve setup"}
    from .edge import predict_edge_text
    try:
        return {"ok": True, **predict_edge_text(STATE, text)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def maintenance(*, mode: str = "safe") -> dict[str, Any]:
    st = status()
    drift_state = {"ok": True, "ready": False, "drift": False}
    if st["champion"] and st["pending_verified_samples"] >= max(1, DRIFT_TRIGGER):
        try:
            drift_state = drift(min_window=DRIFT_TRIGGER)
        except Exception as exc:
            drift_state = {"ok": False, "ready": False, "drift": False, "error": repr(exc)}
    started = None
    should_evolve = st["evolution_due"] or (
        drift_state.get("ready") and drift_state.get("drift") and
        st["pending_verified_samples"] >= max(1, DRIFT_TRIGGER)
    )
    if should_evolve and st["dataset_records"] >= 40 and min(st["class_counts"].values()) >= 4 and not st["running"]:
        started = start(mode=mode, auto_setup=True)
    return {
        "ok": True,
        "status": st,
        "drift": drift_state,
        "evolution_started": started,
        "trigger": "drift" if drift_state.get("drift") and started else ("new_verified_samples" if started else None),
        "pipeline": pipeline_status(),
    }



def autopilot(enable: bool = True, interval_seconds: int = 3600) -> dict[str, Any]:
    from advanced import cancel_job, schedule_job, scheduler_status
    name = "neuroevolution-maintenance"
    if enable:
        interval = max(300, int(interval_seconds))
        job = schedule_job(name, "evolution_maintenance", interval, run_now=False)
        return {"ok": True, "enabled": True, "job": job}
    jobs = {job.get("name"): job for job in scheduler_status().get("jobs", [])}
    if name not in jobs:
        return {"ok": True, "enabled": False, "already_disabled": True}
    cancelled = cancel_job(name)
    return {"ok": True, "enabled": False, "cancelled": cancelled}
