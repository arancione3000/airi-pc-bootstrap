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

from . import lab

STATE = lab.LAB_STATE
PID = STATE / "worker.pid"
LOG = STATE / "worker.log"
AUTOPILOT_DISABLED = STATE / "autopilot.disabled"
DEFAULT_TRIGGER = max(10, int(os.environ.get("AIRI_EVOLUTION_LAB_TRIGGER", "20")))
DEFAULT_INTERVAL = max(300, int(os.environ.get("AIRI_EVOLUTION_LAB_INTERVAL", "900")))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _pid() -> int | None:
    try:
        value = int(PID.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    if not _pid_alive(value):
        PID.unlink(missing_ok=True)
        return None
    return value


def _safe_env() -> dict[str, str]:
    allowed = (
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR",
        "PYTHONHOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH",
    )
    env = {key: os.environ[key] for key in allowed if os.environ.get(key)}
    env.update({
        "AIRI_ROOT": str(lab.ROOT),
        "AIRI_EVOLUTION_LAB_STATE": str(STATE),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "AIRI_LAB_CPU_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(lab.ROOT / "computer"),
        "TMPDIR": str(STATE / "tmp"),
        "TEMP": str(STATE / "tmp"),
        "TMP": str(STATE / "tmp"),
        "XDG_CACHE_HOME": str(STATE / "cache"),
        "TORCH_HOME": str(STATE / "cache" / "torch"),
    })
    return env


def _sandbox_command() -> tuple[list[str], str]:
    base = [sys.executable, "-m", "evolution.lab_worker"]
    if os.name != "nt":
        unshare = shutil.which("unshare")
        if unshare:
            try:
                probe = subprocess.run(
                    [unshare, "-n", "--", "true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
                if probe.returncode == 0:
                    return [unshare, "-n", "--", *base], "network_namespace+python_audit_hook"
            except Exception:
                pass
    return base, "python_audit_hook"


def _apply_resource_limits(pid: int) -> dict[str, Any]:
    if os.name == "nt":
        return {"applied": False, "reason": "windows_no_prlimit"}
    try:
        import resource

        limits = {
            "cpu_seconds": 600,
            "address_space_bytes": 6 * 1024 * 1024 * 1024,
            "file_size_bytes": 512 * 1024 * 1024,
            "open_files": 128,
        }
        if hasattr(resource, "prlimit"):
            resource.prlimit(pid, resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
            resource.prlimit(pid, resource.RLIMIT_AS, (limits["address_space_bytes"], limits["address_space_bytes"]))
            resource.prlimit(pid, resource.RLIMIT_FSIZE, (limits["file_size_bytes"], limits["file_size_bytes"]))
            resource.prlimit(pid, resource.RLIMIT_NOFILE, (limits["open_files"], limits["open_files"]))
            resource.prlimit(pid, resource.RLIMIT_CORE, (0, 0))
            return {"applied": True, **limits}
        return {"applied": False, "reason": "prlimit_unavailable", **limits}
    except Exception as exc:
        return {"applied": False, "reason": repr(exc)}


def audit() -> dict[str, Any]:
    root = STATE.resolve(strict=False)
    workspace = lab.ROOT.resolve(strict=False)
    errors: list[str] = []
    warnings: list[str] = []
    symlinks = []
    if STATE.exists():
        for path in STATE.rglob("*"):
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=False)
            except Exception:
                target = None
            row = {"path": str(path), "target": str(target) if target else None}
            symlinks.append(row)
            if target is None or (target != root and root not in target.parents):
                errors.append(f"sandbox symlink escapes lab root: {path}")
    if root == workspace or workspace not in root.parents:
        errors.append("lab state directory is not isolated under the Airi workspace")
    worker = _pid()
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "state_dir": str(root),
            "worker_running": bool(worker),
            "worker_pid": worker,
            "symlinks": symlinks,
            "shadow_only": True,
            "network_policy": "blocked_by_python_audit_hook",
            "subprocess_policy": "blocked_inside_worker",
            "production_router_write": False,
        },
    }


def status() -> dict[str, Any]:
    base = lab.status()
    worker = _pid()
    return {
        **base,
        "worker_running": bool(worker),
        "worker_pid": worker,
        "trigger_observations": DEFAULT_TRIGGER,
        "autopilot_interval_seconds": DEFAULT_INTERVAL,
        "audit": audit(),
        "autopilot_enabled": not AUTOPILOT_DISABLED.exists(),
    }


def start(*, auto_setup: bool = True) -> dict[str, Any]:
    if _pid():
        return {"ok": True, "started": False, "reason": "already_running", "pid": _pid()}
    state_audit = audit()
    if not state_audit["ok"]:
        return {"ok": False, "started": False, "reason": "sandbox_audit_failed", "audit": state_audit}

    from . import runtime as evolution_runtime
    torch = evolution_runtime.torch_status()
    if not torch.get("available") and auto_setup:
        setup = evolution_runtime.setup()
        if not setup.get("ok"):
            return {"ok": False, "started": False, "reason": "torch_setup_failed", "setup": setup}
    elif not torch.get("available"):
        return {"ok": False, "started": False, "reason": "torch_unavailable"}

    for directory in (STATE, STATE / "tmp", STATE / "cache"):
        directory.mkdir(parents=True, exist_ok=True)

    command, network_isolation = _sandbox_command()
    if LOG.exists() and LOG.stat().st_size > 10 * 1024 * 1024:
        rotated = LOG.with_suffix(".log.1")
        rotated.unlink(missing_ok=True)
        LOG.replace(rotated)
    handle = LOG.open("ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": str(lab.ROOT / "computer"),
        "stdout": handle,
        "stderr": subprocess.STDOUT,
        "env": _safe_env(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)
    finally:
        handle.close()

    limits = _apply_resource_limits(proc.pid)
    PID.write_text(str(proc.pid), encoding="utf-8")
    return {
        "ok": True,
        "started": True,
        "pid": proc.pid,
        "state_dir": str(STATE),
        "resource_limits": limits,
        "shadow_only": True,
        "network_isolation": network_isolation,
    }


def stop() -> dict[str, Any]:
    pid = _pid()
    if not pid:
        return {"ok": True, "stopped": False, "reason": "not_running"}
    try:
        if os.name != "nt" and hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.time() + 8
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
        except OSError:
            pass
    PID.unlink(missing_ok=True)
    return {"ok": True, "stopped": True, "pid": pid}


def maintenance() -> dict[str, Any]:
    st = status()
    started = None
    if not st["worker_running"] and st["pending_observations"] >= DEFAULT_TRIGGER:
        started = start(auto_setup=True)
    return {
        "ok": True,
        "shadow_only": True,
        "status": st,
        "training_started": started,
    }


def autopilot(enable: bool = True, interval_seconds: int = DEFAULT_INTERVAL) -> dict[str, Any]:
    from advanced import cancel_job, schedule_job, scheduler_status

    name = "evolution-lab-shadow-router"
    STATE.mkdir(parents=True, exist_ok=True)
    if enable:
        AUTOPILOT_DISABLED.unlink(missing_ok=True)
        interval = max(300, int(interval_seconds))
        job = schedule_job(name, "evolution_lab_maintenance", interval, run_now=True)
        return {"ok": True, "enabled": True, "job": job, "shadow_only": True}
    AUTOPILOT_DISABLED.write_text("disabled\n", encoding="utf-8")
    jobs = {row.get("name"): row for row in scheduler_status().get("jobs", [])}
    if name not in jobs:
        return {"ok": True, "enabled": False, "already_disabled": True, "persistent": True}
    return {"ok": True, "enabled": False, "persistent": True, "cancelled": cancel_job(name)}


def score(goal: str, operation: str, candidates: list[str], args: Any = None) -> dict[str, Any]:
    return lab.score_candidates(goal, operation, candidates, args)
