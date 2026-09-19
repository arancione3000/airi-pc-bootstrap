from __future__ import annotations

import fcntl
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from . import lab, lab_runtime, state_sync

STATE = lab.LAB_STATE
LOCK = STATE / "daemon.lock"
PID = STATE / "daemon.pid"
STATUS = STATE / "daemon-status.json"

LOOP_SECONDS = max(60, int(os.environ.get("AIRI_EVOLUTION_DAEMON_INTERVAL", "300")))
OFFLINE_EVOLUTION_SECONDS = max(1800, int(os.environ.get("AIRI_EVOLUTION_OFFLINE_INTERVAL", "7200")))
SYNC_SECONDS = max(900, int(os.environ.get("AIRI_EVOLUTION_SYNC_INTERVAL", "1800")))


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _eligible_for_offline_cycle(st: dict[str, Any]) -> bool:
    return (
        not st.get("worker_running")
        and int(st.get("training_records", 0)) >= 40
        and int(st.get("success_records", 0)) >= 4
        and int(st.get("failure_records", 0)) >= 4
        and int(st.get("unique_route_features", 0)) >= 12
    )


class EvolutionDaemon:
    def __init__(self) -> None:
        self.stop_requested = False
        self.lock_fh = None
        self.started_at = time.time()
        self.last_sync_at = 0.0
        self.last_offline_cycle_at = 0.0
        self.previous_worker_running = False

    def acquire(self) -> bool:
        STATE.mkdir(parents=True, exist_ok=True)
        self.lock_fh = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_fh.close()
            self.lock_fh = None
            return False
        PID.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        return True

    def release(self) -> None:
        try:
            if PID.exists() and PID.read_text(encoding="utf-8").strip() == str(os.getpid()):
                PID.unlink()
        except OSError:
            pass
        if self.lock_fh is not None:
            try:
                fcntl.flock(self.lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_fh.close()
                self.lock_fh = None

    def stop(self, *_args) -> None:
        self.stop_requested = True

    def restore_once(self) -> dict[str, Any]:
        marker = STATE / "restore-attempt.json"
        if marker.exists():
            return {"ok": True, "restored": False, "reason": "restore_already_attempted"}
        try:
            result = state_sync.restore_from_git(force=False)
        except Exception as exc:
            result = {"ok": False, "restored": False, "reason": "restore_exception", "error": repr(exc)}
        _write_json(marker, {"attempted_at": time.time(), **result})
        return result

    def run_once(self) -> dict[str, Any]:
        now = time.time()
        before = lab_runtime.status()
        maintenance = lab_runtime.maintenance()

        worker_running = bool(lab_runtime.status().get("worker_running"))
        worker_finished = self.previous_worker_running and not worker_running
        self.previous_worker_running = worker_running

        offline = None
        state = _read_json(STATUS, {})
        last_offline = float(state.get("last_offline_cycle_at", self.last_offline_cycle_at) or 0)
        current = lab_runtime.status()
        if (
            not worker_running
            and maintenance.get("training_started") is None
            and _eligible_for_offline_cycle(current)
            and now - last_offline >= OFFLINE_EVOLUTION_SECONDS
        ):
            offline = lab_runtime.start(auto_setup=True)
            if offline.get("ok") and offline.get("started"):
                self.last_offline_cycle_at = now
                worker_running = True
                self.previous_worker_running = True

        sync = None
        last_sync = float(state.get("last_sync_at", self.last_sync_at) or 0)
        if not worker_running and (worker_finished or now - last_sync >= SYNC_SECONDS):
            try:
                sync = state_sync.sync_to_git(force_heartbeat=worker_finished)
            except Exception as exc:
                sync = {"ok": False, "synced": False, "reason": "sync_exception", "error": repr(exc)}
            self.last_sync_at = now

        result = {
            "ok": True,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "checked_at": now,
            "loop_seconds": LOOP_SECONDS,
            "offline_evolution_seconds": OFFLINE_EVOLUTION_SECONDS,
            "sync_seconds": SYNC_SECONDS,
            "before": {
                "training_records": before.get("training_records"),
                "pending_observations": before.get("pending_observations"),
                "worker_running": before.get("worker_running"),
                "champion_compatible": before.get("champion_compatible"),
            },
            "maintenance": maintenance,
            "offline_cycle": offline,
            "sync": sync,
            "worker_running": worker_running,
            "last_offline_cycle_at": self.last_offline_cycle_at or last_offline,
            "last_sync_at": self.last_sync_at or last_sync,
        }
        _write_json(STATUS, result)
        return result

    def run_forever(self) -> int:
        if not self.acquire():
            return 20
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        try:
            self.restore_once()
            while not self.stop_requested:
                try:
                    self.run_once()
                except Exception as exc:
                    _write_json(STATUS, {
                        "ok": False,
                        "pid": os.getpid(),
                        "checked_at": time.time(),
                        "error": repr(exc),
                    })
                deadline = time.monotonic() + LOOP_SECONDS
                while not self.stop_requested and time.monotonic() < deadline:
                    time.sleep(min(1.0, deadline - time.monotonic()))
        finally:
            try:
                if not lab_runtime.status().get("worker_running"):
                    state_sync.sync_to_git(force_heartbeat=True)
            except Exception:
                pass
            self.release()
        return 0


def main() -> int:
    return EvolutionDaemon().run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
