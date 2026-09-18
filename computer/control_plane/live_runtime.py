from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

from .live_telemetry import live_emit, live_flush, session_id, source_sha
from .screen_share import ScreenShare

ROOT = Path(os.environ.get("AIRI_ROOT") or os.environ.get("AIRIPC_WORKSPACE_ROOT") or ".").resolve()
STATE_PATH = ROOT / ".ai" / "state" / "airi_live_runtime.json"


def _write_state(status: str, *, tunnel_url: str = "", error: str = "") -> None:
    payload = {
        "pid": os.getpid(),
        "session_id": session_id(),
        "source_sha": source_sha(),
        "status": status,
        "tunnel_url": tunnel_url,
        "error": error[:240],
        "updated_at": int(time.time()),
    }
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        tmp.replace(STATE_PATH)
        STATE_PATH.chmod(0o600)
    except OSError:
        pass


def managed_runtime_active(max_age: int = 90) -> bool:
    try:
        row = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        pid = int(row.get("pid") or 0)
        updated = int(row.get("updated_at") or 0)
        if pid <= 0 or int(time.time()) - updated > max_age:
            return False
        if row.get("session_id") != session_id():
            return False
        os.kill(pid, 0)
        return row.get("status") in {"starting", "ready", "reconnecting"}
    except Exception:
        return False


def main() -> int:
    stopping = False

    def stop(*_args) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    _write_state("starting")
    live_emit(
        "runtime",
        "Airi-PC session online",
        session_id(),
        "running",
        dedupe_key="live-runtime:session-online",
    )
    live_flush(5.0)

    while not stopping:
        share = ScreenShare()
        try:
            _write_state("starting")
            share.start()
            _write_state("ready", tunnel_url=share.tunnel_url)
            while not stopping:
                tunnel_ok = share.tunnel is not None and share.tunnel.poll() is None
                server_ok = share.server_thread is not None and share.server_thread.is_alive()
                if not tunnel_ok or not server_ok:
                    raise RuntimeError("POV transport stopped")
                _write_state("ready", tunnel_url=share.tunnel_url)
                time.sleep(10)
        except Exception as exc:
            _write_state("reconnecting", error=str(exc))
            live_emit(
                "runtime",
                "POV reconnecting",
                type(exc).__name__,
                "retrying",
                dedupe_key="live-runtime:reconnecting",
            )
        finally:
            share.close()
        if not stopping:
            time.sleep(3)

    _write_state("stopped")
    live_emit(
        "runtime",
        "Airi-PC live session stopped",
        session_id(),
        "completed",
        dedupe_key="live-runtime:stopped",
    )
    live_flush(3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
