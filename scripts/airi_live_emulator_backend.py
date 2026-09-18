from __future__ import annotations

import os
import signal
import time

from control_plane.live_telemetry import live_emit, live_flush, session_id
from control_plane.screen_share import start_screen_share


def main() -> int:
    share = start_screen_share()
    if share is None:
        raise SystemExit("Airi POV backend did not start")
    live_emit(
        "runtime",
        "Android emulator POV test",
        f"session {session_id()}",
        "running",
        dedupe_key="android-emulator-pov:e2e",
    )
    live_flush(8.0)
    print("AIRI_ANDROID_EMULATOR_BACKEND=READY", flush=True)

    stopping = False

    def stop(*_args):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            time.sleep(0.5)
    finally:
        share.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
