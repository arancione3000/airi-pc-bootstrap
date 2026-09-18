from __future__ import annotations

import os
import signal
import time

from control_plane.live_telemetry import live_emit, live_flush, session_id
from control_plane.screen_share import _capture_jpeg, start_screen_share


def main() -> int:
    share = start_screen_share()
    if share is None:
        raise SystemExit("Airi POV backend did not start")

    # Save and validate the exact framebuffer bytes used by the stream.
    from io import BytesIO
    from pathlib import Path
    from PIL import Image
    out_dir = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "airi-live-pov"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _capture_jpeg()
    (out_dir / "airi-backend-frame.jpg").write_bytes(frame)
    img = Image.open(BytesIO(frame)).convert("RGB")
    cyan = sum(1 for r, g, b in img.getdata() if r < 100 and g > 150 and b > 150)
    print(f"AIRI_BACKEND_CYAN_PIXELS={cyan}", flush=True)
    if cyan < 800:
        raise SystemExit("Airi POV framebuffer capture is not showing the graphical desktop")
    print("AIRI_BACKEND_FRAMEBUFFER_SMOKE=PASS", flush=True)
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
