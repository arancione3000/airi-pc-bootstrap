from __future__ import annotations

import os
import signal
import time

from control_plane.live_telemetry import live_emit, live_flush, session_id
from control_plane.screen_share import _capture_jpeg, start_screen_share
from server import browser


def main() -> int:
    os.environ["AIRI_BROWSER_HEADLESS"] = "0"

    def prepare_browser():
        page = browser()._ensure()
        page.set_viewport_size({"width": 1180, "height": 680})
        page.set_content("""
<!doctype html><html><head><meta charset="utf-8"><title>Airi POV E2E Browser</title>
<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#00d8ff;font-family:Arial,sans-serif}
#title{position:absolute;left:44px;top:38px;font-size:48px;font-weight:900;color:#101014}
#sub{position:absolute;left:48px;top:108px;font-size:26px;font-weight:700;color:#101014}
#runner{position:absolute;width:220px;height:220px;border-radius:42px;background:#ff8a00;top:250px;left:30px;
box-shadow:0 0 0 12px #101014 inset}
#clock{position:absolute;left:48px;bottom:44px;font-size:34px;font-weight:800;color:#101014}
</style></head><body>
<div id="title">AIRI-PC POV E2E</div><div id="sub">REAL HEADED CHROMIUM → ANDROID</div>
<div id="runner"></div><div id="clock"></div>
<script>
const box=document.getElementById('runner'),clock=document.getElementById('clock');
function tick(){const t=performance.now()/1000;box.style.left=(40+Math.abs(Math.sin(t*1.7))*850)+'px';
clock.textContent=new Date().toISOString();requestAnimationFrame(tick)}tick();
</script></body></html>
""", wait_until="load")
        page.bring_to_front()
        return page.title()

    title = browser().call(prepare_browser)
    print(f"AIRI_TEST_BROWSER_READY={title}", flush=True)
    time.sleep(1.5)

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
    if cyan < 20_000:
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
