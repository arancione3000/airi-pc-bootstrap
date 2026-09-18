#!/usr/bin/env bash
set -euo pipefail

cd "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
SHOT_DIR="${RUNNER_TEMP:?RUNNER_TEMP is required}/airi-live-pov"
mkdir -p "$SHOT_DIR"

APK="$(find "$GITHUB_WORKSPACE/emulator-apk" -name '*.apk' -type f | head -n1)"
test -n "$APK"
test -n "${AIRI_E2E_RELAY_URL:-}"
test -n "${AIRI_E2E_TOPIC:-}"

adb install -r "$APK"

# Launcher/Quickstep is irrelevant to this test and can ANR on headless emulators.
adb shell am force-stop com.android.launcher3 >/dev/null 2>&1 || true
adb shell am force-stop com.google.android.apps.nexuslauncher >/dev/null 2>&1 || true
adb shell am force-stop com.airipc.live
adb logcat -c || true

adb shell am start -S -W -n com.airipc.live/.MainActivity \
  --es airi_relay "$AIRI_E2E_RELAY_URL" \
  --es airi_topic "$AIRI_E2E_TOPIC"

python3 - <<'PY'
import json
import os
import time
import urllib.request

relay = os.environ["AIRI_E2E_RELAY_URL"].rstrip("/")
topic = os.environ["AIRI_E2E_TOPIC"]
deadline = time.monotonic() + 35
found = False

while time.monotonic() < deadline:
    try:
        req = urllib.request.Request(
            f"{relay}/{topic}/json?poll=1&since=10m",
            headers={"User-Agent": "Airi-Live-Android-E2E/1.5"},
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            lines = response.read().decode("utf-8", errors="replace").splitlines()
        for line in lines:
            try:
                outer = json.loads(line)
                payload = json.loads(outer.get("message") or "{}")
            except Exception:
                continue
            if payload.get("kind") == "viewer_key" and payload.get("key_id"):
                found = True
                break
    except Exception:
        pass
    if found:
        break
    time.sleep(1)

if not found:
    raise SystemExit("Android APK never published viewer_key to isolated relay")
print("AIRI_ANDROID_CONFIG_OVERRIDE=PASS")
PY

# Wait for the real cryptographic rendezvous instead of repeatedly invoking
# uiautomator, which can hang when the launcher is unhealthy.
offer_ok=0
for i in $(seq 1 75); do
  sleep 2
  if adb logcat -d -t 600 | grep -q 'AiriLivePOV.*screen offer decrypted'; then
    offer_ok=1
    break
  fi
  # If a stale system ANR dialog appears, tapping this harmless lower-middle
  # coordinate selects "Wait" on the fixed Pixel 6 CI profile. In Airi Live
  # itself the same coordinate has no destructive action.
  if [ $((i % 5)) -eq 0 ]; then
    adb shell input tap 240 1030 >/dev/null 2>&1 || true
  fi
done

if [ "$offer_ok" -ne 1 ]; then
  adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-no-offer.png" || true
  adb logcat -d -t 1600 | grep -E 'AiriLivePOV|com.airipc.live|AndroidRuntime' > "$SHOT_DIR/airi-android-logcat.txt" || true
  echo "Encrypted POV screen offer was never decrypted by the Android app" >&2
  exit 1
fi
echo "AIRI_ANDROID_POV_OFFER_SMOKE=PASS"

# Wait for actual remote Chromium pixels, not merely the POV card shell.
pixels_ok=0
for i in $(seq 1 50); do
  adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-card-1.png"
  if python3 - <<'PY'
from PIL import Image
from pathlib import Path
import os
img = Image.open(Path(os.environ["RUNNER_TEMP"]) / "airi-live-pov" / "airi-live-emulator-card-1.png").convert("RGB")
cyan = sum(1 for r,g,b in img.getdata() if r < 100 and g > 150 and b > 150)
print("remote_cyan_pixels=", cyan)
raise SystemExit(0 if cyan > 800 else 1)
PY
  then
    pixels_ok=1
    break
  fi
  sleep 2
done
if [ "$pixels_ok" -ne 1 ]; then
  adb logcat -d -t 1600 | grep -E 'AiriLivePOV|com.airipc.live|AndroidRuntime' > "$SHOT_DIR/airi-android-logcat.txt" || true
  echo "POV offer decrypted but remote Chromium pixels never rendered" >&2
  exit 1
fi
echo "AIRI_ANDROID_POV_PIXEL_SMOKE=PASS"

# Require the continuous stream to beat the old ~4-5 FPS polling path.
fps_ok=0
for i in $(seq 1 20); do
  sleep 1
  line="$(adb logcat -d -t 1200 | grep 'AiriLivePOV.*POV_STREAM_FPS=' | tail -n1 || true)"
  if [ -n "$line" ]; then
    FPS="$(printf '%s\n' "$line" | sed -n 's/.*POV_STREAM_FPS=\([0-9.]*\).*/\1/p')"
    MODE="$(printf '%s\n' "$line" | sed -n 's/.*mode=\([^ ]*\).*/\1/p')"
    if python3 - "$FPS" "$MODE" <<'PY'
import sys
fps=float(sys.argv[1] or 0)
mode=sys.argv[2]
print("measured_stream_fps=", fps, "mode=", mode)
raise SystemExit(0 if mode == "stream" and fps >= 7.0 else 1)
PY
    then
      fps_ok=1
      break
    fi
  fi
done
if [ "$fps_ok" -ne 1 ]; then
  adb logcat -d -t 1600 | grep -E 'AiriLivePOV|com.airipc.live|AndroidRuntime' > "$SHOT_DIR/airi-android-logcat.txt" || true
  echo "Continuous POV stream did not reach 7 FPS" >&2
  exit 1
fi
echo "AIRI_ANDROID_POV_FPS_SMOKE=PASS"

# Prove the displayed remote browser is moving/updating.
sleep 1.6
adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-card-2.png"
python3 - <<'PY'
from PIL import Image, ImageChops
from pathlib import Path
import os

base = Path(os.environ["RUNNER_TEMP"]) / "airi-live-pov"
a = Image.open(base / "airi-live-emulator-card-1.png").convert("RGB")
b = Image.open(base / "airi-live-emulator-card-2.png").convert("RGB")
points=[]
for y in range(b.height):
    for x in range(b.width):
        r,g,bl=b.getpixel((x,y))
        if r < 100 and g > 150 and bl > 150:
            points.append((x,y))
assert len(points) > 800, "remote browser disappeared before motion test"
x1=min(x for x,_ in points); y1=min(y for _,y in points)
x2=max(x for x,_ in points)+1; y2=max(y for _,y in points)+1
diff=ImageChops.difference(a.crop((x1,y1,x2,y2)), b.crop((x1,y1,x2,y2)))
changed=sum(1 for px in diff.getdata() if max(px) > 18)
print("remote_motion_pixels=", changed)
assert changed > 80, "remote POV rendered but did not visibly update"
print("AIRI_ANDROID_POV_MOTION_SMOKE=PASS")
PY

# One accessibility lookup after the UI is known-good, then verify fullscreen.
adb shell uiautomator dump /sdcard/airi.xml >/dev/null 2>&1
adb pull /sdcard/airi.xml "$SHOT_DIR/airi-ui.xml" >/dev/null 2>&1
python3 - <<'PY' > /tmp/airi-tap.txt
import re, os
from pathlib import Path
text=(Path(os.environ["RUNNER_TEMP"]) / "airi-live-pov" / "airi-ui.xml").read_text(encoding="utf-8")
m=re.search(r'content-desc="Airi POV fullscreen"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', text)
if not m:
    raise SystemExit("fullscreen accessibility target not found")
x1,y1,x2,y2=map(int,m.groups())
print((x1+x2)//2,(y1+y2)//2)
PY
read -r X Y < /tmp/airi-tap.txt
adb shell input tap "$X" "$Y"
sleep 2.5
adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-fullscreen.png"

python3 - <<'PY'
from PIL import Image
from pathlib import Path
import os
img=Image.open(Path(os.environ["RUNNER_TEMP"]) / "airi-live-pov" / "airi-live-emulator-fullscreen.png").convert("RGB")
cyan=sum(1 for r,g,b in img.getdata() if r < 100 and g > 150 and b > 150)
print("fullscreen_remote_cyan_pixels=",cyan)
assert cyan > 4000, "full-screen POV did not contain the remote Chromium desktop"
print("AIRI_ANDROID_POV_FULLSCREEN_SMOKE=PASS")
PY

adb logcat -d -t 1200 | grep -E 'AiriLivePOV|com.airipc.live|AndroidRuntime' > "$SHOT_DIR/airi-android-logcat.txt" || true
echo "AIRI_ANDROID_POV_E2E=PASS"
