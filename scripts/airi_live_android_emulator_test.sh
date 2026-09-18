#!/usr/bin/env bash
set -euo pipefail

cd "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
SHOT_DIR="${RUNNER_TEMP:?RUNNER_TEMP is required}/airi-live-pov"
mkdir -p "$SHOT_DIR"

APK="$(find "$GITHUB_WORKSPACE/emulator-apk" -name '*.apk' -type f | head -n1)"
test -n "$APK"

adb install -r "$APK"
adb shell am force-stop com.airipc.live
adb shell am start -n com.airipc.live/.MainActivity

found=0
for i in $(seq 1 30); do
  sleep 3
  adb shell uiautomator dump /sdcard/airi.xml >/dev/null 2>&1 || true
  adb pull /sdcard/airi.xml "$SHOT_DIR/airi-ui.xml" >/dev/null 2>&1 || true
  if grep -q 'POV Airi-PC' "$SHOT_DIR/airi-ui.xml" 2>/dev/null; then
    found=1
    break
  fi
done
if [ "$found" -ne 1 ]; then
  adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-no-pov.png" || true
  adb shell dumpsys activity activities | tail -n 80 || true
  adb logcat -d -t 300 | grep -E 'com.airipc.live|chromium|WebView|AndroidRuntime' > "$SHOT_DIR/airi-android-logcat.txt" || true
  echo "POV card never appeared" >&2
  exit 1
fi

pixels_ok=0
for i in $(seq 1 25); do
  adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-card-1.png"
  if python3 - <<'PY'
from PIL import Image
from pathlib import Path
img = Image.open(Path(__import__("os").environ["RUNNER_TEMP"]) / "airi-live-pov" / "airi-live-emulator-card-1.png").convert("RGB")
cyan = sum(1 for r,g,b in img.getdata() if r < 100 and g > 150 and b > 150)
print("remote_cyan_pixels=", cyan)
raise SystemExit(0 if cyan > 800 else 1)
PY
  then
    pixels_ok=1
    break
  fi
  sleep 3
done
if [ "$pixels_ok" -ne 1 ]; then
  adb logcat -d -t 400 | grep -E 'com.airipc.live|chromium|WebView|AndroidRuntime' > "$SHOT_DIR/airi-android-logcat.txt" || true
  echo "POV card appeared but remote pixels never rendered" >&2
  exit 1
fi
echo "AIRI_ANDROID_POV_PIXEL_SMOKE=PASS"

sleep 1.4
adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-card-2.png"
python3 - <<'PY'
from PIL import Image, ImageChops
from pathlib import Path
import os

base = Path(os.environ["RUNNER_TEMP"]) / "airi-live-pov"
a = Image.open(base / "airi-live-emulator-card-1.png").convert("RGB")
b = Image.open(base / "airi-live-emulator-card-2.png").convert("RGB")

points = []
for y in range(b.height):
    for x in range(b.width):
        r,g,bl = b.getpixel((x,y))
        if r < 100 and g > 150 and bl > 150:
            points.append((x,y))
assert len(points) > 800, "remote cyan desktop disappeared before motion test"
x1=min(x for x,_ in points); y1=min(y for _,y in points)
x2=max(x for x,_ in points)+1; y2=max(y for _,y in points)+1
crop_a=a.crop((x1,y1,x2,y2))
crop_b=b.crop((x1,y1,x2,y2))
diff=ImageChops.difference(crop_a,crop_b)
changed=sum(1 for px in diff.getdata() if max(px) > 18)
print("remote_motion_pixels=", changed)
assert changed > 80, "remote POV rendered a frame but did not visibly update"
print("AIRI_ANDROID_POV_MOTION_SMOKE=PASS")
PY

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
print((x1+x2)//2, (y1+y2)//2)
PY
read -r X Y < /tmp/airi-tap.txt
adb shell input tap "$X" "$Y"
sleep 2
adb exec-out screencap -p > "$SHOT_DIR/airi-live-emulator-fullscreen.png"

python3 - <<'PY'
from PIL import Image
from pathlib import Path
import os
img = Image.open(Path(os.environ["RUNNER_TEMP"]) / "airi-live-pov" / "airi-live-emulator-fullscreen.png").convert("RGB")
cyan = sum(1 for r,g,b in img.getdata() if r < 100 and g > 150 and b > 150)
print("fullscreen_remote_cyan_pixels=", cyan)
assert cyan > 4000, "full-screen POV did not contain the remote desktop"
print("AIRI_ANDROID_POV_FULLSCREEN_SMOKE=PASS")
PY
