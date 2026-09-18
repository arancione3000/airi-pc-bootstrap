#!/usr/bin/env bash
set -euo pipefail

cd "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"

APK="$(find "$GITHUB_WORKSPACE/emulator-apk" -name '*.apk' -type f | head -n1)"
test -n "$APK"

adb install -r "$APK"
adb shell am force-stop com.airipc.live
adb shell am start -n com.airipc.live/.MainActivity

found=0
for i in $(seq 1 30); do
  sleep 3
  adb shell uiautomator dump /sdcard/airi.xml >/dev/null 2>&1 || true
  adb pull /sdcard/airi.xml airi-ui.xml >/dev/null 2>&1 || true
  if grep -q 'POV Airi-PC' airi-ui.xml 2>/dev/null; then
    found=1
    break
  fi
done
if [ "$found" -ne 1 ]; then
  adb exec-out screencap -p > airi-live-emulator-no-pov.png || true
  adb shell dumpsys activity activities | tail -n 80 || true
  adb logcat -d -t 300 | grep -E 'com.airipc.live|chromium|WebView|AndroidRuntime' || true
  echo "POV card never appeared" >&2
  exit 1
fi

pixels_ok=0
for i in $(seq 1 25); do
  adb exec-out screencap -p > airi-live-emulator-card.png
  if python3 - <<'PY'
from PIL import Image
img = Image.open("airi-live-emulator-card.png").convert("RGB")
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
  adb logcat -d -t 400 | grep -E 'com.airipc.live|chromium|WebView|AndroidRuntime' || true
  echo "POV card appeared but remote pixels never rendered" >&2
  exit 1
fi
echo "AIRI_ANDROID_POV_PIXEL_SMOKE=PASS"

adb shell uiautomator dump /sdcard/airi.xml >/dev/null 2>&1
adb pull /sdcard/airi.xml airi-ui.xml >/dev/null 2>&1
python3 - <<'PY' > /tmp/airi-tap.txt
import re
text=open("airi-ui.xml",encoding="utf-8").read()
m=re.search(r'text="SCHERMO INTERO"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',text)
if not m:
    raise SystemExit("full-screen button not found")
x1,y1,x2,y2=map(int,m.groups())
print((x1+x2)//2, (y1+y2)//2)
PY
read -r X Y < /tmp/airi-tap.txt
adb shell input tap "$X" "$Y"
sleep 4
adb exec-out screencap -p > airi-live-emulator-fullscreen.png

python3 - <<'PY'
from PIL import Image
img = Image.open("airi-live-emulator-fullscreen.png").convert("RGB")
cyan = sum(1 for r,g,b in img.getdata() if r < 100 and g > 150 and b > 150)
print("fullscreen_remote_cyan_pixels=", cyan)
assert cyan > 4000, "full-screen POV did not contain the remote desktop"
print("AIRI_ANDROID_POV_FULLSCREEN_SMOKE=PASS")
PY
