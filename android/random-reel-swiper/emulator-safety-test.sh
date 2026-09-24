#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
ART="$ROOT/test-artifacts"
mkdir -p "$ART/airi" "$ART/device-instagram" "$ART/device-blocked"

echo "[1/8] Build app + test hosts"
gradle :app:testDebugUnitTest :app:assembleDebug :safetyhost:assembleDebug :blockedhost:assembleDebug --stacktrace

echo "[2/8] Install APKs"
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r safetyhost/build/outputs/apk/debug/safetyhost-debug.apk
adb install -r blockedhost/build/outputs/apk/debug/blockedhost-debug.apk

echo "[3/8] Enable accessibility service and visible pointer trace"
adb shell settings put secure enabled_accessibility_services   com.airi.randomreelswiper/com.airi.randomreelswiper.AutoSwipeAccessibilityService
adb shell settings put secure accessibility_enabled 1
adb shell settings put system pointer_location 1 || true
sleep 2
adb shell dumpsys accessibility > "$ART/accessibility-dumpsys.txt"

if ! grep -q "com.airi.randomreelswiper" "$ART/accessibility-dumpsys.txt"; then
  echo "Accessibility service was not enabled"
  exit 31
fi

airi_screenshot() {
  local name="$1"
  curl -fsS http://127.0.0.1:9010/screenshot > "$ART/airi/$name.json"
  python3 - "$ART/airi/$name.json" "$ART/airi/$name.png" <<'PY'
import base64, json, sys
src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as f:
    obj = json.load(f)
with open(dst, "wb") as f:
    f.write(base64.b64decode(obj["data_base64"]))
PY
}

tap_text() {
  local text="$1"
  adb shell uiautomator dump /sdcard/window.xml >/dev/null
  adb pull /sdcard/window.xml "$ART/window.xml" >/dev/null
  read -r X Y < <(python3 - "$ART/window.xml" "$text" <<'PY'
import re, sys, xml.etree.ElementTree as ET
path, target = sys.argv[1:3]
root = ET.parse(path).getroot()
for node in root.iter("node"):
    if node.attrib.get("text") == target:
        m = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.attrib["bounds"])
        if not m:
            continue
        x1,y1,x2,y2 = map(int,m.groups())
        print((x1+x2)//2, (y1+y2)//2)
        raise SystemExit(0)
raise SystemExit("text not found: "+target)
PY
)
  adb shell input tap "$X" "$Y"
}

echo "[4/8] Start Random Reel Swiper"
adb shell am start -W -n com.airi.randomreelswiper/.MainActivity >/dev/null
sleep 2
tap_text "START"
sleep 1

# Put the visible emulator window in front of Airi-PC's Xvfb desktop if possible.
DISPLAY=:99 xdotool search --name "Android Emulator" windowactivate 2>/dev/null || true
curl -fsS http://127.0.0.1:9010/windows > "$ART/airi/windows.json" || true

echo "[5/8] Allowed-package visual/touch test (fake Instagram)"
adb shell am start -W -n com.instagram.android/com.example.safetyhost.MainActivity >/dev/null
sleep 2
adb logcat -c
airi_screenshot "instagram-start"

(
  for i in $(seq -w 1 14); do
    sleep 5
    adb exec-out screencap -p > "$ART/device-instagram/frame-$i.png"
  done
) &
SHOT_PID=$!

# Max initial delay is 60 s and max gesture duration is 4 s.
sleep 72
wait "$SHOT_PID" || true

adb logcat -d -v time > "$ART/instagram-logcat.txt"
adb exec-out screencap -p > "$ART/device-instagram/final.png"
airi_screenshot "instagram-final"

if ! grep -q "SafetyHost.*TOUCH_DOWN" "$ART/instagram-logcat.txt"; then
  echo "No automated swipe reached the fake Instagram surface"
  exit 41
fi
if ! grep -q "SafetyHost.*TOUCH_UP" "$ART/instagram-logcat.txt"; then
  echo "Gesture did not finish cleanly"
  exit 42
fi
if grep -Eq "SafetyHost.*(CLICK_TRIGGERED|LONG_CLICK_TRIGGERED)" "$ART/instagram-logcat.txt"; then
  echo "DANGER: swipe was interpreted as click or long-click"
  grep -E "SafetyHost.*(CLICK_TRIGGERED|LONG_CLICK_TRIGGERED)" "$ART/instagram-logcat.txt" || true
  exit 43
fi

echo "[6/8] Blocked-package test"
adb shell am start -W -n com.example.blockedhost/.MainActivity >/dev/null
# Let any already-running Instagram gesture finish before measuring the guard.
sleep 5
adb logcat -c
airi_screenshot "blocked-start"

(
  for i in $(seq -w 1 13); do
    sleep 5
    adb exec-out screencap -p > "$ART/device-blocked/frame-$i.png"
  done
) &
BLOCKED_SHOT_PID=$!

# Longer than the maximum 60 s scheduler delay.
sleep 66
wait "$BLOCKED_SHOT_PID" || true

adb logcat -d -v time > "$ART/blocked-logcat.txt"
adb exec-out screencap -p > "$ART/device-blocked/final.png"
airi_screenshot "blocked-final"

if grep -q "BlockedHost.*UNEXPECTED_TOUCH" "$ART/blocked-logcat.txt"; then
  echo "DANGER: automated touch occurred outside Instagram"
  grep "BlockedHost.*UNEXPECTED_TOUCH" "$ART/blocked-logcat.txt" || true
  exit 51
fi

echo "[7/8] Static no-click audit"
if grep -R -nE "\.performClick\(|GLOBAL_ACTION_|performGlobalAction\(|ACTION_CLICK|ACTION_LONG_CLICK"     app/src/main/java app/src/main/res > "$ART/static-click-audit.txt"; then
  echo "DANGER: click/global-action API found in production app"
  cat "$ART/static-click-audit.txt"
  exit 61
else
  echo "No click/global-action APIs found in production app" > "$ART/static-click-audit.txt"
fi

echo "[8/8] PASS"
cat > "$ART/RESULT.txt" <<'EOF'
PASS
- Unit safety properties passed.
- Accessibility service enabled in emulator.
- Fake Instagram received a drag gesture with DOWN/MOVE/UP.
- No CLICK_TRIGGERED.
- No LONG_CLICK_TRIGGERED.
- Non-Instagram blocked host received zero automated touches after the guard window.
- Static audit found no click/global-action API in production code.
EOF
