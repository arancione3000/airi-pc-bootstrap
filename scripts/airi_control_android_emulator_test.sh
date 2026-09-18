#!/usr/bin/env bash
set -euo pipefail

PKG="com.airipc.control"
ACTIVITY="$PKG/.MainActivity"
ROOT="${GITHUB_WORKSPACE:?}"
OUT="${RUNNER_TEMP:-/tmp}/airi-control-e2e"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/computer"
export AIRI_CONTROL_RELAY_BASE="${AIRI_CONTROL_RELAY_BASE:-https://ntfy.sh}"
export AIRI_CONTROL_TOPIC="${AIRI_CONTROL_TOPIC:?}"

APK="$(find "$ROOT/airi-control-android/app/build/outputs/apk/debug" -name '*.apk' -type f | head -n1)"
test -n "$APK"
adb install -r "$APK"

if adb shell pm list permissions -g 2>/dev/null | grep -q 'android.permission.POST_NOTIFICATIONS'; then
  adb shell pm grant "$PKG" android.permission.POST_NOTIFICATIONS 2>/dev/null || true
fi

dump_ui() {
  adb shell uiautomator dump /sdcard/airi-control-window.xml >/dev/null 2>&1 || true
  adb exec-out cat /sdcard/airi-control-window.xml > "$OUT/window.xml"
}

coords_for() {
  local pattern="$1"
  dump_ui
  python3 - "$OUT/window.xml" "$pattern" <<'PY'
import re, sys, xml.etree.ElementTree as ET
path, pattern = sys.argv[1], sys.argv[2]
root = ET.parse(path).getroot()
rx = re.compile(pattern, re.I)
for node in root.iter("node"):
    text = (node.attrib.get("text","") + " " + node.attrib.get("content-desc","")).strip()
    if not rx.search(text):
        continue
    b = node.attrib.get("bounds","")
    m = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", b)
    if not m:
        continue
    x1,y1,x2,y2 = map(int,m.groups())
    if x2 > x1 and y2 > y1:
        print((x1+x2)//2, (y1+y2)//2)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

tap_matching() {
  local pattern="$1"
  local tries="${2:-20}"
  local xy=""
  for _ in $(seq 1 "$tries"); do
    if xy="$(coords_for "$pattern" 2>/dev/null)"; then
      adb shell input tap ${xy}
      return 0
    fi
    sleep 0.5
  done
  dump_ui
  echo "Could not find UI node: $pattern" >&2
  cat "$OUT/window.xml" >&2
  return 1
}

diagnostics() {
  local rc="$1"
  echo "AIRI_CONTROL_E2E_DIAGNOSTIC rc=$rc" >&2
  adb shell settings get secure enabled_accessibility_services > "$OUT/enabled-accessibility.txt" 2>&1 || true
  adb shell dumpsys accessibility > "$OUT/accessibility-failure.txt" 2>&1 || true
  adb logcat -d -t 1200 > "$OUT/logcat.txt" 2>&1 || true
  dump_ui || true
  adb exec-out screencap -p > "$OUT/failure-screen.png" 2>/dev/null || true
}
trap 'rc=$?; diagnostics "$rc"; exit "$rc"' ERR

adb shell settings put secure enabled_accessibility_services "$PKG/$PKG.AiriAccessibilityService"
adb shell settings put secure accessibility_enabled 1
adb shell am start -n "$ACTIVITY"
sleep 1

bound=0
for attempt in $(seq 1 40); do
  enabled="$(adb shell settings get secure enabled_accessibility_services | tr -d '\r')"
  adb shell dumpsys accessibility > "$OUT/accessibility-before.txt" 2>&1 || true
  if echo "$enabled" | grep -q "$PKG" && grep -q 'AiriAccessibilityService' "$OUT/accessibility-before.txt"; then
    bound=1
    break
  fi
  if [ $((attempt % 4)) -eq 0 ]; then
    adb shell settings put secure enabled_accessibility_services "$PKG/$PKG.AiriAccessibilityService"
    adb shell settings put secure accessibility_enabled 1
    adb shell am start -n "$ACTIVITY" >/dev/null 2>&1 || true
  fi
  sleep 0.5
done
test "$bound" -eq 1
echo "AIRI_CONTROL_ACCESSIBILITY_BOUND=PASS"

adb shell input keyevent KEYCODE_HOME
sleep 0.4
adb shell am start -n "$ACTIVITY" >/dev/null
for _ in $(seq 1 20); do
  if coords_for 'Abilitata' >/dev/null 2>&1; then
    break
  fi
  sleep 0.4
done
coords_for 'Abilitata' >/dev/null
echo "AIRI_CONTROL_ACCESSIBILITY_UI=PASS"

rm -f "$ROOT/.ai/state/phone_control_controller.json"
tap_matching '^AVVIA CONTROLLO$'
for _ in $(seq 1 20); do
  if coords_for 'In attesa di Airi-PC' >/dev/null 2>&1; then
    break
  fi
  sleep 0.4
done
coords_for 'In attesa di Airi-PC' >/dev/null
echo "AIRI_CONTROL_ONE_TAP_START=PASS"

CTL=(python3 "$ROOT/computer/control_plane/phone_control.py" --timeout 30)

"${CTL[@]}" status > "$OUT/status.json"
python3 - "$OUT/status.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
assert x.get("ok") is True, x
assert x.get("active") is True, x
assert x.get("controller_connected") is True, x
assert x.get("accessibility") is True, x
assert x.get("screen_capture") is True, x
print("AIRI_CONTROL_AUTO_PAIR=PASS")
print("AIRI_CONTROL_REMOTE_STATUS=PASS")
PY

"${CTL[@]}" screenshot "$OUT/phone.jpg" > "$OUT/screenshot.json"
python3 - "$OUT/phone.jpg" <<'PY'
import pathlib, sys
p=pathlib.Path(sys.argv[1])
data=p.read_bytes()
assert len(data) > 5000, len(data)
assert data[:2] == b'\xff\xd8', data[:8]
print("AIRI_CONTROL_ACCESSIBILITY_SCREENSHOT=PASS", len(data))
PY

TARGET="$(coords_for '^Area test controllo$')"
TX="$(echo "$TARGET" | awk '{print $1}')"
TY="$(echo "$TARGET" | awk '{print $2}')"
"${CTL[@]}" tap "$TX" "$TY" > "$OUT/tap.json"
sleep 1
"${CTL[@]}" observe > "$OUT/observe-tap.json"
grep -q 'TOCCO RICEVUTO' "$OUT/observe-tap.json"
echo "AIRI_CONTROL_REMOTE_TAP=PASS"

FIELD="$(coords_for 'Campo test testo')"
FX="$(echo "$FIELD" | awk '{print $1}')"
FY="$(echo "$FIELD" | awk '{print $2}')"
"${CTL[@]}" tap "$FX" "$FY" > "$OUT/focus.json"
sleep 0.5
"${CTL[@]}" text 'AIRI_PHONE_OK' > "$OUT/text.json"
sleep 0.5
"${CTL[@]}" observe > "$OUT/observe-text.json"
grep -q 'AIRI_PHONE_OK' "$OUT/observe-text.json"
echo "AIRI_CONTROL_REMOTE_TEXT=PASS"

adb shell input keyevent KEYCODE_BACK || true
sleep 0.5
tap_matching '^STOP IMMEDIATO$'

for _ in $(seq 1 20); do
  if coords_for 'Controllo disattivato' >/dev/null 2>&1; then
    break
  fi
  sleep 0.4
done
coords_for 'Controllo disattivato' >/dev/null

enabled="$(adb shell settings get secure enabled_accessibility_services | tr -d '\r')"
echo "$enabled" | grep -q "$PKG"
adb shell dumpsys accessibility > "$OUT/accessibility-after-stop.txt"
grep -q 'AiriAccessibilityService' "$OUT/accessibility-after-stop.txt"
echo "AIRI_CONTROL_STOP_KEEPS_PERMISSION=PASS"

if "${CTL[@]}" --timeout 5 status > "$OUT/status-after-stop.json" 2>&1; then
  echo "Remote control unexpectedly answered after local STOP" >&2
  exit 1
fi

adb exec-out screencap -p > "$OUT/final-screen.png" || true
echo "AIRI_CONTROL_LOCAL_STOP=PASS"
echo "AIRI_CONTROL_ANDROID_E2E=PASS"
