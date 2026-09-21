#!/usr/bin/env bash
set -euo pipefail

APK="$(find "$GITHUB_WORKSPACE/emulator-apk" -name '*.apk' -type f | head -n1)"
test -n "$APK"
test -f "$APK"

adb install -r "$APK"
adb shell am start -W -n com.airipc.generalist/.MainActivity
sleep 8

PID="$(adb shell pidof com.airipc.generalist | tr -d '\r')"
test -n "$PID"

OUT="$RUNNER_TEMP/airi-generalist-android-smoke"
mkdir -p "$OUT"
adb exec-out screencap -p > "$OUT/home.png"
test -s "$OUT/home.png"

adb shell uiautomator dump /sdcard/airi-home.xml >/dev/null
adb pull /sdcard/airi-home.xml "$OUT/home.xml" >/dev/null
grep -q 'Champion pronto' "$OUT/home.xml"

adb logcat -c
adb shell input tap 320 2180
adb shell input text ciao
sleep 1
adb shell uiautomator dump /sdcard/airi-chat.xml >/dev/null
adb pull /sdcard/airi-chat.xml "$OUT/chat-before.xml" >/dev/null

read SEND_X SEND_Y < <(python3 - "$OUT/chat-before.xml" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
for node in root.iter("node"):
    if node.attrib.get("text") != "Invia":
        continue
    match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.attrib.get("bounds", ""))
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        print((x1 + x2) // 2, (y1 + y2) // 2)
        raise SystemExit(0)
raise SystemExit("Invia button not found in UI hierarchy")
PY
)
adb shell input tap "$SEND_X" "$SEND_Y"

for _ in $(seq 1 45); do
  if adb logcat -d | grep -q 'AIRI_GENERALIST_INFERENCE=PASS'; then
    break
  fi
  sleep 1
done
adb logcat -d > "$OUT/logcat.txt"
grep -q 'AIRI_GENERALIST_INFERENCE=PASS' "$OUT/logcat.txt"

adb exec-out screencap -p > "$OUT/chat-after.png"
test -s "$OUT/chat-after.png"

echo "AIRI_GENERALIST_ANDROID_SMOKE=PASS pid=$PID apk=$APK inference=PASS"
