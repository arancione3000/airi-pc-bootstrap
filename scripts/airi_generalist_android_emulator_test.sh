#!/usr/bin/env bash
set -euo pipefail

APK="$(find "$GITHUB_WORKSPACE/emulator-apk" -name '*.apk' -type f | head -n1)"
test -n "$APK"
test -f "$APK"

OUT="$RUNNER_TEMP/airi-generalist-android-smoke"
mkdir -p "$OUT"

dump_ui() {
  local remote="$1"
  local local_file="$2"
  adb shell uiautomator dump "$remote" >/dev/null 2>&1 || return 1
  adb pull "$remote" "$local_file" >/dev/null 2>&1 || return 1
}

center_for_text() {
  local xml="$1"
  local text="$2"
  python3 - "$xml" "$text" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
target = sys.argv[2]
for node in root.iter("node"):
    if node.attrib.get("text") != target:
        continue
    match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.attrib.get("bounds", ""))
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        print((x1 + x2) // 2, (y1 + y2) // 2)
        raise SystemExit(0)
raise SystemExit(f"text not found: {target}")
PY
}

center_for_edit_text() {
  local xml="$1"
  python3 - "$xml" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
for node in root.iter("node"):
    if not node.attrib.get("class", "").endswith("EditText"):
        continue
    match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.attrib.get("bounds", ""))
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        print((x1 + x2) // 2, (y1 + y2) // 2)
        raise SystemExit(0)
raise SystemExit("EditText not found")
PY
}

dismiss_system_anr() {
  local xml="$OUT/system-anr.xml"
  if dump_ui /sdcard/airi-system-anr.xml "$xml"; then
    if grep -Eq "isn't responding|non risponde" "$xml"; then
      if read WAIT_X WAIT_Y < <(center_for_text "$xml" "Wait" 2>/dev/null); then
        adb shell input tap "$WAIT_X" "$WAIT_Y" || true
      else
        adb shell input keyevent 4 || true
      fi
      sleep 1
      adb shell am start -n com.airipc.generalist/.MainActivity >/dev/null 2>&1 || true
      sleep 1
    fi
  fi
}

adb install -r "$APK"
adb shell settings put global hide_error_dialogs 1 || true
adb shell settings put global anr_show_background 0 || true
adb shell am start -W -n com.airipc.generalist/.MainActivity

PID="$(adb shell pidof com.airipc.generalist | tr -d '\r')"
test -n "$PID"

# Wait for the GitHub-backed model to be downloaded and loaded. Some Android
# runner images occasionally show a transient Quickstep ANR dialog; dismiss it
# and bring our activity back to foreground instead of treating launcher noise
# as an app failure.
READY=0
for attempt in $(seq 1 35); do
  adb shell am start -n com.airipc.generalist/.MainActivity >/dev/null 2>&1 || true
  sleep 2
  if dump_ui /sdcard/airi-home.xml "$OUT/home.xml"; then
    if grep -Eq 'AIRI Live pronto|AIRI Live sincronizzato|AI aggiornata · modello mobile in sincronizzazione|cache offline' "$OUT/home.xml"; then
      READY=1
      break
    fi
    if grep -Eq "isn't responding|non risponde" "$OUT/home.xml"; then
      if read WAIT_X WAIT_Y < <(center_for_text "$OUT/home.xml" "Wait" 2>/dev/null); then
        adb shell input tap "$WAIT_X" "$WAIT_Y" || true
      else
        adb shell input keyevent 4 || true
      fi
      adb shell am start -n com.airipc.generalist/.MainActivity >/dev/null 2>&1 || true
    fi
  fi
done
test "$READY" -eq 1

adb exec-out screencap -p > "$OUT/home.png"
test -s "$OUT/home.png"

# Reproduce the user's manual refresh path. The tap must start a real
# commit-pinned synchronization and emit PASS after the click, not merely
# return immediately with a cached branch-name response.
adb logcat -c
dump_ui /sdcard/airi-refresh.xml "$OUT/refresh-before.xml"
read REFRESH_X REFRESH_Y < <(center_for_text "$OUT/refresh-before.xml" "Aggiorna")
adb shell input tap "$REFRESH_X" "$REFRESH_Y"

REFRESH_PASS=0
for _ in $(seq 1 45); do
  if adb logcat -d | grep -q 'AIRI_GENERALIST_REFRESH=PASS'; then
    REFRESH_PASS=1
    break
  fi
  sleep 1
done
test "$REFRESH_PASS" -eq 1
adb logcat -d > "$OUT/refresh-logcat.txt"
grep -q 'AIRI_GENERALIST_REFRESH=PASS' "$OUT/refresh-logcat.txt"

for _ in $(seq 1 20); do
  dump_ui /sdcard/airi-refresh-after.xml "$OUT/refresh-after.xml" || true
  if grep -q 'text="Aggiorna"' "$OUT/refresh-after.xml" 2>/dev/null; then
    break
  fi
  sleep 1
done
grep -q 'text="Aggiorna"' "$OUT/refresh-after.xml"

# Real on-device inference.
adb logcat -c
read INPUT_X INPUT_Y < <(center_for_edit_text "$OUT/home.xml")
adb shell input tap "$INPUT_X" "$INPUT_Y"
adb shell input text ciao
sleep 1
dismiss_system_anr

dump_ui /sdcard/airi-chat.xml "$OUT/chat-before.xml"
read SEND_X SEND_Y < <(center_for_text "$OUT/chat-before.xml" "Invia")
adb shell input tap "$SEND_X" "$SEND_Y"

for _ in $(seq 1 60); do
  if adb logcat -d | grep -q 'AIRI_GENERALIST_INFERENCE=PASS'; then
    break
  fi
  sleep 1
done
adb logcat -d > "$OUT/logcat.txt"
grep -q 'AIRI_GENERALIST_INFERENCE=PASS' "$OUT/logcat.txt"
adb exec-out screencap -p > "$OUT/chat-after.png"
test -s "$OUT/chat-after.png"

# A distinct research snapshot is optional. Under the unified-lineage policy,
# the mobile exporter may publish "research" as a byte-identical alias of AIRI
# Live. The UI intentionally hides that duplicate instead of pretending there
# are two models. If a genuinely distinct research snapshot exists, smoke-test
# it separately.
dismiss_system_anr
dump_ui /sdcard/airi-research-select.xml "$OUT/research-select.xml"
RESEARCH_INFERENCE="ALIAS_SKIPPED"
if grep -q 'text="Research Snapshot"' "$OUT/research-select.xml"; then
  read RESEARCH_X RESEARCH_Y < <(center_for_text "$OUT/research-select.xml" "Research Snapshot")
  adb shell input tap "$RESEARCH_X" "$RESEARCH_Y"

  RESEARCH_READY=0
  for _ in $(seq 1 45); do
    sleep 1
    dismiss_system_anr
    dump_ui /sdcard/airi-research-ready.xml "$OUT/research-ready.xml" || true
    if grep -Eq 'Research Snapshot sincronizzato|Research Snapshot pronto|AI aggiornata · modello mobile in sincronizzazione|cache offline' "$OUT/research-ready.xml" 2>/dev/null; then
      RESEARCH_READY=1
      break
    fi
  done
  test "$RESEARCH_READY" -eq 1

  adb logcat -c
  read RESEARCH_INPUT_X RESEARCH_INPUT_Y < <(center_for_edit_text "$OUT/research-ready.xml")
  adb shell input tap "$RESEARCH_INPUT_X" "$RESEARCH_INPUT_Y"
  adb shell input text hello
  sleep 1
  dismiss_system_anr

  dump_ui /sdcard/airi-research-chat.xml "$OUT/research-chat-before.xml"
  read RESEARCH_SEND_X RESEARCH_SEND_Y < <(center_for_text "$OUT/research-chat-before.xml" "Invia")
  adb shell input tap "$RESEARCH_SEND_X" "$RESEARCH_SEND_Y"

  RESEARCH_INFERENCE=0
  for _ in $(seq 1 60); do
    if adb logcat -d | grep -q 'AIRI_GENERALIST_INFERENCE=PASS'; then
      RESEARCH_INFERENCE=1
      break
    fi
    sleep 1
  done
  test "$RESEARCH_INFERENCE" -eq 1
  adb logcat -d > "$OUT/research-logcat.txt"
  grep -q 'AIRI_GENERALIST_INFERENCE=PASS' "$OUT/research-logcat.txt"
  adb exec-out screencap -p > "$OUT/research-chat-after.png"
  test -s "$OUT/research-chat-after.png"
else
  grep -q 'text="AIRI Live"' "$OUT/research-select.xml"
fi

# Verify all requested observability screens.
for TAB in Live Neural Airi-PC; do
  dismiss_system_anr
  dump_ui "/sdcard/airi-$TAB.xml" "$OUT/tab.xml"
  read TAB_X TAB_Y < <(center_for_text "$OUT/tab.xml" "$TAB")
  adb shell input tap "$TAB_X" "$TAB_Y"
  sleep 2
  adb exec-out screencap -p > "$OUT/tab-$TAB.png"
  test -s "$OUT/tab-$TAB.png"
done

echo "AIRI_GENERALIST_ANDROID_SMOKE=PASS pid=$PID apk=$APK refresh=PASS airi_live_inference=PASS research_inference=$RESEARCH_INFERENCE observability=PASS"
