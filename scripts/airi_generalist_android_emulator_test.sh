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

echo "AIRI_GENERALIST_ANDROID_SMOKE=PASS pid=$PID apk=$APK"
