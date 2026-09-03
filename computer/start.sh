#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT="${AIRIPC_WORKSPACE_ROOT:-${AIRI_WORKSPACE_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}}"
ROOT=$(CDPATH= cd -- "$ROOT" && pwd)
VENV="${AIRIPC_VENV:-$ROOT/.venv}"
LOG_DIR="${AIRI_LOG_DIR:-$ROOT/logs}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT/.cache/ms-playwright}"
mkdir -p "$LOG_DIR" "$PLAYWRIGHT_BROWSERS_PATH"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
case "$DISPLAY_NUM" in
  ''|*[!0-9]*) echo 'AIRI_START_INVALID_DISPLAY_NUM' >&2; exit 2 ;;
esac
export DISPLAY=":$DISPLAY_NUM"
SERVER_PID_FILE="$LOG_DIR/computer-server.pid"
RUNTIME_SHA_FILE="$ROOT/.ai/.runtime_source_sha"
server_pids() {
  pgrep -f 'uvicorn (server|contract_server):app --host 127\.0\.0\.1 --port 9010' 2>/dev/null || true
}
stop_server() {
  if [ -f "$SERVER_PID_FILE" ]; then
    PID=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
    if [ -n "$PID" ]; then
      kill "$PID" 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 0.2
      done
      kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$SERVER_PID_FILE"
  fi
  for PID in $(server_pids); do
    kill "$PID" 2>/dev/null || true
  done
  sleep 0.5
}
if [ "${AIRI_FORCE_RESTART:-0}" = "1" ]; then stop_server; fi
if ! command -v python3 >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y python3 python3-venv >/dev/null 2>&1
fi
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
  rm -rf "$VENV"
  python3 -m venv --system-site-packages --without-pip "$VENV"
fi
PYTHON_BIN="$VENV/bin/python"
export PYTHONPATH="$ROOT/computer${PYTHONPATH:+:$PYTHONPATH}"

# Resolve the current source identity only after the Python runtime is ready.
# Git HEAD is authoritative; the deployment marker is only a fallback without Git.
export ROOT_FOR_AIRI="$ROOT"
export RUNTIME_SHA_FILE_FOR_AIRI="$RUNTIME_SHA_FILE"
if [ -z "${AIRI_EXPECTED_SHA:-}" ]; then
  AIRI_EXPECTED_SHA="$("$PYTHON_BIN" -c 'import os; from startup import resolve_expected_sha; print(resolve_expected_sha(os.environ["ROOT_FOR_AIRI"], os.environ["RUNTIME_SHA_FILE_FOR_AIRI"], os.environ))')"
  export AIRI_EXPECTED_SHA
fi
AIRI_BOOTSTRAP_SHA="$("$PYTHON_BIN" -c 'import os; from startup import resolve_runtime_sha; print(resolve_runtime_sha(os.environ["ROOT_FOR_AIRI"], os.environ["RUNTIME_SHA_FILE_FOR_AIRI"], os.environ))')"
export AIRI_BOOTSTRAP_SHA
if [ -f "$ROOT/computer/requirements.txt" ]; then
  if ! "$PYTHON_BIN" -c 'import fastapi,uvicorn,pyautogui,pytesseract,PIL,playwright' >/dev/null 2>&1; then
    command -v pip3 >/dev/null 2>&1 || { echo 'AIRI_START_NO_PIP3' >&2; exit 2; }
    SITE_PACKAGES="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    pip3 install --disable-pip-version-check --quiet --target "$SITE_PACKAGES" -r "$ROOT/computer/requirements.txt"
  fi
fi
if ! "$PYTHON_BIN" -c 'import tkinter' >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y python3-tk python3-dev >/dev/null 2>&1
fi
if ! command -v Xvfb >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y xvfb xdotool openbox xterm x11-utils >/dev/null 2>&1
fi
if ! command -v xdpyinfo >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y x11-utils >/dev/null 2>&1
fi
if ! pgrep -f "[X]vfb ${DISPLAY} -screen 0 1280x800x24" >/dev/null 2>&1; then
  nohup Xvfb "$DISPLAY" -screen 0 1280x800x24 -ac >"$LOG_DIR/xvfb.log" 2>&1 < /dev/null &
  sleep 1
fi
if ! DISPLAY="$DISPLAY" xdpyinfo >/dev/null 2>&1; then
  pkill -f "[X]vfb ${DISPLAY} " >/dev/null 2>&1 || true
  sleep 1
  nohup Xvfb "$DISPLAY" -screen 0 1280x800x24 -ac >"$LOG_DIR/xvfb.log" 2>&1 < /dev/null &
  sleep 2
  DISPLAY="$DISPLAY" xdpyinfo >/dev/null 2>&1 || { echo 'Xvfb health check failed after recovery' >&2; exit 3; }
fi
if ! DISPLAY="$DISPLAY" "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from playwright.sync_api import sync_playwright
try:
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True,args=["--no-sandbox"]); b.close()
except Exception:
    raise SystemExit(1)
PY
then
  "$PYTHON_BIN" -m playwright install chromium >/dev/null 2>&1
fi
if ! DISPLAY="$DISPLAY" "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from playwright.sync_api import sync_playwright
try:
    with sync_playwright() as p:
        b=p.chromium.launch(headless=False,args=["--no-sandbox"]); b.close()
except Exception:
    raise SystemExit(1)
PY
then
  echo 'AIRI_START_BROWSER_NOT_READY' >&2
  exit 5
fi
if ! pgrep -f "[o]penbox.*${DISPLAY}" >/dev/null 2>&1; then
  nohup env DISPLAY="$DISPLAY" openbox >"$LOG_DIR/openbox.log" 2>&1 < /dev/null & sleep 1
fi
if ! DISPLAY="$DISPLAY" xdotool search --name 'Airi Terminal' >/dev/null 2>&1; then
  nohup env DISPLAY="$DISPLAY" xterm -title 'Airi Terminal' >"$LOG_DIR/xterm.log" 2>&1 < /dev/null & sleep 1
fi
runtime_ready_matches() {
  READY_JSON="$(curl -sS --max-time 2 http://127.0.0.1:9010/ready 2>/dev/null || true)"
  [ -n "$READY_JSON" ] || return 1
  printf '%s' "$READY_JSON" | EXPECTED_SHA="$AIRI_EXPECTED_SHA" "$PYTHON_BIN" -c '
import json, os, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
from startup import runtime_needs_restart
raise SystemExit(1 if runtime_needs_restart(payload, os.environ.get("EXPECTED_SHA")) else 0)
'
}

# Never reuse a server merely because /status answers: /ready must prove both
# readiness and source identity for the current checkout.
if curl -fsS --max-time 2 http://127.0.0.1:9010/status >/dev/null 2>&1; then
  if ! runtime_ready_matches; then
    stop_server
  fi
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:9010/status >/dev/null 2>&1; then
  cd "$ROOT/computer"
  nohup "$PYTHON_BIN" -m uvicorn contract_server:app --host 127.0.0.1 --port 9010 >"$LOG_DIR/computer-server.log" 2>&1 &
  echo $! > "$SERVER_PID_FILE"
fi
READY=0
for _ in $(seq 1 30); do
  if runtime_ready_matches; then READY=1; break; fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo 'AIRI_START_NOT_READY' >&2
  cat "$LOG_DIR/computer-server.log" 2>/dev/null || true
  exit 4
fi
if [ -x "$ROOT/scripts/airi-supervisor" ] && ! pgrep -f '[a]iri-supervisor' >/dev/null 2>&1; then
  nohup "$ROOT/scripts/airi-supervisor" >"$LOG_DIR/supervisor.log" 2>&1 < /dev/null &
fi
curl -fsS http://127.0.0.1:9010/status
