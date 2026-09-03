#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT="${AIRIPC_WORKSPACE_ROOT:-${AIRI_WORKSPACE_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}}"
ROOT=$(CDPATH= cd -- "$ROOT" && pwd)
VENV="${AIRIPC_VENV:-$ROOT/.venv}"
LOG_DIR="${AIRI_LOG_DIR:-$ROOT/logs}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT/.cache/ms-playwright}"
mkdir -p "$LOG_DIR" "$PLAYWRIGHT_BROWSERS_PATH"
export DISPLAY=":${DISPLAY_NUM:-99}"
SERVER_PID_FILE="$LOG_DIR/computer-server.pid"
RUNTIME_SHA_FILE="$ROOT/.ai/.runtime_source_sha"
stop_server() {
  if [ -f "$SERVER_PID_FILE" ]; then
    PID=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
    [ -z "$PID" ] || kill "$PID" 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
  fi
  pkill -f 'uvicorn (server|contract_server):app --host 127\\.0\\.0\\.1 --port 9010' 2>/dev/null || true
}
if [ "${AIRI_FORCE_RESTART:-0}" = "1" ]; then stop_server; sleep 1; fi
if [ -f "$RUNTIME_SHA_FILE" ]; then export AIRI_BOOTSTRAP_SHA="$(cat "$RUNTIME_SHA_FILE")"; fi
if [ -z "${AIRI_BOOTSTRAP_SHA:-}" ] && command -v git >/dev/null 2>&1; then
  AIRI_BOOTSTRAP_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
  [ -n "$AIRI_BOOTSTRAP_SHA" ] && export AIRI_BOOTSTRAP_SHA
fi
export AIRI_EXPECTED_SHA="${AIRI_EXPECTED_SHA:-${AIRI_BOOTSTRAP_SHA:-}}"
if ! command -v python3 >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y python3 python3-venv >/dev/null 2>&1
fi
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
  rm -rf "$VENV"
  python3 -m venv --system-site-packages --without-pip "$VENV"
fi
PYTHON_BIN="$VENV/bin/python"
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
if ! pgrep -f '[X]vfb :99 -screen 0 1280x800x24' >/dev/null 2>&1; then
  nohup Xvfb :99 -screen 0 1280x800x24 -ac >"$LOG_DIR/xvfb.log" 2>&1 < /dev/null &
  sleep 1
fi
if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
  pkill -f '[X]vfb :99 ' >/dev/null 2>&1 || true
  sleep 1
  nohup Xvfb :99 -screen 0 1280x800x24 -ac >"$LOG_DIR/xvfb.log" 2>&1 < /dev/null &
  sleep 2
  DISPLAY=:99 xdpyinfo >/dev/null 2>&1 || { echo 'Xvfb health check failed after recovery' >&2; exit 3; }
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
if ! pgrep -f '[o]penbox.*:99' >/dev/null 2>&1; then
  nohup env DISPLAY=:99 openbox >"$LOG_DIR/openbox.log" 2>&1 < /dev/null & sleep 1
fi
if ! DISPLAY=:99 xdotool search --name 'Airi Terminal' >/dev/null 2>&1; then
  nohup env DISPLAY=:99 xterm -title 'Airi Terminal' >"$LOG_DIR/xterm.log" 2>&1 < /dev/null & sleep 1
fi
if ! curl -fsS http://127.0.0.1:9010/status >/dev/null 2>&1; then
  cd "$ROOT/computer"
  nohup "$PYTHON_BIN" -m uvicorn contract_server:app --host 127.0.0.1 --port 9010 >"$LOG_DIR/computer-server.log" 2>&1 &
  echo $! > "$SERVER_PID_FILE"
fi
READY=0
for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:9010/ready >/dev/null 2>&1; then READY=1; break; fi
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
