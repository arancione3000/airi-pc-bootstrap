#!/bin/sh
set -eu
ROOT=/home/user/airi
VENV="$ROOT/.venv"
mkdir -p "$ROOT/logs"
GATEWAY_PORT="${AIRI_MODEL_GATEWAY_PORT:-17893}"
if ! curl -fsS --max-time 1 "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
  nohup env PYTHONPATH="$ROOT/computer" AIRI_MODEL_GATEWAY_PORT="$GATEWAY_PORT" AIRI_MODEL_GATEWAY_AUTH_FILE="${AIRI_MODEL_GATEWAY_AUTH_FILE:-/tmp/airi-model-gateway.token}" "$VENV/bin/python" -c "from control_plane.model_gateway import serve; serve()" >"$ROOT/logs/model-gateway.log" 2>&1 &
fi
export DISPLAY=":${DISPLAY_NUM:-99}"
SERVER_PID_FILE="$ROOT/logs/computer-server.pid"
RUNTIME_SHA_FILE="$ROOT/.ai/.runtime_source_sha"

stop_server() {
  if [ -f "$SERVER_PID_FILE" ]; then
    PID=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
    [ -z "$PID" ] || kill "$PID" 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
  fi
  pkill -f 'uvicorn (server|contract_server):app --host 127\\.0\\.0\\.1 --port 9010' 2>/dev/null || true
}

if [ "${AIRI_FORCE_RESTART:-0}" = "1" ]; then
  stop_server
  sleep 1
fi
if [ -f "$RUNTIME_SHA_FILE" ]; then export AIRI_BOOTSTRAP_SHA="$(cat "$RUNTIME_SHA_FILE")"; fi
export AIRI_EXPECTED_SHA="${AIRI_EXPECTED_SHA:-${AIRI_BOOTSTRAP_SHA:-}}"

if ! command -v python3 >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y python3 python3-venv >/dev/null 2>&1
fi

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi

if [ -f "$ROOT/computer/requirements.txt" ]; then
  "$VENV/bin/python" -m pip install --disable-pip-version-check --quiet -r "$ROOT/computer/requirements.txt"
fi

if ! command -v Xvfb >/dev/null 2>&1; then
  sudo -n apt-get update >/dev/null 2>&1
  sudo -n apt-get install -y xvfb xdotool openbox xterm >/dev/null 2>&1
fi

if ! pgrep -f '[X]vfb :99 -screen 0 1280x800x24' >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1280x800x24 -ac >"$ROOT/logs/xvfb.log" 2>&1 &
  sleep 1
fi

if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
  pkill -f '[X]vfb :99 ' >/dev/null 2>&1 || true
  sleep 1
  Xvfb :99 -screen 0 1280x800x24 -ac >"$ROOT/logs/xvfb.log" 2>&1 &
  sleep 2
  if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
    echo 'Xvfb health check failed after recovery' >&2
    exit 3
  fi
fi

if ! DISPLAY=:99 "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
from pathlib import Path
root=Path.home()/'.cache'/'ms-playwright'
paths=list(root.glob('chromium-*/chrome-linux*/chrome'))+list(root.glob('chromium-*/chrome'))+list(root.glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
raise SystemExit(0 if any(p.exists() for p in paths) else 1)
PY
then
  "$VENV/bin/python" -m playwright install chromium >/dev/null 2>&1
fi

if ! pgrep -f '[o]penbox.*:99' >/dev/null 2>&1; then
  DISPLAY=:99 openbox >"$ROOT/logs/openbox.log" 2>&1 &
  sleep 1
fi

if ! DISPLAY=:99 xdotool search --name 'Airi Terminal' >/dev/null 2>&1; then
  DISPLAY=:99 xterm -title 'Airi Terminal' >"$ROOT/logs/xterm.log" 2>&1 &
  sleep 1
fi

if [ -x "$ROOT/scripts/airi-supervisor" ] && ! pgrep -f '[a]iri-supervisor' >/dev/null 2>&1; then
  "$ROOT/scripts/airi-supervisor" >/dev/null 2>&1 &
fi

if ! curl -fsS http://127.0.0.1:9010/status >/dev/null 2>&1; then
  cd "$ROOT/computer"
  nohup "$VENV/bin/uvicorn" contract_server:app --host 127.0.0.1 --port 9010 >"$ROOT/logs/computer-server.log" 2>&1 &
  echo $! > "$SERVER_PID_FILE"
fi

READY=0
for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:9010/ready >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" != "1" ]; then
  echo 'AIRI_START_NOT_READY' >&2
  cat "$ROOT/logs/computer-server.log" 2>/dev/null || true
  exit 4
fi

curl -fsS http://127.0.0.1:9010/status
