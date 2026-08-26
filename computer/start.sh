#!/bin/sh
set -eu
ROOT=/home/user/airi
VENV="$ROOT/.venv"
mkdir -p "$ROOT/logs"
export DISPLAY=":${DISPLAY_NUM:-99}"

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
  # Recover from stale/dead Xvfb processes instead of blocking a fresh session.
  pkill -f '[X]vfb :99 ' >/dev/null 2>&1 || true
  sleep 1
  Xvfb :99 -screen 0 1280x800x24 -ac >"$ROOT/logs/xvfb.log" 2>&1 &
  sleep 2
  if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
    echo 'Xvfb health check failed after recovery' >&2
    exit 3
  fi
fi

# Browser preflight: install Chromium once when missing. This is skipped when cached.
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

if ! curl -fsS http://127.0.0.1:9010/status >/dev/null 2>&1; then
  cd "$ROOT/computer"
  nohup "$VENV/bin/uvicorn" server:app --host 127.0.0.1 --port 9010 >"$ROOT/logs/computer-server.log" 2>&1 &
  sleep 2
fi

curl -fsS http://127.0.0.1:9010/status
