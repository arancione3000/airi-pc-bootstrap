#!/bin/sh
set -eu
ROOT="${AIRIPC_INSTALL_DIR:-$HOME/.airi-pc-companion}"
pkill -f "$ROOT/venv/bin/python -m companion.cli" 2>/dev/null || true
rm -rf "$ROOT"
printf '%s\n' 'Airi-PC Companion removed.'
