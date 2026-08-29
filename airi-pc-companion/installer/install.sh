#!/bin/sh
set -eu
ROOT="${AIRIPC_INSTALL_DIR:-$HOME/.airi-pc-companion}"
mkdir -p "$ROOT/app"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cp -R "$SCRIPT_DIR/companion" "$ROOT/app/"
cp "$SCRIPT_DIR/requirements.txt" "$ROOT/"
python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --upgrade pip >/dev/null
"$ROOT/venv/bin/pip" install -r "$ROOT/requirements.txt"
cat > "$ROOT/run.sh" <<EOF
#!/bin/sh
exec "$ROOT/venv/bin/python" -m companion.cli
EOF
chmod 700 "$ROOT/run.sh"
printf '%s\n' "Installed to $ROOT" "Start with: $ROOT/run.sh"
