#!/bin/bash
set -euo pipefail

OS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$OS_DIR/.." && pwd)
WORK=$OS_DIR/build-work
OUT=$OS_DIR/out
ISO_NAME=Airi-OS-0.1-amd64.iso

command -v lb >/dev/null || { echo "live-build (lb) is required" >&2; exit 2; }
command -v xorriso >/dev/null || { echo "xorriso is required" >&2; exit 2; }
[ "$(id -u)" -eq 0 ] || { echo "Run build.sh as root (sudo)." >&2; exit 2; }

rm -rf "$WORK" "$OUT"
mkdir -p "$WORK" "$OUT"
cp -a "$OS_DIR/auto" "$WORK/"
cp -a "$OS_DIR/config" "$WORK/"

DEST=$WORK/config/includes.chroot/opt/airi-pc/releases/builtin
mkdir -p "$DEST"
git -C "$ROOT" archive --format=tar HEAD | tar -xf - -C "$DEST"
mkdir -p "$DEST/.ai"
git -C "$ROOT" rev-parse HEAD > "$DEST/.ai/.runtime_source_sha"

chmod +x "$WORK/auto/config"
find "$WORK/config/hooks" -type f -name '*.hook.chroot' -exec chmod +x {} +

cd "$WORK"
bash auto/config
lb build

SOURCE_ISO=live-image-amd64.hybrid.iso
[ -s "$SOURCE_ISO" ] || { echo "live-build did not produce $SOURCE_ISO" >&2; exit 3; }
cp "$SOURCE_ISO" "$OUT/$ISO_NAME"
(
  cd "$OUT"
  sha256sum "$ISO_NAME" > "$ISO_NAME.sha256"
)

echo "AIRI_OS_ISO=$OUT/$ISO_NAME"
cat "$OUT/$ISO_NAME.sha256"
