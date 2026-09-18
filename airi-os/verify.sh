#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OS=$ROOT/airi-os

required=(
  "$OS/auto/config"
  "$OS/config/package-lists/airi.list.chroot"
  "$OS/config/includes.chroot/etc/systemd/system/airi-pc.service"
  "$OS/config/includes.chroot/usr/local/bin/airi-status"
  "$OS/config/includes.chroot/usr/local/sbin/airi-update"
  "$OS/config/includes.chroot/usr/local/sbin/airi-rollback"
  "$OS/config/hooks/live/010-airi-setup.hook.chroot"
  "$OS/build.sh"
)
for f in "${required[@]}"; do
  [ -s "$f" ] || { echo "missing required file: $f" >&2; exit 1; }
done

bash -n "$OS/build.sh"
bash -n "$OS/auto/config"
bash -n "$OS/config/hooks/live/010-airi-setup.hook.chroot"
bash -n "$OS/config/includes.chroot/usr/local/sbin/airi-update"
bash -n "$OS/config/includes.chroot/usr/local/sbin/airi-rollback"

grep -q 'distribution trixie' "$OS/auto/config"
grep -q 'airi-pc.service' "$OS/config/hooks/live/010-airi-setup.hook.chroot"
grep -q 'Running verification before activation' "$OS/config/includes.chroot/usr/local/sbin/airi-update"
grep -q 'AIRI_LIVE_TELEMETRY=1' "$OS/config/includes.chroot/etc/systemd/system/airi-pc.service"

echo "AIRI_OS_SOURCE_VERIFY=PASS"
