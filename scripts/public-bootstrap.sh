#!/bin/sh
set -eu
ROOT=/home/user/airi
REPO_URL=https://github.com/arancione3000/airi-pc-bootstrap.git
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$ROOT"
TMP_SRC="$TMP/source"
git init -q "$TMP_SRC"
git -C "$TMP_SRC" remote add origin "$REPO_URL"
EXPECTED_SHA="$(git ls-remote "$REPO_URL" refs/heads/main | awk '{print $1}')"
[ -n "$EXPECTED_SHA" ] || { echo AIRI_SOURCE_REF_MISSING >&2; exit 3; }
git -C "$TMP_SRC" fetch -q --depth=1 origin main || { echo AIRI_GIT_FETCH_FAILED >&2; exit 4; }
FETCHED_SHA="$(git -C "$TMP_SRC" rev-parse FETCH_HEAD)"
[ "$FETCHED_SHA" = "$EXPECTED_SHA" ] || { echo AIRI_SOURCE_SHA_MISMATCH >&2; exit 5; }
git -C "$TMP_SRC" checkout -q -b main "$FETCHED_SHA" || { echo AIRI_BRANCH_CHECKOUT_FAILED >&2; exit 6; }
if [ -e "$ROOT/computer/start.sh" ]; then BACKUP="$ROOT.previous.$(date +%s)"; mv "$ROOT" "$BACKUP" || { echo AIRI_WORKSPACE_BACKUP_FAILED >&2; exit 7; }; fi
mv "$TMP_SRC" "$ROOT"
chmod +x "$ROOT/computer/start.sh" "$ROOT/scripts/airi-control" "$ROOT/scripts/airi-selftest" "$ROOT/scripts/airi-agent.py" 2>/dev/null || true
sh "$ROOT/computer/start.sh"
python3 "$ROOT/scripts/airi-agent.py"
