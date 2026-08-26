#!/bin/sh
set -eu
ROOT=/home/user/airi
REPO_URL=https://github.com/arancione3000/airi-pc-bootstrap/archive/refs/heads/main.zip
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$ROOT"
python3 - <<PY
import urllib.request, zipfile, os
url=os.environ.get("REPO_URL")
zip_path=os.path.join("$TMP","airi.zip")
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path) as z: z.extractall("$TMP/extracted")
root=os.path.join("$TMP","extracted", os.listdir(os.path.join("$TMP","extracted"))[0])
for name in os.listdir(root):
    src=os.path.join(root,name); dst=os.path.join("$ROOT",name)
    if os.path.isdir(src):
        import shutil; shutil.copytree(src,dst,dirs_exist_ok=True)
    else:
        import shutil; shutil.copy2(src,dst)
PY
chmod +x "$ROOT/computer/start.sh" "$ROOT/scripts/airi-control" "$ROOT/scripts/airi-selftest" "$ROOT/scripts/airi-agent.py" 2>/dev/null || true
sh "$ROOT/computer/start.sh"
python3 "$ROOT/scripts/airi-agent.py"
