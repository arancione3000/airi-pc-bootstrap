from __future__ import annotations
import os, shlex
from pathlib import Path
ROOT=Path(os.environ.get("AIRIPC_WORKSPACE_ROOT","/home/user/airi")).resolve()

# Fail-safe security helpers. Normal workspace operations remain allowed.
def safe_path(raw: str, *, allow_create: bool=True) -> Path:
    p=Path(raw).expanduser()
    if not p.is_absolute(): p=ROOT/p
    p=p.resolve(strict=False)
    try: p.relative_to(ROOT)
    except ValueError: raise PermissionError(f"path outside workspace: {raw}")
    return p

def validate_delete_target(raw: str) -> Path:
    p=safe_path(raw, allow_create=False)
    if p == ROOT: raise PermissionError("refusing to delete workspace root")
    return p

DANGEROUS_PATTERNS=(
    "rm -rf /", "mkfs", "fdisk", "parted", "shutdown", "reboot",
    "poweroff", ":(){ :|:& };:", "dd if=",
)

def shell_command(command: str, allow_shell: bool=False) -> str:
    if not isinstance(command,str) or not command.strip(): raise ValueError("empty command")
    c=command.strip()
    if not allow_shell:
        for pat in DANGEROUS_PATTERNS:
            if pat in c:
                raise PermissionError("dangerous command requires allow_shell=true")
    return c
