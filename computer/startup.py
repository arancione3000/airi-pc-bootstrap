from __future__ import annotations
import os, subprocess
from pathlib import Path
from typing import Mapping, Any

def _clean(value: str | None) -> str: return (value or '').strip()
def read_sha_marker(path: str | Path) -> str:
    try: return _clean(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError): return ''
def git_source_sha(root: str | Path) -> str:
    try:
        p=subprocess.run(['git','-C',str(Path(root).resolve()),'rev-parse','HEAD'],text=True,capture_output=True,timeout=5,check=False)
    except (OSError, subprocess.SubprocessError): return ''
    return _clean(p.stdout) if p.returncode==0 else ''
def resolve_runtime_sha(root, marker_path, environ=None):
    env=os.environ if environ is None else environ
    return git_source_sha(root) or _clean(env.get('AIRI_BOOTSTRAP_SHA')) or read_sha_marker(marker_path)
def resolve_expected_sha(root, marker_path, environ=None):
    env=os.environ if environ is None else environ
    return _clean(env.get('AIRI_EXPECTED_SHA')) or git_source_sha(root) or read_sha_marker(marker_path)
def runtime_needs_restart(payload: Any, expected_sha: str | None) -> bool:
    if not isinstance(payload, dict) or payload.get('ready') is not True: return True
    checks=payload.get('checks')
    if not isinstance(checks, dict) or checks.get('source_match') is not True: return True
    expected=_clean(expected_sha)
    return bool(expected and payload.get('source_sha') != expected)
