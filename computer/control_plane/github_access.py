from __future__ import annotations
"""Safe GitHub persistence primitives using the authenticated local Git client."""
import os, subprocess
from pathlib import Path
ROOT=Path(os.environ.get("AIRI_ROOT","/home/user/airi")).resolve()
BLOCKED_PATHS=(".git",".ssh","auth","credentials","secrets")
def _run(args:list[str],timeout:int=120)->subprocess.CompletedProcess[str]: return subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout,check=False)
def repository_root()->Path:
    r=_run(["git","rev-parse","--show-toplevel"]);
    if r.returncode: raise RuntimeError(r.stderr.strip() or "Not a Git repository")
    return Path(r.stdout.strip()).resolve()
def current_branch()->str:
    r=_run(["git","branch","--show-current"]);
    if r.returncode or not r.stdout.strip(): raise RuntimeError(r.stderr.strip() or "Detached Git HEAD")
    return r.stdout.strip()
def current_sha()->str:
    r=_run(["git","rev-parse","HEAD"]);
    if r.returncode: raise RuntimeError(r.stderr.strip() or "Unable to resolve HEAD")
    return r.stdout.strip()
def remote_url(name:str="origin")->str:
    r=_run(["git","remote","get-url",name]);
    if r.returncode: raise RuntimeError(r.stderr.strip() or f"Git remote {name!r} not configured")
    return r.stdout.strip()
def assert_safe_paths(paths:list[str])->None:
    for raw in paths:
        path=Path(raw).resolve()
        try: relative=path.relative_to(ROOT)
        except ValueError as exc: raise PermissionError(f"Path outside repository: {raw}") from exc
        if {p.lower() for p in relative.parts}.intersection(BLOCKED_PATHS) or path.name.lower() in {"id_rsa","id_ed25519"}: raise PermissionError(f"Sensitive Git path blocked: {raw}")
def status()->dict[str,object]:
    r=_run(["git","status","--short","--branch"]); return {"ok":r.returncode==0,"output":r.stdout,"error":r.stderr,"branch":current_branch() if r.returncode==0 else None,"sha":current_sha() if r.returncode==0 else None}
def commit(message:str,paths:list[str]|None=None)->str:
    if not message.strip(): raise ValueError("Commit message must not be empty")
    selected=[str(Path(p)) for p in (paths or ["."])]; assert_safe_paths(selected)
    a=_run(["git","add","--",*selected]);
    if a.returncode: raise RuntimeError(a.stderr.strip() or "git add failed")
    c=_run(["git","commit","-m",message]);
    if c.returncode: raise RuntimeError(c.stderr.strip() or c.stdout.strip() or "git commit failed")
    return current_sha()
def push(branch:str|None=None)->dict[str,object]:
    target=branch or current_branch(); r=_run(["git","push","--set-upstream","origin",target],timeout=180); return {"ok":r.returncode==0,"branch":target,"stdout":r.stdout,"stderr":r.stderr}
