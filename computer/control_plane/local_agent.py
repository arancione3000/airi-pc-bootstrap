from __future__ import annotations

"""Disabled legacy local-agent compatibility surface.

ChatGPT is the sole reasoning authority. Airi-PC exposes execution tools and
persistence but never invokes a local or remote reasoning model.
"""
import os, subprocess
from pathlib import Path
from typing import Any
ROOT = Path(os.environ.get("AIRI_ROOT", "/home/user/airi")).resolve()
MAX_ITERATIONS = max(1, min(int(os.environ.get("AIRI_AUTONOMOUS_ITERATIONS", "5")), 20))
TEST_COMMAND = os.environ.get("AIRI_AUTONOMOUS_TEST", "python3 -m pytest -q")
CHATGPT_ONLY = True
REASONING_AUTHORITY = "chatgpt"
PROVIDER = "disabled"
MODEL = None
def _disabled(operation: str) -> RuntimeError:
    return RuntimeError(f"{operation} is disabled: ChatGPT is the sole reasoning authority; Airi-PC provides execution tools only and contains no reasoning model.")
def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
def _context_at(root: Path) -> str:
    def run_local(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=30, check=False)
    status=run_local(["git","status","--short"]); diff=run_local(["git","diff","--","."]); log=run_local(["git","log","-5","--oneline"])
    return "\n".join([f"REPOSITORY={root}","\nGIT_STATUS:\n"+status.stdout,"\nGIT_DIFF:\n"+diff.stdout[:30000],"\nGIT_LOG:\n"+log.stdout[:4000]])
def _context() -> str: return _context_at(ROOT)
def ask_local_model(goal: str, feedback: str = "") -> str:
    del goal, feedback; raise _disabled("local model invocation")
def ask_local_model_changes(goal: str, context: str, feedback: str = "", root: str | Path | None = None, model: str | None = None) -> list[dict[str, Any]]:
    del goal, context, feedback, root, model; raise _disabled("model-driven change generation")
def _provider_request(messages: list[dict[str,str]], model: str) -> dict[str,Any]:
    del messages, model; raise _disabled("provider request")
def _safe_patch(patch: str) -> None:
    blocked=(".git/",".ssh/","auth/","id_rsa","credentials","secret")
    for line in patch.splitlines():
        if line.startswith(("+++ ","--- ")):
            path=line[4:].split("\t",1)[0]; path=path[2:] if path.startswith(("a/","b/")) else path
            if any(token in path.lower() for token in blocked): raise RuntimeError(f"Unsafe patch path: {path}")
            candidate=(ROOT/path).resolve()
            if ROOT not in candidate.parents and candidate != ROOT: raise RuntimeError(f"Patch escapes repository: {path}")
def apply_patch(patch: str) -> None:
    _safe_patch(patch)
    check=subprocess.run(["git","apply","--check","--whitespace=nowarn","-"],cwd=ROOT,text=True,input=patch,capture_output=True,check=False)
    if check.returncode: raise RuntimeError("git apply --check failed: "+check.stderr[-8000:])
    applied=subprocess.run(["git","apply","--whitespace=nowarn","-"],cwd=ROOT,text=True,input=patch,capture_output=True,check=False)
    if applied.returncode: raise RuntimeError("git apply failed: "+applied.stderr[-8000:])
def run_tests() -> dict[str,Any]:
    p=subprocess.run(TEST_COMMAND,cwd=ROOT,shell=True,text=True,capture_output=True,timeout=180,check=False); return {"returncode":p.returncode,"stdout":p.stdout[-12000:],"stderr":p.stderr[-12000:]}
def autonomous_cycle(goal: str, iterations: int = MAX_ITERATIONS, commit: bool = True) -> dict[str,Any]:
    del goal, iterations, commit; raise _disabled("autonomous reasoning cycle")
def provider_status() -> dict[str,Any]:
    return {"available":False,"provider":"disabled","model":None,"chatgpt_only":True,"reason":"No local or remote reasoning provider is implemented in Airi-PC."}
if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Disabled legacy local-agent compatibility surface"); parser.add_argument("goal"); parser.add_argument("--iterations",type=int,default=MAX_ITERATIONS); parser.add_argument("--no-commit",action="store_true"); parser.parse_args(); raise SystemExit(_disabled("local-agent CLI"))
