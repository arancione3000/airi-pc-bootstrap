#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, subprocess, sys, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASE=os.environ.get("AIRIPC_BASE_URL","http://127.0.0.1:9010")
REQUIRED=["AIRI_AGENT_ENTRYPOINT.md","config/AIRI_SYSTEM_MANIFEST.json","config/AIRI_CHATGPT_ONLY.json","config/DEFAULT_REASONING_DIRECTIVE.md","config/AGENT_PROMPT_CONFIG.json","config/AGENT_PROMPT_AUTOLOAD.md","computer/control_plane/github_access.py","scripts/airi-rebuild-verify.py","computer/control_plane/local_agent.py","computer/control_plane/model_router.py","computer/control_plane/model_gateway.py","scripts/airi-local-autonomous","AIRI_LOCAL_AUTONOMOUS.md","api/index.py","computer/server.py","computer/start.sh","tests/test_openrouter_provider.py"]
FORBIDDEN=("ollama","openrouter","openrouter_bridge","airi_ollama_url","openrouter_api_key","openrouter_url","openrouter_model")
ALLOW_LEGACY={"computer/control_plane/model_gateway.py","scripts/airi-local-autonomous"}
def check(name,ok,detail=""):
    print(f"[{"OK" if ok else "ERROR"}] {name}"+(f": {detail}" if detail else "")); return bool(ok)
def git(*args): return subprocess.run(["git","-C",str(ROOT),*args],text=True,capture_output=True,check=False)
def runtime(path):
    try:
        with urllib.request.urlopen(BASE+path,timeout=5) as r: return json.loads(r.read().decode())
    except Exception: return None
def main():
    failures=[]
    for path in REQUIRED:
        if not check(f"file {path}",(ROOT/path).is_file()): failures.append(path)
    try:
        manifest=json.loads((ROOT/"config/AIRI_SYSTEM_MANIFEST.json").read_text()); cfg=json.loads((ROOT/"config/AIRI_CHATGPT_ONLY.json").read_text()); prompt=json.loads((ROOT/"config/AGENT_PROMPT_CONFIG.json").read_text())
    except Exception as exc:
        print(f"[ERROR] configuration JSON parse: {exc}"); failures.append("config"); manifest=cfg=prompt={}
    for name,ok in [("manifest_chatgpt_only",manifest.get("architecture")=="chatgpt-only" and manifest.get("reasoning_authority")=="chatgpt" and manifest.get("second_llm_allowed") is False),("config_chatgpt_only",cfg.get("architecture")=="chatgpt-only" and cfg.get("reasoning_authority")=="chatgpt" and cfg.get("allow_local_llm") is False and cfg.get("allow_remote_llm") is False),("prompt_chatgpt_only",prompt.get("reasoning_provider")=="chatgpt" and prompt.get("execution_provider")=="airi-pc")]:
        if not check(name,ok): failures.append(name)
    syntax_fail=False
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts or ".git" in p.parts: continue
        r=subprocess.run([sys.executable,"-m","py_compile",str(p)],cwd=ROOT,text=True,capture_output=True,check=False)
        if r.returncode: syntax_fail=True; check(f"syntax {p.relative_to(ROOT)}",False,r.stderr.strip()[-300:])
    if not check("python syntax",not syntax_fail): failures.append("syntax")
    branch=git("branch","--show-current").stdout.strip(); sha=git("rev-parse","HEAD").stdout.strip(); check("git branch",bool(branch),branch or "unavailable"); check("git SHA",bool(re.fullmatch(r"[0-9a-f]{40}",sha)),sha or "unavailable")
    hits=[]
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or ".venv" in p.parts or p.suffix.lower() not in {".py",".sh",".json"}: continue
        rel=str(p.relative_to(ROOT))
        if rel=="scripts/airi-rebuild-verify.py" or rel in ALLOW_LEGACY or rel.startswith("tests/"): continue
        text=p.read_text(encoding="utf-8",errors="ignore").lower()
        if any(term in text for term in FORBIDDEN): hits.append(rel)
    if not check("forbidden providers absent from operational code",not hits,", ".join(hits)): failures.extend(hits)
    for rel in ALLOW_LEGACY:
        p=ROOT/rel
        if p.exists():
            text=p.read_text(encoding="utf-8",errors="ignore").lower(); active=any(x in text for x in ("urllib.request.urlopen","requests.post","http://127.0.0.1:11434","https://openrouter.ai/api","curl -fss"))
            if not check(f"legacy surface disabled: {rel}",not active): failures.append(rel)
    status=runtime("/status"); ready=runtime("/ready")
    if status is None or ready is None: print("[SKIP] runtime endpoints unavailable")
    else:
        ok=status.get("ok") is True and status.get("display")==":99" and status.get("gui_available") is True and status.get("resolution")=="1280x800"
        if not check("runtime /status",ok,json.dumps(status)): failures.append("runtime/status")
        rc=ready.get("checks",{}); ok=ready.get("ready") is True and all(rc.get(k) is True for k in ("status","gui","browser","mcp","source_match"))
        if not check("runtime /ready",ok,json.dumps(ready)): failures.append("runtime/ready")
    print(json.dumps({"branch":branch,"sha":sha,"failures":failures,"ok":not failures},indent=2)); return 1 if failures else 0
if __name__=="__main__": raise SystemExit(main())
