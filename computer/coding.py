from __future__ import annotations
import os, re, shutil, subprocess, time
from pathlib import Path
from typing import Any, Optional
ROOT=Path(os.environ.get('AIRIPC_WORKSPACE_ROOT','/home/user/airi')).resolve()
SKILLS=ROOT/'skills'; AI=ROOT/'.ai'; MAX_READ=200000
SAFE_EXTS={'.py','.js','.ts','.tsx','.jsx','.json','.toml','.yaml','.yml','.md','.txt','.html','.css','.scss','.sh','.java','.c','.cpp','.h','.hpp','.rs','.go','.sql','.xml','.ini','.cfg'}

def safe_path(p: str|Path, missing=False):
    q=(ROOT/str(p)).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    try:q.relative_to(ROOT)
    except ValueError:raise ValueError('path outside Airi-PC workspace')
    if not missing and not q.exists():raise FileNotFoundError(str(q))
    return q

def run(cmd,cwd='.',timeout=120,allow_shell=False):
    if not allow_shell and any(x in cmd for x in ('rm -rf','shutdown','reboot','mkfs','dd ','sudo ','curl |','wget -O-')):raise PermissionError('risky shell blocked')
    p=subprocess.run(['/bin/bash','-lc',cmd],cwd=str(safe_path(cwd)),text=True,capture_output=True,timeout=timeout,check=False)
    return {'returncode':p.returncode,'stdout':p.stdout[-20000:],'stderr':p.stderr[-20000:],'command':cmd}

def tree(path='.',limit=4000):
    r=safe_path(path); out=[]
    for p in r.rglob('*'):
        if p.is_file() and '.git' not in p.parts and '.venv' not in p.parts and '__pycache__' not in p.parts:
            out.append(str(p.relative_to(ROOT)))
            if len(out)>=limit:break
    return {'root':str(r.relative_to(ROOT)),'file_count':len(out),'files':sorted(out),'truncated':len(out)>=limit}

def analyze(path='.'):
    r=safe_path(path); exts={}; tests=[]
    for p in r.rglob('*'):
        if not p.is_file() or '.git' in p.parts or '.venv' in p.parts or '__pycache__' in p.parts:continue
        if p.suffix in SAFE_EXTS:exts[p.suffix]=exts.get(p.suffix,0)+1
        if 'test' in p.name.lower() or p.parent.name.lower() in {'test','tests'}:tests.append(str(p.relative_to(ROOT)))
    configs=[x for x in ['requirements.txt','pyproject.toml','package.json','Cargo.toml','go.mod','pom.xml','build.gradle','Dockerfile','Makefile','pytest.ini','tox.ini'] if (r/x).exists()]
    frameworks=[]
    if (r/'pyproject.toml').exists() or (r/'requirements.txt').exists():frameworks.append('python')
    if (r/'package.json').exists():frameworks.append('javascript/typescript')
    if (r/'Cargo.toml').exists():frameworks.append('rust')
    if (r/'go.mod').exists():frameworks.append('go')
    if (r/'pom.xml').exists() or (r/'build.gradle').exists():frameworks.append('java')
    if (r/'Dockerfile').exists():frameworks.append('docker')
    return {'project':str(r.relative_to(ROOT)),'tree':tree(path),'languages':exts,'framework_hints':frameworks,'configs':configs,'tests':tests[:200],'git':git_status(path)}

def read(path):
    p=safe_path(path); s=p.read_text(errors='replace'); return {'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'content':s[:MAX_READ],'truncated':len(s)>MAX_READ}

def search(query,path='.',limit=100):
    r=safe_path(path); n=query.casefold(); out=[]
    for p in r.rglob('*'):
        if not p.is_file() or '.git' in p.parts or '.venv' in p.parts or '__pycache__' in p.parts:continue
        try:lines=p.read_text(errors='replace').splitlines()
        except Exception:continue
        for i,line in enumerate(lines,1):
            if n in line.casefold():out.append({'path':str(p.relative_to(ROOT)),'line':i,'text':line[:500]})
            if len(out)>=limit:return {'query':query,'matches':out,'count':len(out)}
    return {'query':query,'matches':out,'count':len(out)}

def write(path,content):
    p=safe_path(path,True);p.parent.mkdir(parents=True,exist_ok=True);backup=None
    if p.exists():
        backup=AI/'backups'/f'{int(time.time()*1000)}-{p.name}.bak';backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
    p.write_text(content);return {'ok':True,'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'backup':str(backup.relative_to(ROOT)) if backup else None}

def patch(path,old,new,replace_all=False):
    p=safe_path(path);s=p.read_text(errors='replace');c=s.count(old)
    if c==0:raise ValueError('patch target not found')
    if c!=1 and not replace_all:raise ValueError(f'patch target occurs {c} times')
    return write(path,s.replace(old,new,-1 if replace_all else 1))

def test(command='pytest -q',cwd='.',timeout=170):return run(command,cwd,timeout,True)
def build(command='python -m compileall -q .',cwd='.',timeout=170):return run(command,cwd,timeout,True)
def lint(command='python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")',cwd='.',timeout=170):return run(command,cwd,timeout,True)
def shell(command,cwd='.',timeout=120,allow_shell=False):return run(command,cwd,timeout,allow_shell)

def git_status(path='.'):
    r=safe_path(path)
    if not (r/'.git').exists():return {'available':False}
    return {'available':True,**run('git status --short --branch',path,60,True)}
def git_diff(path='.'):return run('git diff -- .',path,60,True) if (safe_path(path)/'.git').exists() else {'available':False}
def git_log(path='.',n=10):return run(f'git log -n {int(n)} --oneline --decorate',path,60,True) if (safe_path(path)/'.git').exists() else {'available':False}
def git_commit(message,path='.'):
    r=safe_path(path)
    if not (r/'.git').exists():raise RuntimeError('not a git repository')
    run('git add -A',path,60,True);return run(f'git commit -m {shlex_quote(message)}',path,120,True)

def shlex_quote(s):return "'"+s.replace("'","'\\''")+"'"
