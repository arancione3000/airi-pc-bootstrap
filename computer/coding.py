from __future__ import annotations
import json, os, re, shlex, shutil, subprocess, time, hashlib
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(os.environ.get('AIRIPC_WORKSPACE_ROOT','/home/user/airi')).resolve()
SKILLS = ROOT / 'skills'
AI = ROOT / '.ai'
BACKUPS = AI / 'backups'
MAX_READ = 200_000
SAFE_EXTS = {'.py','.js','.ts','.tsx','.jsx','.json','.toml','.yaml','.yml','.md','.txt','.html','.css','.scss','.sh','.java','.c','.cpp','.h','.hpp','.rs','.go','.sql','.xml','.ini','.cfg'}


def safe_path(p: str | Path, missing: bool = False) -> Path:
    q = (ROOT / str(p)).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    try:
        q.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('path outside Airi-PC workspace') from exc
    if not missing and not q.exists():
        raise FileNotFoundError(str(q))
    return q


def _inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _git_root(path: str | Path = '.') -> Optional[Path]:
    p = safe_path(path)
    cur = p if p.is_dir() else p.parent
    while _inside(cur, ROOT):
        if (cur / '.git').exists():
            return cur
        if cur == ROOT:
            break
        cur = cur.parent
    return None


def project_context(path: str = '.') -> dict[str, Any]:
    root = safe_path(path)
    if not root.is_dir():
        root = root.parent
    cur = root
    while _inside(cur, ROOT):
        for name in ('AGENTS.md','CLAUDE.md'):
            candidate = cur / name
            if candidate.exists():
                content = candidate.read_text(errors='replace')
                return {
                    'loaded': True,
                    'path': str(candidate.relative_to(ROOT)),
                    'bytes': candidate.stat().st_size,
                    'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    'content': content[:MAX_READ],
                }
        if cur == ROOT: break
        cur = cur.parent
    return {'loaded': False, 'path': None, 'bytes': 0, 'sha256': None, 'content': ''}


def load_project_context(path: str = '.') -> dict[str, Any]:
    ctx = project_context(path)
    if not ctx['loaded']:
        raise RuntimeError('project context missing: create AGENTS.md or CLAUDE.md before coding')
    return ctx


def _normalize_scope(scope: Optional[Iterable[str]]) -> list[Path]:
    return [safe_path(x, missing=True) for x in (scope or [])]


def scope_check(paths: Iterable[str], declared_scope: Optional[Iterable[str]]) -> dict[str, Any]:
    scope = _normalize_scope(declared_scope)
    if not scope:
        raise PermissionError('declared task scope is required for autonomous coding changes')
    checked = []
    for raw in paths:
        p = safe_path(raw, missing=True)
        if not any(_inside(p, s) or _inside(s, p) for s in scope):
            raise PermissionError(f'path outside declared task scope: {p.relative_to(ROOT)}')
        checked.append(str(p.relative_to(ROOT)))
    return {'ok': True, 'paths': checked, 'scope': [str(x.relative_to(ROOT)) for x in scope]}


def run(cmd: str, cwd: str = '.', timeout: int = 120, allow_shell: bool = False) -> dict[str, Any]:
    blocked = ('rm -rf','shutdown','reboot','mkfs','dd ','sudo ','curl |','wget -O-')
    if not allow_shell and any(x in cmd for x in blocked):
        raise PermissionError('risky shell blocked; explicit allow_shell=true required')
    try:
        p = subprocess.run(['/bin/bash','-lc',cmd], cwd=str(safe_path(cwd)), text=True,
                           capture_output=True, timeout=timeout, check=False)
        return {'returncode': p.returncode, 'stdout': p.stdout[-20000:], 'stderr': p.stderr[-20000:], 'command': cmd}
    except subprocess.TimeoutExpired as exc:
        return {'returncode': 124, 'stdout': (exc.stdout or '')[-20000:] if isinstance(exc.stdout,str) else '',
                'stderr': ((exc.stderr or '') if isinstance(exc.stderr,str) else '')[-20000:] + '\nTIMEOUT', 'command': cmd}


def tree(path='.', limit=4000):
    r = safe_path(path); out = []
    for p in r.rglob('*'):
        if p.is_file() and '.git' not in p.parts and '.venv' not in p.parts and '__pycache__' not in p.parts:
            out.append(str(p.relative_to(ROOT)))
            if len(out) >= limit:
                break
    return {'root': str(r.relative_to(ROOT)), 'file_count': len(out), 'files': sorted(out), 'truncated': len(out) >= limit}


def analyze(path='.'):
    r = safe_path(path); exts = {}; tests = []
    for p in r.rglob('*'):
        if not p.is_file() or '.git' in p.parts or '.venv' in p.parts or '__pycache__' in p.parts:
            continue
        if p.suffix in SAFE_EXTS:
            exts[p.suffix] = exts.get(p.suffix, 0) + 1
        if 'test' in p.name.lower() or p.parent.name.lower() in {'test','tests'}:
            tests.append(str(p.relative_to(ROOT)))
    configs = [x for x in ['requirements.txt','pyproject.toml','package.json','Cargo.toml','go.mod','pom.xml','build.gradle','Dockerfile','Makefile','pytest.ini','tox.ini'] if (r/x).exists()]
    frameworks = []
    if (r/'pyproject.toml').exists() or (r/'requirements.txt').exists(): frameworks.append('python')
    if (r/'package.json').exists(): frameworks.append('javascript/typescript')
    if (r/'Cargo.toml').exists(): frameworks.append('rust')
    if (r/'go.mod').exists(): frameworks.append('go')
    if (r/'pom.xml').exists() or (r/'build.gradle').exists(): frameworks.append('java')
    if (r/'Dockerfile').exists(): frameworks.append('docker')
    return {'project': str(r.relative_to(ROOT)), 'tree': tree(path), 'languages': exts, 'framework_hints': frameworks,
            'configs': configs, 'tests': tests[:200], 'context': project_context(path), 'git': git_status(path)}


def read(path):
    p = safe_path(path); s = p.read_text(errors='replace')
    return {'path': str(p.relative_to(ROOT)), 'size_bytes': p.stat().st_size, 'content': s[:MAX_READ], 'truncated': len(s) > MAX_READ}


def search(query: str, path='.', limit=100, regex: bool = False, case_sensitive: bool = False):
    r = safe_path(path); out = []
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query, flags) if regex else None
    needle = query if case_sensitive else query.casefold()
    for p in r.rglob('*'):
        if not p.is_file() or '.git' in p.parts or '.venv' in p.parts or '__pycache__' in p.parts:
            continue
        try:
            lines = p.read_text(errors='replace').splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            matched = bool(pattern.search(line)) if pattern else (needle in (line if case_sensitive else line.casefold()))
            if matched:
                out.append({'path': str(p.relative_to(ROOT)), 'line': i, 'text': line[:500]})
                if len(out) >= limit:
                    return {'query': query, 'regex': regex, 'matches': out, 'count': len(out), 'truncated': True}
    return {'query': query, 'regex': regex, 'matches': out, 'count': len(out), 'truncated': False}


def snapshot(paths: Iterable[str], label='task') -> dict[str, Any]:
    selected = scope_check(paths, paths)['paths'] if paths else []
    stamp = f'{int(time.time()*1000)}-{hashlib.sha1(label.encode()).hexdigest()[:8]}'
    dest = BACKUPS / stamp
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel in selected:
        src = safe_path(rel, missing=True)
        target = dest / rel
        if src.exists() and src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            manifest.append({'path': rel, 'exists': True})
        else:
            manifest.append({'path': rel, 'exists': False})
    (dest/'manifest.json').write_text(json.dumps({'label': label, 'files': manifest}, indent=2))
    return {'ok': True, 'snapshot': str(dest.relative_to(ROOT)), 'files': manifest}


def restore_snapshot(snapshot_path: str) -> dict[str, Any]:
    base = safe_path(snapshot_path)
    manifest = json.loads((base/'manifest.json').read_text())
    restored = []
    for item in manifest['files']:
        target = safe_path(item['path'], missing=True)
        backup = base / item['path']
        if item['exists']:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
            restored.append(item['path'])
        elif target.exists():
            target.unlink()
            restored.append(item['path'])
    return {'ok': True, 'restored': restored, 'snapshot': str(base.relative_to(ROOT))}


def _diff_text(root: Path) -> str:
    if not (root/'.git').exists():
        return ''
    return run('git diff --no-ext-diff --unified=3 -- .', str(root), 60, True)['stdout']


def guardrail_check(path='.', declared_scope: Optional[Iterable[str]] = None,
                    allow_test_changes: bool = False, allow_security_changes: bool = False) -> dict[str, Any]:
    root = safe_path(path)
    git = _git_root(path)
    if git is None:
        return {'ok': True, 'git_available': False, 'violations': []}
    status = run('git status --short', str(git), 60, True)['stdout'].splitlines()
    changed = []
    for line in status:
        raw = line[3:] if len(line) > 3 else line
        if ' -> ' in raw:
            raw = raw.split(' -> ',1)[-1]
        if raw:
            changed.append(raw)
    violations = []
    if declared_scope:
        try:
            scope_check([str(git / rel) for rel in changed], declared_scope)
        except Exception as exc:
            violations.append(str(exc))
    for rel in changed:
        low = rel.lower()
        if not allow_security_changes and low in {'computer/security.py','computer/cleanup.py'}:
            violations.append(f'security/cleanup file change requires allow_security_changes=true: {rel}')
    diff = _diff_text(git)
    if not allow_test_changes:
        for line in diff.splitlines():
            if line.startswith('-') and not line.startswith('---') and re.search(r'\b(assert|pytest|ALL=True|selftest)\b', line, re.I):
                violations.append('possible test weakening detected in diff')
                break
    if 'computer/server.py' in changed and not allow_security_changes:
        for line in diff.splitlines():
            if line.startswith('-') and re.search(r'(RISKY_ACTIONS|mcp_auth_middleware|Confirmation required)', line):
                violations.append('possible security guard removal detected in server.py diff')
                break
    return {'ok': not violations, 'git_available': True, 'changed_paths': changed, 'violations': sorted(set(violations))}


def diff_summary(path='.', declared_scope: Optional[Iterable[str]] = None,
                 allow_test_changes=False, allow_security_changes=False) -> dict[str, Any]:
    git = _git_root(path)
    if git is None:
        return {'available': False, 'summary': 'No Git repository found'}
    guard = guardrail_check(path, declared_scope, allow_test_changes, allow_security_changes)
    diff = _diff_text(git)
    files = run('git diff --name-status', str(git), 60, True)['stdout'].splitlines()
    additions = len([x for x in diff.splitlines() if x.startswith('+') and not x.startswith('+++')])
    deletions = len([x for x in diff.splitlines() if x.startswith('-') and not x.startswith('---')])
    names = [x for x in files if x]
    human = f'{len(names)} file(s) changed; {additions} additions; {deletions} deletions.'
    if names:
        human += ' Paths: ' + ', '.join(names)
    return {'available': True, 'summary': human, 'files': names, 'additions': additions, 'deletions': deletions,
            'guardrails': guard, 'diff': diff[-50000:]}


def write(path, content, declared_scope: Optional[Iterable[str]] = None):
    if declared_scope is not None:
        scope_check([path], declared_scope)
    p = safe_path(path, True); p.parent.mkdir(parents=True, exist_ok=True); backup = None
    if p.exists():
        backup = BACKUPS / f'{int(time.time()*1000)}-{p.name}.bak'
        backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, backup)
    p.write_text(content)
    return {'ok': True, 'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'backup': str(backup.relative_to(ROOT)) if backup else None}


def patch(path, old, new, replace_all=False, declared_scope: Optional[Iterable[str]] = None):
    if declared_scope is not None:
        scope_check([path], declared_scope)
    p = safe_path(path); s = p.read_text(errors='replace'); c = s.count(old)
    if c == 0: raise ValueError('patch target not found')
    if c != 1 and not replace_all: raise ValueError(f'patch target occurs {c} times')
    return write(path, s.replace(old, new, -1 if replace_all else 1), declared_scope=declared_scope)


def test(command='pytest -q', cwd='.', timeout=170): return run(command, cwd, timeout, True)
def build(command='python -m compileall -q .', cwd='.', timeout=170): return run(command, cwd, timeout, True)
def lint(command='python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")', cwd='.', timeout=170): return run(command, cwd, timeout, True)
def shell(command, cwd='.', timeout=120, allow_shell=False): return run(command, cwd, timeout, allow_shell)


def git_status(path='.'):
    r = safe_path(path); g = _git_root(path)
    if g is None: return {'available': False}
    return {'available': True, **run('git status --short --branch', str(g), 60, True)}


def git_diff(path='.'):
    g = _git_root(path)
    return diff_summary(path) if g else {'available': False}


def git_log(path='.', n=10):
    g = _git_root(path)
    return run(f'git log -n {int(n)} --oneline --decorate', str(g), 60, True) if g else {'available': False}


def git_commit(message, path='.', declared_scope: Optional[Iterable[str]] = None,
               allow_test_changes=False, allow_security_changes=False):
    if not message.strip(): raise ValueError('commit message required')
    g = _git_root(path)
    if g is None: raise RuntimeError('not a git repository')
    summary = diff_summary(path, declared_scope, allow_test_changes, allow_security_changes)
    if not summary.get('guardrails',{}).get('ok', True):
        raise PermissionError('; '.join(summary['guardrails']['violations']))
    if declared_scope:
        run('git reset -- .', str(g), 60, True)
        for rel in summary['files']:
            rel_path = rel.split('\t')[-1]
            scope_check([str(g / rel_path)], declared_scope)
        for rel in summary['files']:
            rel_path = rel.split('\t')[-1]
            run(f'git add -- {shlex.quote(rel_path)}', str(g), 60, True)
    else:
        run('git add -A', str(g), 60, True)
    staged = run('git diff --cached --name-only', str(g), 60, True)['stdout'].splitlines()
    if not staged:
        return {'ok': True, 'committed': False, 'summary': summary, 'message': 'nothing to commit'}
    result = run(f'git commit -m {shlex.quote(message)}', str(g), 120, True)
    return {'ok': result['returncode'] == 0, 'committed': result['returncode'] == 0,
            'summary': summary, 'staged': staged, 'commit_output': result,
            'commit_sha': run('git rev-parse HEAD', str(g), 30, True)['stdout'].strip() if result['returncode'] == 0 else None}
