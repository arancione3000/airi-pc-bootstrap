from __future__ import annotations
import json, os, re, subprocess, threading, time, urllib.parse, urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get('AIRIPC_WORKSPACE_ROOT', '/home/user/airi')).resolve()
AI = ROOT / '.ai'
STATE = AI / 'state'
RECOVERY = STATE / 'recovery.json'
DECISIONS = STATE / 'decisions.jsonl'
SCHEDULER = STATE / 'scheduler.json'
LOCK = threading.RLock()


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def checkpoint(goal: str, scope: list[str], step: int, note: str = '', artifacts: list[str] | None = None, status: str = 'active') -> dict[str, Any]:
    with LOCK:
        data = {
            'version': 1, 'goal': goal, 'scope': list(scope), 'step': int(step), 'note': note,
            'artifacts': list(artifacts or []), 'status': status, 'updated_at': time.time(),
        }
        _atomic_json(RECOVERY, data)
        return data


def recovery_read() -> dict[str, Any]:
    with LOCK:
        if not RECOVERY.exists():
            return {'active': False, 'checkpoint': None}
        data = json.loads(RECOVERY.read_text(encoding='utf-8'))
        return {'active': data.get('status') == 'active', 'checkpoint': data}


def recovery_finish(status: str = 'done', note: str = '') -> dict[str, Any]:
    cur = recovery_read().get('checkpoint') or {}
    cur.update({'status': status, 'note': note or cur.get('note', ''), 'updated_at': time.time()})
    _atomic_json(RECOVERY, cur)
    return cur


def record_decision(decision: str, reason: str = '', evidence: list[Any] | None = None,
                    files: list[str] | None = None, commit: str | None = None, result: str = '') -> dict[str, Any]:
    row = {
        'timestamp': time.time(), 'decision': str(decision), 'reason': str(reason),
        'evidence': list(evidence or []), 'files': list(files or []), 'commit': commit, 'result': str(result),
    }
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


def decisions(limit: int = 50) -> dict[str, Any]:
    if not DECISIONS.exists(): return {'count': 0, 'decisions': []}
    rows = []
    for line in DECISIONS.read_text(encoding='utf-8').splitlines()[-max(1, int(limit)):]:
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: continue
    return {'count': len(rows), 'decisions': rows}


def _git(cmd: str, cwd: Path) -> dict[str, Any]:
    p = subprocess.run(['/bin/bash', '-lc', cmd], cwd=str(cwd), text=True, capture_output=True, timeout=60)
    return {'returncode': p.returncode, 'stdout': p.stdout[-12000:], 'stderr': p.stderr[-12000:]}


def _git_root(path: Path | None = None) -> Path | None:
    p = (path or ROOT).resolve()
    while p == ROOT or ROOT in p.parents:
        if (p / '.git').exists(): return p
        if p == ROOT: break
        p = p.parent
    return None


def persistence_status() -> dict[str, Any]:
    repo = _git_root(ROOT)
    if repo is None:
        return {'persistent': False, 'reason': 'git repository unavailable', 'repo': str(ROOT)}
    head = _git('git rev-parse HEAD', repo)
    branch = _git('git branch --show-current', repo)
    remote = _git('git remote get-url origin', repo)
    if head['returncode'] != 0 or remote['returncode'] != 0:
        return {'persistent': False, 'reason': 'git HEAD or origin unavailable', 'repo': str(repo)}
    remote_url = remote['stdout'].strip()
    current_branch = branch['stdout'].strip() or 'main'
    canonical_branch = os.environ.get('AIRI_CANONICAL_BRANCH', 'main').strip() or 'main'
    ls = _git('git ls-remote origin refs/heads/' + canonical_branch, repo)
    remote_sha = ls['stdout'].split()[0] if ls['returncode'] == 0 and ls['stdout'].split() else None
    local_sha = head['stdout'].strip()
    persistent = bool(current_branch == canonical_branch and remote_sha and remote_sha == local_sha)
    if persistent:
        reason = 'verified'
    elif current_branch != canonical_branch:
        reason = 'local branch is not canonical'
    else:
        reason = 'remote verification failed' if remote_sha else ls['stderr'].strip() or 'remote unavailable'
    return {'persistent': persistent, 'repo': str(repo), 'branch': current_branch,
            'canonical_branch': canonical_branch, 'local_sha': local_sha, 'remote_sha': remote_sha,
            'origin': remote_url, 'reason': reason}


def persist_current(message: str, branch: str | None = None, push: bool = True, scope: list[str] | None = None) -> dict[str, Any]:
    repo = _git_root(ROOT)
    if repo is None: return {'ok': False, 'persistent': False, 'reason': 'git repository unavailable'}
    status = _git('git status --short', repo)
    if status['returncode'] != 0: return {'ok': False, 'persistent': False, 'reason': status['stderr']}
    if not status['stdout'].strip(): return {**persistence_status(), 'ok': True, 'committed': False, 'message': 'nothing to commit'}
    if scope:
        _git('git reset -- .', repo)
        for rel in scope:
            rel = str(rel).lstrip('/')
            if '..' in Path(rel).parts: return {'ok': False, 'persistent': False, 'reason': 'invalid scope path'}
            _git('git add -- ' + subprocess.list2cmdline([rel]), repo)
    else:
        return {'ok': False, 'persistent': False, 'reason': 'scope is required for persistence'}
    commit = _git("git commit -m " + __import__('shlex').quote(message), repo)
    if commit['returncode'] != 0: return {'ok': False, 'persistent': False, 'reason': commit['stderr'], 'commit': commit}
    sha = _git('git rev-parse HEAD', repo)['stdout'].strip()
    if push:
        target = branch or _git('git branch --show-current', repo)['stdout'].strip() or 'main'
        pushed = _git('git push origin HEAD:' + target, repo)
        if pushed['returncode'] != 0:
            return {'ok': False, 'persistent': False, 'local_commit': sha, 'reason': 'push failed', 'push': pushed}
    verified = persistence_status()
    verified.update({'ok': verified.get('persistent') is True, 'committed': True, 'local_commit': sha})
    record_decision('persisted change', reason='atomic local commit followed by remote verification', commit=sha,
                    files=[x.strip() for x in status['stdout'].splitlines() if x.strip()], result=verified.get('reason', ''))
    return verified


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.bits: list[str] = []; self.title_bits: list[str] = []; self.in_title = False
    def handle_starttag(self, tag, attrs): self.in_title = tag.lower() == 'title' or self.in_title
    def handle_endtag(self, tag):
        if tag.lower() == 'title': self.in_title = False
        self.bits.append('\n')
    def handle_data(self, data):
        s = re.sub(r'\s+', ' ', data).strip()
        if not s: return
        if self.in_title: self.title_bits.append(s)
        self.bits.append(s)


def _fetch(url: str, timeout: int = 20) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {'http', 'https'}: raise ValueError('only http/https URLs are allowed')
    req = urllib.request.Request(url, headers={'User-Agent': 'Airi-PC-Research/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(2_000_000)
        charset = r.headers.get_content_charset() or 'utf-8'
        text = raw.decode(charset, errors='replace')
        parser = _TextParser(); parser.feed(text)
        body = re.sub(r'\n{3,}', '\n\n', '\n'.join(parser.bits)).strip()
        return {'url': r.geturl(), 'title': ' '.join(parser.title_bits), 'text': body[:12000]}


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.items=[]; self._href=None; self._is_result=False; self._text=[]
    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a': return
        d=dict(attrs); href=d.get('href',''); cls=d.get('class','') or ''
        if href and ('result__a' in cls or 'result-link' in cls):
            self._href=href; self._is_result=True; self._text=[]
    def handle_data(self, data):
        if self._is_result: self._text.append(data)
    def handle_endtag(self, tag):
        if tag.lower() == 'a' and self._is_result:
            href=urllib.parse.unquote(self._href or '')
            m=re.search(r'uddg=([^&]+)', href)
            if m: href=urllib.parse.unquote(m.group(1))
            if href.startswith(('http://','https://')):
                self.items.append({'url':href,'title':re.sub(r'\s+',' ',' '.join(self._text)).strip()})
            self._href=None; self._is_result=False; self._text=[]


def _search_web(query: str, limit: int) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f'https://html.duckduckgo.com/html/?q={q}'
    req=urllib.request.Request(url,headers={'User-Agent':'Airi-PC-Research/1.0'})
    with urllib.request.urlopen(req,timeout=20) as r:
        raw=r.read(2_000_000).decode(r.headers.get_content_charset() or 'utf-8',errors='replace')
    parser=_SearchParser(); parser.feed(raw)
    out=[]; seen=set()
    for item in parser.items:
        u=item['url']
        if 'duckduckgo.com' in urllib.parse.urlparse(u).netloc: continue
        if u in seen: continue
        seen.add(u); out.append(item)
        if len(out)>=limit: break
    return out


def research(topic: str, urls: list[str] | None = None, max_sources: int = 5) -> dict[str, Any]:
    if not str(topic).strip(): raise ValueError('topic is required')
    max_sources = max(1, min(int(max_sources), 10))
    targets = [{'url': u} for u in (urls or [])[:max_sources]]
    if not targets: targets = _search_web(topic, max_sources)
    sources, errors = [], []
    for target in targets:
        try:
            page = _fetch(target['url'])
            excerpt = page['text'][:3500]
            sources.append({'url': page['url'], 'title': page['title'], 'excerpt': excerpt, 'text_length': len(page['text'])})
        except Exception as exc:
            errors.append({'url': target['url'], 'error': str(exc)})
    common = set(re.findall(r'\b[a-zA-Z]{5,}\b', ' '.join(x['excerpt'].lower() for x in sources)))
    evidence = [{'source': x['url'], 'title': x['title'], 'excerpt': x['excerpt']} for x in sources]
    result = {
        'ok': bool(sources), 'topic': topic, 'sources': sources, 'errors': errors,
        'source_count': len(sources), 'evidence': evidence,
        'method': 'multi-source web fetch with explicit source URLs; search via DuckDuckGo HTML when URLs omitted',
        'shared_terms': sorted(common)[:100],
        'limitations': ['The tool returns evidence; semantic synthesis should be performed by the calling agent.',
                        'Sites that block automated access may appear in errors.'],
    }
    record_decision('research completed', reason=topic, evidence=evidence, result=f'{len(sources)} sources')
    return result


_jobs: dict[str, dict[str, Any]] = {}
_scheduler_started = False

def _load_jobs():
    global _jobs
    try: _jobs = json.loads(SCHEDULER.read_text(encoding='utf-8')) if SCHEDULER.exists() else {}
    except Exception: _jobs = {}


def _save_jobs(): _atomic_json(SCHEDULER, _jobs)


def _scheduler_loop():
    while True:
        now = time.time()
        changed = False
        for job in list(_jobs.values()):
            if not job.get('enabled', True) or now < job.get('next_run', now): continue
            action = job.get('action')
            try:
                if action == 'health': result = health_check()
                elif action == 'cleanup_scan':
                    from cleanup import scan
                    result = scan()
                elif action == 'persistence_verify': result = persistence_status()
                else: raise ValueError(f'unsupported scheduled action: {action}')
                job['last_result'] = result; job['last_error'] = None
            except Exception as exc:
                job['last_error'] = str(exc)
            job['last_run'] = now; job['next_run'] = now + int(job['interval_seconds']); changed = True
        if changed:
            try: _save_jobs()
            except Exception: pass
        time.sleep(1)


def scheduler_start():
    global _scheduler_started
    with LOCK:
        _load_jobs()
        if _scheduler_started: return
        _scheduler_started = True
        threading.Thread(target=_scheduler_loop, name='airi-scheduler', daemon=True).start()


def schedule_job(name: str, action: str, interval_seconds: int, run_now: bool = False) -> dict[str, Any]:
    scheduler_start()
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', name): raise ValueError('invalid job name')
    if action not in {'health', 'cleanup_scan', 'persistence_verify'}: raise ValueError('unsupported scheduled action')
    if int(interval_seconds) < 5: raise ValueError('interval_seconds must be >= 5')
    _jobs[name] = {'name': name, 'action': action, 'interval_seconds': int(interval_seconds), 'enabled': True,
                   'created_at': time.time(), 'next_run': time.time() if run_now else time.time() + int(interval_seconds),
                   'last_run': None, 'last_error': None}
    _save_jobs(); return _jobs[name]


def cancel_job(name: str) -> dict[str, Any]:
    scheduler_start()
    if name not in _jobs: raise KeyError(name)
    job = _jobs.pop(name); _save_jobs(); return {'ok': True, 'cancelled': job}


def scheduler_status() -> dict[str, Any]:
    scheduler_start(); return {'running': _scheduler_started, 'jobs': list(_jobs.values())}


def health_check() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        checks['gui'] = {'ok': bool(os.environ.get('DISPLAY')), 'display': os.environ.get('DISPLAY', '')}
    except Exception as exc: checks['gui'] = {'ok': False, 'error': str(exc)}
    try:
        import coding
        checks['workspace'] = {'ok': coding.ROOT.exists(), 'path': str(coding.ROOT)}
    except Exception as exc: checks['workspace'] = {'ok': False, 'error': str(exc)}
    try:
        checks['persistence'] = persistence_status()
    except Exception as exc: checks['persistence'] = {'persistent': False, 'reason': str(exc)}
    try:
        from skills import list_skills, memory_read, task_read
        checks['skills'] = list_skills(); checks['memory'] = {'ok': True, 'bytes': memory_read().get('size_bytes', 0)}; checks['recovery'] = recovery_read(); checks['task'] = task_read()
    except Exception as exc: checks['skills'] = {'count': 0, 'error': str(exc)}
    try:
        from cleanup import scan
        d = scan()['disk']; checks['disk'] = {'ok': True, 'free_gb': d['free_gb'], 'used_percent': d['used_percent']}
    except Exception as exc: checks['disk'] = {'ok': False, 'error': str(exc)}
    try: checks['scheduler'] = scheduler_status()
    except Exception as exc: checks['scheduler'] = {'running': False, 'error': str(exc)}
    checks['backup_state_dir'] = {'ok': STATE.exists(), 'path': str(STATE)}
    essentials = ['gui', 'workspace', 'persistence', 'skills', 'disk', 'scheduler']
    ok = all(bool(checks.get(k, {}).get('ok', checks.get(k, {}).get('persistent', False) if k == 'persistence' else True)) for k in essentials)
    checks['overall_ok'] = ok
    return checks


# Start the scheduler only when the module is imported by the live Airi server.
scheduler_start()
