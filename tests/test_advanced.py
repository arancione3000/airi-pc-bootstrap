from __future__ import annotations
import json, os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / 'computer'))
import advanced


def test_recovery_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(advanced, 'STATE', tmp_path / 'state')
    monkeypatch.setattr(advanced, 'RECOVERY', advanced.STATE / 'recovery.json')
    item = advanced.checkpoint('goal', ['x.py'], 2, 'step two', ['x.py'])
    assert item['step'] == 2
    assert advanced.recovery_read()['checkpoint']['goal'] == 'goal'
    done = advanced.recovery_finish('done', 'complete')
    assert done['status'] == 'done'
    assert advanced.recovery_read()['active'] is False


def test_decision_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(advanced, 'DECISIONS', tmp_path / 'decisions.jsonl')
    row = advanced.record_decision('use X', 'Y', evidence=['source'], files=['a.py'], commit='abc', result='ok')
    assert row['commit'] == 'abc'
    got = advanced.decisions(10)
    assert got['count'] == 1 and got['decisions'][0]['decision'] == 'use X'


def test_research_local_http(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'<html><title>Test Source</title><p>Evidence alpha beta gamma.</p></html>'
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *_): pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    monkeypatch.setattr(advanced, 'DECISIONS', tmp_path / 'decisions.jsonl')
    try:
        result = advanced.research('alpha', [f'http://127.0.0.1:{server.server_port}/'], 2)
        assert result['ok'] and result['source_count'] == 1
        assert result['sources'][0]['title'] == 'Test Source'
        assert 'Evidence alpha beta gamma.' in result['sources'][0]['excerpt']
    finally:
        server.shutdown()


def test_scheduler_rejects_unsafe_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(advanced, 'SCHEDULER', tmp_path / 'scheduler.json')
    monkeypatch.setattr(advanced, '_scheduler_started', False)
    monkeypatch.setattr(advanced, '_jobs', {})
    try:
        advanced.schedule_job('health', 'health', 5)
        assert advanced.scheduler_status()['jobs'][0]['action'] == 'health'
        try:
            advanced.schedule_job('bad', 'shell', 5)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe scheduler action accepted')
    finally:
        advanced.cancel_job('health')


def test_persistence_without_git_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(advanced, 'ROOT', tmp_path)
    assert advanced.persistence_status()['persistent'] is False


def test_persistence_success_with_bare_remote(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; remote = tmp_path / 'remote.git'
    repo.mkdir();
    import subprocess
    subprocess.run(['git','init','-q',str(remote),'--bare'], check=True)
    subprocess.run(['git','init','-q',str(repo)], check=True)
    subprocess.run(['git','-C',str(repo),'config','user.name','Test'], check=True)
    subprocess.run(['git','-C',str(repo),'config','user.email','test@example.com'], check=True)
    (repo/'AGENTS.md').write_text('# ctx\n'); (repo/'x.txt').write_text('one\n')
    subprocess.run(['git','-C',str(repo),'add','.'], check=True); subprocess.run(['git','-C',str(repo),'commit','-qm','base'], check=True)
    subprocess.run(['git','-C',str(repo),'branch','-M','main'], check=True); subprocess.run(['git','-C',str(repo),'remote','add','origin',str(remote)], check=True)
    subprocess.run(['git','-C',str(repo),'push','-qu','origin','main'], check=True)
    monkeypatch.setattr(advanced, 'ROOT', repo); monkeypatch.setattr(advanced, 'AI', repo/'.ai'); monkeypatch.setattr(advanced, 'STATE', advanced.AI/'state'); monkeypatch.setattr(advanced, 'DECISIONS', advanced.STATE/'decisions.jsonl')
    (repo/'x.txt').write_text('two\n')
    got=advanced.persist_current('feat: persist test',scope=['x.txt'],push=True)
    assert got['ok'] is True and got['persistent'] is True and got['local_commit']==got['remote_sha']


def test_prune_backups_removes_only_old_excess(tmp_path, monkeypatch):
    import cleanup
    monkeypatch.setattr(cleanup, 'HOME', tmp_path)
    root=tmp_path/'airi'/'.ai'/'backups'
    root.mkdir(parents=True)
    import time, os
    for i in range(3):
        d=root/f'b{i}'; d.mkdir(); (d/'manifest.json').write_text('{}')
        ts=time.time()-(40-i)*86400; os.utime(d,(ts,ts))
    result=cleanup.prune_backups(max_entries=1,max_age_days=30,dry_run=False)
    assert result['candidate_count']==2
    assert (root/'b2').exists() and not (root/'b0').exists() and not (root/'b1').exists()

def test_persistent_store_does_not_silently_reset_corrupt_json(tmp_path, monkeypatch):
    import control_plane.store as store
    monkeypatch.setattr(store, 'CP', tmp_path)
    p = tmp_path / 'broken.json'
    p.write_text('{broken', encoding='utf-8')
    try:
        store.load_json('broken.json', {'fallback': True})
    except RuntimeError:
        assert (tmp_path / 'broken.json.corrupt').exists()
    else:
        raise AssertionError('corrupt state was silently replaced by default')
