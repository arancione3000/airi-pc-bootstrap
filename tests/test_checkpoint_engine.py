from __future__ import annotations
import json, multiprocessing as mp, os, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'computer'))
from control_plane.store import atomic_json, load_json
import advanced


def _writer(path, value):
    from control_plane.store import atomic_json
    for i in range(20):
        atomic_json(Path(path), {'writer': value, 'i': i})


def test_atomic_json_is_valid_and_leaves_no_partial_tmp(tmp_path):
    path=tmp_path/'checkpoint.json'
    atomic_json(path, {'a': 1, 'status': 'pending'})
    assert json.loads(path.read_text())['a'] == 1
    assert not list(tmp_path.glob('checkpoint.json.tmp.*'))


def test_atomic_json_is_multiprocess_safe(tmp_path):
    path=tmp_path/'checkpoint.json'
    procs=[mp.Process(target=_writer, args=(str(path), i)) for i in range(4)]
    for p in procs: p.start()
    for p in procs: p.join(10)
    assert all(p.exitcode == 0 for p in procs)
    data=json.loads(path.read_text())
    assert set(data) == {'writer','i'}
    assert not list(tmp_path.glob('checkpoint.json.tmp.*'))


def test_corrupt_checkpoint_fails_closed_and_preserves_evidence(tmp_path, monkeypatch):
    import control_plane.store as store
    monkeypatch.setattr(store, 'CP', tmp_path)
    path=tmp_path/'checkpoint.json'; path.write_text('{broken', encoding='utf-8')
    try:
        load_json('checkpoint.json', {})
    except RuntimeError:
        assert (tmp_path/'checkpoint.json.corrupt').exists()
    else:
        raise AssertionError('corrupt checkpoint silently accepted')


def test_checkpoint_has_resume_schema_and_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr(advanced, 'STATE', tmp_path)
    monkeypatch.setattr(advanced, 'RECOVERY', tmp_path/'checkpoint.json')
    row=advanced.checkpoint('task', ['x'], 1, 'started', status='active', task='task', phase='phase', last_verified_sha='abc')
    for key in ('task','phase','step','started_at','completed_at','status','error','retry_count','last_verified_sha'):
        assert key in row
    done=advanced.checkpoint('task', ['x'], 1, 'done', status='completed', completed_at=time.time(), last_verified_sha='abc')
    assert done['status']=='completed' and done['completed_at'] is not None and done['last_verified_sha']=='abc'
