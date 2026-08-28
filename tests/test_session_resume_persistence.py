from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'computer'))
import control_plane.task_engine as te
from control_plane.store import atomic_json


def test_task_resume_across_fresh_engine_instances(tmp_path, monkeypatch):
    state_file=tmp_path/'tasks.json'
    def load(name, default):
        if not state_file.exists(): return default
        return json.loads(state_file.read_text(encoding='utf-8'))
    def save(name, data): atomic_json(state_file, data)
    monkeypatch.setattr(te, 'load_json', load); monkeypatch.setattr(te, 'save_json', save)

    first=te.TaskEngine().start('resume-demo',['STEP 1','STEP 2','STEP 3'])
    e1=te.TaskEngine(); e1.update('n1','completed',output={'done':'step1'},task_id=first['id'])

    e2=te.TaskEngine(); row=e2.read(first['id'])
    assert row['nodes'][0]['status']=='completed'
    assert row['nodes'][1]['status']=='running'
    assert row['nodes'][2]['status']=='pending'
    e2.update('n2','completed',output={'done':'step2'},task_id=first['id'])

    e3=te.TaskEngine(); row=e3.read(first['id'])
    assert [n['status'] for n in row['nodes']]==['completed','completed','running']
    e3.update('n3','completed',output={'done':'step3'},task_id=first['id'])

    final=te.TaskEngine().read(first['id'])
    assert final['status']=='completed'
    assert [n['output'] for n in final['nodes']]==[{'done':'step1'},{'done':'step2'},{'done':'step3'}]
