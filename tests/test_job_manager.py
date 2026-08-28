import os, time
from pathlib import Path

def test_background_job_persists_and_completes(tmp_path, monkeypatch):
    monkeypatch.setenv('AIRIPC_WORKSPACE_ROOT', str(tmp_path))
    repo=tmp_path/'.repo'; repo.mkdir(); (repo/'.git').mkdir()
    from control_plane import job_manager
    monkeypatch.setattr(job_manager, 'ROOT', tmp_path)
    monkeypatch.setattr(job_manager, 'LOG_DIR', tmp_path/'.ai'/'state'/'jobs')
    jm=job_manager.JobManager()
    row=jm.start("printf 'AIRI_JOB_OK\\n'; sleep 0.2; printf 'DONE\\n'", cwd='.', timeout=5, scope=['.'])
    assert row['id']
    time.sleep(0.6)
    out=jm.status(row['id'])
    assert out['status']=='completed'
    assert out['exit_code']==0
    assert 'DONE' in out['last_output']
    fresh=job_manager.JobManager()
    assert fresh.status(row['id'])['status']=='completed'

def test_cancel_background_job(tmp_path, monkeypatch):
    monkeypatch.setenv('AIRIPC_WORKSPACE_ROOT', str(tmp_path))
    from control_plane import job_manager
    monkeypatch.setattr(job_manager, 'ROOT', tmp_path)
    monkeypatch.setattr(job_manager, 'LOG_DIR', tmp_path/'.ai'/'state'/'jobs')
    jm=job_manager.JobManager()
    row=jm.start('sleep 30', cwd='.', timeout=60, scope=['.'])
    result=jm.cancel(row['id'])
    assert result['status']=='cancelled'
