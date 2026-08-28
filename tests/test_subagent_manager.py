from pathlib import Path

def test_subagent_worktree_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv('AIRIPC_WORKSPACE_ROOT', str(tmp_path))
    repo=tmp_path/'repo'; repo.mkdir();
    import subprocess
    subprocess.run(['git','init','-q'],cwd=repo,check=True)
    subprocess.run(['git','config','user.name','Test'],cwd=repo,check=True)
    subprocess.run(['git','config','user.email','test@example.com'],cwd=repo,check=True)
    (repo/'README.md').write_text('x')
    subprocess.run(['git','add','.'],cwd=repo,check=True); subprocess.run(['git','commit','-qm','base'],cwd=repo,check=True)
    from control_plane import subagent_manager
    monkeypatch.setattr(subagent_manager,'ROOT',tmp_path)
    m=subagent_manager.SubagentManager(); row=m.create('isolated coding goal','repo')
    assert Path(row['worktree']).is_dir(); assert row['branch'].startswith('agent/')
    assert m.status(row['id'])['status']=='ready'
    m.remove(row['id']); assert not Path(row['worktree']).exists()
