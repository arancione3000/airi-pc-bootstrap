from __future__ import annotations
import json, os, subprocess, uuid
from pathlib import Path
from .store import load_json, save_json, now
from coding import ROOT

FILE='subagents.json'
class SubagentManager:
    def __init__(self): self.state=load_json(FILE, {'version':1,'agents':{}})
    def _repo_path(self, path):
        q=Path(path)
        q=(ROOT/q).resolve() if not q.is_absolute() else q.resolve()
        try: q.relative_to(ROOT.resolve())
        except ValueError as exc: raise ValueError('repository outside Airi-PC workspace') from exc
        if not (q/'.git').exists(): raise ValueError('git repository required')
        return q

    def _git(self,args,cwd):
        p=subprocess.run(['git',*args],cwd=str(cwd),text=True,capture_output=True,timeout=60,check=False)
        return p
    def create(self, goal, repo_path='.', branch_prefix='agent'):
        repo=self._repo_path(repo_path); branch=f"{branch_prefix}/{uuid.uuid4().hex[:8]}"; worktrees=repo/'.ai'/'worktrees'; worktrees.mkdir(parents=True,exist_ok=True); wt=worktrees/branch.replace('/','-')
        p=self._git(['worktree','add','-b',branch,str(wt),'HEAD'],repo)
        if p.returncode!=0: raise RuntimeError(p.stderr.strip() or 'git worktree creation failed')
        aid=uuid.uuid4().hex[:12]; row={'id':aid,'goal':goal,'branch':branch,'worktree':str(wt),'repo':str(repo),'status':'ready','created_at':now(),'updated_at':now()}
        self.state['agents'][aid]=row; save_json(FILE,self.state); return row
    def status(self, agent_id): return self.state['agents'][agent_id]
    def list(self): return {'count':len(self.state['agents']),'agents':list(self.state['agents'].values())}
    def finish(self, agent_id, status='ready'):
        row=self.state['agents'][agent_id]; row['status']=status; row['updated_at']=now(); save_json(FILE,self.state); return row
    def remove(self, agent_id, force=False):
        row=self.state['agents'][agent_id]; repo=Path(row['repo']); wt=Path(row['worktree']); args=['worktree','remove'];
        if force: args.append('--force')
        args.append(str(wt)); p=self._git(args,repo)
        if p.returncode!=0: raise RuntimeError(p.stderr.strip() or 'git worktree remove failed')
        self.state['agents'].pop(agent_id,None); save_json(FILE,self.state); return {'ok':True,'id':agent_id}
