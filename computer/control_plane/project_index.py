from __future__ import annotations
import re,hashlib
from pathlib import Path
from coding import ROOT
from .store import load_json,save_json,now
class ProjectIndex:
    def __init__(self): self.state=load_json("project-index.json", {"version":1,"files":{},"symbols":{},"updated_at":None})
    def refresh(self, paths=None):
        targets=[]
        if paths:
            for x in paths:
                p=(ROOT/x).resolve()
                if p.is_file(): targets.append(p)
                elif p.is_dir(): targets.extend(q for q in p.rglob('*') if q.is_file())
        else: targets=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '.venv' not in p.parts and '__pycache__' not in p.parts]
        for p in targets:
            rel=str(p.relative_to(ROOT))
            try: data=p.read_text(errors='replace')
            except Exception: continue
            sha=hashlib.sha256(data.encode()).hexdigest(); prev=self.state['files'].get(rel)
            if prev and prev.get('sha256')==sha: continue
            self.state['files'][rel]={"sha256":sha,"bytes":len(data.encode()),"updated_at":now()}
            syms=re.findall(r'^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)',data,re.M)
            self.state['symbols'][rel]=syms
        self.state['updated_at']=now(); save_json('project-index.json',self.state); return self.summary()
    def search(self,q,limit=50):
        q=q.casefold(); out=[]
        for path,syms in self.state['symbols'].items():
            for s in syms:
                if q in s.casefold() or q in path.casefold(): out.append({"path":path,"symbol":s})
                if len(out)>=limit: return out
        return out
    def summary(self): return {"files":len(self.state['files']),"symbol_files":len(self.state['symbols']),"updated_at":self.state['updated_at']}
