from __future__ import annotations
import shutil, uuid
from pathlib import Path
from coding import ROOT, safe_path
from .store import load_json, save_json, now

class TransactionEngine:
    def __init__(self): self.state=load_json("transactions.json", {"transactions":{}})
    def begin(self, paths:list[str], label="task"):
        tid=uuid.uuid4().hex[:12]; base=ROOT/".ai"/"control_plane"/"transactions"/tid; base.mkdir(parents=True,exist_ok=True)
        files=[]
        for raw in paths:
            p=safe_path(raw, missing=True); rel=p.relative_to(ROOT); target=base/rel; target.parent.mkdir(parents=True,exist_ok=True)
            if p.exists(): shutil.copy2(p,target); exists=True
            else: exists=False
            files.append({"path":str(rel),"exists":exists})
        row={"id":tid,"label":label,"status":"active","created_at":now(),"updated_at":now(),"files":files,"steps":[]}
        self.state["transactions"][tid]=row; save_json("transactions.json",self.state); return row
    def step(self, tid,label,note="",tool=None,input_data=None,result=None,error=None):
        row=self.state["transactions"][tid]
        row["steps"].append({"label":label,"note":note,"tool":tool,"input":input_data,"result":result,"error":error,"timestamp":now()})
        row["updated_at"]=now(); save_json("transactions.json",self.state); return row
    def commit(self,tid):
        row=self.state["transactions"][tid]; row["status"]="committed"; row["updated_at"]=now(); save_json("transactions.json",self.state); return row
    def rollback(self,tid):
        row=self.state["transactions"][tid]
        if row.get("status") != "active":
            raise ValueError(f"cannot rollback transaction in status {row.get('status')}")
        base=ROOT/".ai"/"control_plane"/"transactions"/tid
        for f in row["files"]:
            p=ROOT/f["path"]; snap=base/f["path"]
            if f["exists"]:
                p.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(snap,p)
            elif p.exists():
                if p.is_dir(): shutil.rmtree(p)
                else: p.unlink()
        row["status"]="rolled_back"; row["updated_at"]=now(); save_json("transactions.json",self.state); return row
    def read(self,tid=None): return self.state["transactions"].get(tid) if tid else self.state
