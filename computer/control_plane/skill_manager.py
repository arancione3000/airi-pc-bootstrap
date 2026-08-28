from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from coding import ROOT
from .store import load_json,save_json,now
class SkillManager:
    def __init__(self): self.state=load_json('skill-registry.json', {'version':1,'skills':{}})
    def refresh(self):
        skills=ROOT/'skills'; skills.mkdir(exist_ok=True)
        for p in skills.iterdir():
            f=p/'SKILL.md'
            if not p.is_dir() or not f.exists(): continue
            text=f.read_text(errors='replace'); sha=hashlib.sha256(f.read_bytes()).hexdigest()
            tools=re.findall(r'^-\s+(.+)$', text, re.M);
            previous=self.state['skills'].get(p.name,{})
            ts=now()
            self.state['skills'][p.name]={'name':p.name,'version':previous.get('version','1.0'),'path':str(f.relative_to(ROOT)),'description':self._section(text,'Description'),'required_tools':tools,'permissions':['workspace-scoped'],'status':'valid','checksum':sha,'verified_at':ts,'last_verified':ts,'origin':previous.get('origin','repository'),'dependencies':previous.get('dependencies',tools),'tests':previous.get('tests',[]),'rollback':previous.get('rollback',{'strategy':'restore previous checksum'})}
        save_json('skill-registry.json',self.state); return self.state
    def _section(self,text,name):
        m=re.search(rf'## {re.escape(name)}\n(.*?)(?:\n## |\Z)',text,re.S); return (m.group(1).strip() if m else '')[:500]
    def list(self): return self.state
    def verify(self,name):
        row=self.state['skills'][name]; sha=hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(); row['status']='valid' if sha==row['checksum'] else 'modified'; row['current_checksum']=sha; save_json('skill-registry.json',self.state); return row
