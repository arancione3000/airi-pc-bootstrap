from __future__ import annotations
import re,hashlib,ast
from pathlib import Path
from coding import ROOT
from .store import load_json,save_json,now
class ProjectIndex:
    def __init__(self):
        self.state=load_json("project-index.json", {"version":2,"files":{},"symbols":{},"dependencies":{},"updated_at":None})
        self.state.setdefault('dependencies', {})
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
            try:
                stat=p.stat()
                prev=self.state['files'].get(rel)
                if prev and prev.get('mtime_ns')==stat.st_mtime_ns and prev.get('bytes')==stat.st_size:
                    continue
                data=p.read_text(errors='replace')
            except Exception:
                continue
            sha=hashlib.sha256(data.encode()).hexdigest()
            if prev and prev.get('sha256')==sha:
                prev.update({'mtime_ns': stat.st_mtime_ns, 'bytes': stat.st_size, 'updated_at': now()})
                continue
            self.state['files'][rel]={"sha256":sha,"bytes":len(data.encode()),"mtime_ns":p.stat().st_mtime_ns,"updated_at":now()}
            if p.suffix == '.py':
                try:
                    tree_ast=ast.parse(data)
                    imports=[n.names[0].name for n in ast.walk(tree_ast) if isinstance(n, ast.Import) and n.names] + [n.module for n in ast.walk(tree_ast) if isinstance(n, ast.ImportFrom) and n.module]
                except Exception:
                    imports=[]
            else: imports=[]
            syms=re.findall(r'^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)',data,re.M)
            self.state['symbols'][rel]=syms
            self.state.setdefault('dependencies', {})[rel]=sorted({x for x in imports if x})
        self.state['updated_at']=now(); save_json('project-index.json',self.state); return self.summary()
    def search(self,q,limit=50):
        q=q.casefold(); out=[]
        for path,syms in self.state['symbols'].items():
            for s in syms:
                if q in s.casefold() or q in path.casefold(): out.append({"path":path,"symbol":s})
                if len(out)>=limit: return out
        return out
    def context_pack(self, query: str, limit_files: int = 12, max_bytes: int = 120000):
        q=str(query).casefold().strip(); scored=[]
        for path, meta in self.state.get('files',{}).items():
            symbols=self.state.get('symbols',{}).get(path,[])
            hay=(path+' '+' '.join(symbols)).casefold()
            score=sum(2 for tok in q.split() if tok and tok in path.casefold()) + sum(1 for tok in q.split() if tok and tok in hay)
            if score: scored.append((score,path))
        scored.sort(key=lambda x:(-x[0],x[1])); selected=[p for _,p in scored[:max(1,int(limit_files))]]
        snippets=[]; total=0
        for rel in selected:
            p=ROOT/rel
            try: text=p.read_text(errors='replace')[:max(0,max_bytes-total)]
            except Exception: continue
            snippets.append({'path':rel,'score':next((s for s,r in scored if r==rel),0),'content':text}); total += len(text.encode())
            if total>=max_bytes: break
        return {'query':query,'files':snippets,'bytes':total,'truncated':total>=max_bytes}

    def summary(self): return {"files":len(self.state['files']),"symbol_files":len(self.state['symbols']),"dependency_files":len(self.state.get('dependencies',{})),"updated_at":self.state['updated_at']}
