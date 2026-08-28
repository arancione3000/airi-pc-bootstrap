from __future__ import annotations
import uuid
from .store import load_json, save_json, now, redact

FILE='experience.json'
class ExperienceStore:
    def __init__(self): self.state=load_json(FILE, {'version':1,'experiences':{}})
    def record(self, problem, context, solution, tools=None, failure_modes=None, successful_strategy='', verification=None, tags=None):
        eid=uuid.uuid4().hex[:12]
        row={'id':eid,'created_at':now(),'problem':redact(problem),'context':redact(context),'solution':redact(solution),'tools':redact(tools or []),'failure_modes':redact(failure_modes or []),'successful_strategy':redact(successful_strategy),'verification':redact(verification or {}),'tags':redact(tags or [])}
        self.state['experiences'][eid]=row; save_json(FILE,self.state); return row
    def list(self, tag=None, limit=20):
        rows=list(self.state['experiences'].values())
        if tag: rows=[r for r in rows if tag in r.get('tags',[])]
        rows.sort(key=lambda r:r.get('created_at',0),reverse=True); return rows[:max(1,int(limit))]
    def match(self, query, limit=5):
        q=str(query).casefold(); scored=[]
        for r in self.state['experiences'].values():
            hay=' '.join([str(r.get('problem','')),str(r.get('solution','')),' '.join(r.get('tags',[]))]).casefold()
            score=sum(1 for token in q.split() if token and token in hay)
            if score: scored.append((score,r))
        scored.sort(key=lambda x:x[0],reverse=True); return [r for _,r in scored[:max(1,int(limit))]]
