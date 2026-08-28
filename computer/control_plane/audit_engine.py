from __future__ import annotations
from .store import append_jsonl, load_json
class AuditEngine:
    def event(self, **kwargs):
        from .store import now,redact
        row={"timestamp":now(), **redact(kwargs)}; append_jsonl("audit.jsonl",row); return row
    def tail(self,limit=100):
        from .store import CP
        p=CP/"audit.jsonl"
        if not p.exists(): return []
        rows=[]
        for line in p.read_text(encoding="utf-8").splitlines()[-max(1,int(limit)):]:
            try: rows.append(__import__("json").loads(line))
            except Exception: pass
        return rows
