from __future__ import annotations
import time
from typing import Any, Iterable
from .store import load_json, save_json, now

class CapabilityManager:
    def __init__(self) -> None:
        self.data = load_json("capabilities.json", {"version":1,"capabilities":{}})
        self.data.setdefault("capabilities", {})

    def discover(self, tool_names: Iterable[str], schemas: dict[str, Any] | None = None) -> dict[str, Any]:
        schemas = schemas or {}
        for name in tool_names:
            row = self.data["capabilities"].setdefault(name, {"name":name,"category":self._category(name),"description":"Airi-PC MCP capability","dependencies":[],"available":False,"health":"unknown","latency_ms":None,"error_rate":0.0,"fallbacks":[]})
            row["input_schema"] = schemas.get(name, row.get("input_schema", {"type":"object","properties":{}}))
            row["available"] = False
            row["health"] = "unknown"
            row["last_discovered"] = now()
        save_json("capabilities.json", self.data)
        return self.summary()

    def _category(self, name: str) -> str:
        for prefix,cat in (("computer_browser_","browser"),("computer_file_","filesystem"),("computer_git_","git"),("computer_skill_","skills"),("computer_task_","tasks"),("computer_recovery_","recovery"),("computer_scheduler_","scheduler"),("computer_","computer")):
            if name.startswith(prefix): return cat
        return "other"

    def probe(self, name: str, ok: bool, latency_ms: float | None = None, error: str = "") -> dict[str, Any]:
        row=self.data["capabilities"].setdefault(name,{"name":name})
        row["available"]=bool(ok); row["health"]="healthy" if ok else "degraded"; row["last_verified"]=now()
        if latency_ms is not None:
            old=row.get("latency_ms"); row["latency_ms"]=round(latency_ms if old is None else old*0.8+latency_ms*0.2,2)
        if not ok: row["last_error"]=error; row["error_rate"]=min(1.0,float(row.get("error_rate",0))*0.9+0.1)
        else: row["error_rate"]=max(0.0,float(row.get("error_rate",0))*0.95)
        save_json("capabilities.json",self.data); return row

    def score(self, name: str) -> float:
        r=self.data["capabilities"].get(name) or {};
        if not r.get("available",False) or r.get("health") not in {"healthy", "degraded"}: return -1.0
        health={"healthy":1.0,"unknown":0.8,"degraded":0.35}.get(r.get("health"),0.2)
        latency=max(0.0,1.0-min(float(r.get("latency_ms") or 0)/5000.0,1.0))
        return round(health*0.55+latency*0.25+(1.0-float(r.get("error_rate",0)))*0.20,4)

    def route(self, candidates: Iterable[str]) -> dict[str, Any]:
        scored=sorted(((n,self.score(n)) for n in candidates), key=lambda x:x[1], reverse=True)
        selected=scored[0][0] if scored and scored[0][1] >= 0 else None
        return {"selected": selected, "candidates":[{"name":n,"score":s} for n,s in scored]}

    def invalidate(self, name: str, reason: str = "") -> dict[str, Any]:
        r=self.data["capabilities"].setdefault(name,{"name":name}); r.update({"available":False,"health":"failed","last_error":reason,"invalidated_at":now()}); save_json("capabilities.json",self.data); return r

    def summary(self) -> dict[str, Any]:
        rows=list(self.data["capabilities"].values()); healthy=sum(1 for r in rows if r.get("health")=="healthy")
        return {"count":len(rows),"healthy":healthy,"capabilities":rows}
