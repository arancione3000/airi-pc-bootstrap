from __future__ import annotations
import uuid
from .store import load_json, save_json, now

STATES={"pending","running","completed","failed","blocked","cancelled"}
class TaskEngine:
    def __init__(self): self.state=load_json("tasks.json", {"tasks":{},"active":None})
    def start(self, goal:str, nodes:list[dict]|list[str], scope:list[str]|None=None):
        tid=uuid.uuid4().hex[:12]; norm=[]
        for i,n in enumerate(nodes):
            if isinstance(n,str): n={"id":f"n{i+1}","title":n,"depends_on":[]}
            else: n={**n}; n.setdefault("id",f"n{i+1}"); n.setdefault("title",n["id"]); n.setdefault("depends_on",[])
            n.update({"status":"pending","input":n.get("input"),"output":None,"error":None,"retry_count":0,"checkpoint":None})
            norm.append(n)
        ids=[n["id"] for n in norm]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task node id")
        known=set(ids)
        for n in norm:
            missing=[d for d in n["depends_on"] if d not in known]
            if missing:
                raise ValueError(f"unknown task dependency: {missing}")
        visiting, visited=set(), set()
        graph={n["id"]: list(n["depends_on"]) for n in norm}
        def visit(node):
            if node in visiting: raise ValueError("task dependency cycle detected")
            if node in visited: return
            visiting.add(node)
            for dep in graph[node]: visit(dep)
            visiting.remove(node); visited.add(node)
        for node in ids: visit(node)
        runnable=[n for n in norm if not n["depends_on"]]
        if runnable:
            runnable[0]["status"]="running"
        row={"id":tid,"goal":goal,"scope":scope or [],"created_at":now(),"updated_at":now(),"nodes":norm,"current":norm[0]["id"] if norm else None}
        self.state["tasks"][tid]=row; self.state["active"]=tid; save_json("tasks.json",self.state); return row
    def read(self,tid=None):
        return self.state["tasks"].get(tid or self.state.get("active"))
    def update(self,node_id,status,output=None,error=None,checkpoint=None,task_id=None):
        if status not in STATES: raise ValueError(status)
        row=self.read(task_id);
        if row is None: raise KeyError(task_id or self.state.get("active"))
        node=next((n for n in row["nodes"] if n["id"]==node_id), None)
        if node is None: raise KeyError(node_id)
        node.update({"status":status,"output":output,"error":error,"checkpoint":checkpoint})
        if status=="failed": node["retry_count"]+=1
        if status == "completed":
            for nxt in row["nodes"]:
                if nxt["status"]=="pending" and all(next(m for m in row["nodes"] if m["id"]==d)["status"]=="completed" for d in nxt["depends_on"]):
                    nxt["status"]="running"; row["current"]=nxt["id"]; break
        if any(n["status"]=="failed" for n in row["nodes"]):
            row["status"] = "failed"
        elif all(n["status"]=="completed" for n in row["nodes"]):
            row["status"] = "completed"
            self.state["active"] = None
        else:
            row["status"] = row.get("status","running")
        row["updated_at"]=now(); save_json("tasks.json",self.state); return row
    def finish(self,status="completed"):
        row=self.read()
        if row is None: raise KeyError(self.state.get("active"))
        if status == "completed" and any(n["status"] != "completed" for n in row["nodes"]):
            raise ValueError("cannot mark task completed while nodes are unfinished")
        if status not in STATES: raise ValueError(status)
        row["status"]=status; row["updated_at"]=now()
        self.state["active"]=None; save_json("tasks.json",self.state); return row
