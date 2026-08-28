from __future__ import annotations
import uuid
from .store import load_json, save_json, now

STATES={"pending","ready","running","verifying","completed","failed","retrying","fallback","replanning","blocked","human_review","cancelled"}
class TaskEngine:
    def __init__(self): self.state=load_json("tasks.json", {"tasks":{},"active":None})
    def start(self, goal:str, nodes:list[dict]|list[str], scope:list[str]|None=None):
        tid=uuid.uuid4().hex[:12]; norm=[]
        for i,n in enumerate(nodes):
            if isinstance(n,str): n={"id":f"n{i+1}","title":n,"depends_on":[]}
            else: n={**n}; n.setdefault("id",f"n{i+1}"); n.setdefault("title",n["id"]); n.setdefault("depends_on",[])
            n.setdefault("repository", n.get("repo", '.')); n.setdefault("workspace", n.get("workspace", '.')); n.setdefault("owner", n.get("owner", 'control_plane')); n.setdefault("expected_result", n.get("expected", '')); n.setdefault("retry_policy", {}); n.setdefault("operation", 'analyze'); n.setdefault("args", {}); n.setdefault("verification", {}); n.setdefault("created_by", 'task_engine')
            n.setdefault("created_at", now()); n.setdefault("updated_at", now())
            n.update({"status":n.get("status","pending"),"input":n.get("input"),"output":n.get("output"),"error":None,"retry_count":n.get("retry_count",0),"checkpoint":n.get("checkpoint")})
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
        repositories=sorted({str(n.get('repository','.')) for n in norm})
        row={"id":tid,"goal":goal,"scope":scope or [],"repositories":repositories,"created_at":now(),"updated_at":now(),"nodes":norm,"current":norm[0]["id"] if norm else None}
        self.state["tasks"][tid]=row; self.state["active"]=tid; save_json("tasks.json",self.state); return row
    def read(self,tid=None):
        return self.state["tasks"].get(tid or self.state.get("active"))
    def update(self,node_id,status,output=None,error=None,checkpoint=None,task_id=None):
        if status not in STATES: raise ValueError(status)
        row=self.read(task_id);
        if row is None: raise KeyError(task_id or self.state.get("active"))
        node=next((n for n in row["nodes"] if n["id"]==node_id), None)
        if node is None: raise KeyError(node_id)
        node.update({"status":status,"output":output,"error":error,"checkpoint":checkpoint,"updated_at":now()})
        if status=="failed": node["retry_count"]+=1
        if status == "completed":
            for nxt in row["nodes"]:
                if nxt["status"]=="pending" and all(next(m for m in row["nodes"] if m["id"]==d)["status"]=="completed" for d in nxt["depends_on"]):
                    nxt["status"]="running"; row["current"]=nxt["id"]; break
        if any(n["status"]=="failed" for n in row["nodes"]):
            row["status"] = "failed"
        elif all(n["status"] in {"completed","cancelled"} for n in row["nodes"]):
            row["status"] = "completed"
            self.state["active"] = None
        else:
            row["status"] = row.get("status","running")
        row["updated_at"]=now(); save_json("tasks.json",self.state); return row
    def add_node(self, task_id, node, created_by='dynamic'):
        row=self.read(task_id)
        if row is None: raise KeyError(task_id)
        n=dict(node); n.setdefault('id', uuid.uuid4().hex[:10]); n.setdefault('title', n['id']); n.setdefault('depends_on',[])
        n.setdefault('operation','analyze'); n.setdefault('args',{}); n.setdefault('verification',{}); n.setdefault('created_by',created_by)
        n.update({'status':'pending','input':n.get('input'),'output':None,'error':None,'retry_count':0,'checkpoint':None,'created_at':now(),'updated_at':now()})
        ids={x['id'] for x in row['nodes']}
        if n['id'] in ids: raise ValueError('duplicate task node id')
        if any(d not in ids and d != n['id'] for d in n['depends_on']): raise ValueError('unknown task dependency')
        row['nodes'].append(n); row['repositories']=sorted({str(x.get('repository','.')) for x in row['nodes']}); row['updated_at']=now(); save_json('tasks.json',self.state); return n

    def add_dependency(self, task_id, node_id, depends_on):
        row=self.read(task_id)
        if row is None: raise KeyError(task_id)
        node=next((n for n in row['nodes'] if n['id']==node_id),None)
        if node is None: raise KeyError(node_id)
        if not any(n['id']==depends_on for n in row['nodes']): raise KeyError(depends_on)
        if depends_on == node_id: raise ValueError('self dependency')
        if depends_on not in node['depends_on']: node['depends_on'].append(depends_on)
        node['updated_at']=now(); row['updated_at']=now(); save_json('tasks.json',self.state); return node

    def skip(self, task_id, node_id, reason='not needed'):
        row=self.read(task_id)
        if row is None: raise KeyError(task_id)
        node=next((n for n in row['nodes'] if n['id']==node_id),None)
        if node is None: raise KeyError(node_id)
        if node['status']=='completed': return node
        node.update({'status':'cancelled','error':reason,'updated_at':now()}); save_json('tasks.json',self.state); return row

    def runnable(self, task_id=None):
        row=self.read(task_id)
        if row is None: return []
        out=[]
        for n in row['nodes']:
            if n['status'] not in {'pending','running'}: continue
            if all(next(m for m in row['nodes'] if m['id']==d)['status']=='completed' for d in n.get('depends_on',[])): out.append(n)
        return out

    def finish(self,status="completed"):
        row=self.read()
        if row is None: raise KeyError(self.state.get("active"))
        if status == "completed" and any(n["status"] != "completed" for n in row["nodes"]):
            raise ValueError("cannot mark task completed while nodes are unfinished")
        if status not in STATES: raise ValueError(status)
        row["status"]=status; row["updated_at"]=now()
        self.state["active"]=None; save_json("tasks.json",self.state); return row
