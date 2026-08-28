from __future__ import annotations
from typing import Any
from .capability_manager import CapabilityManager
from .transaction_engine import TransactionEngine
from .task_engine import TaskEngine
from .audit_engine import AuditEngine
from .project_index import ProjectIndex
from .maintenance import MaintenanceManager
from .skill_manager import SkillManager
from coding import analyze, read, search, write, patch, test, build, lint, git_status, git_diff, snapshot, restore_snapshot, safe_path

class ControlPlane:
    def __init__(self):
        self.capabilities=CapabilityManager(); self.transactions=TransactionEngine(); self.tasks=TaskEngine(); self.audit=AuditEngine(); self.index=ProjectIndex(); self.maintenance=MaintenanceManager(); self.skills=SkillManager()
    def bootstrap(self, tool_names, schemas=None):
        names=list(tool_names); caps=self.capabilities.discover(names,schemas); idx=self.index.refresh(); skills=self.skills.refresh(); self.audit.event(kind='control_plane_bootstrap',tool_count=len(names),index=idx,skill_count=len(skills.get('skills',{}))); return {'ok':True,'capabilities':caps,'index':idx,'skills':len(skills.get('skills',{}))}
    def status(self): return {'ok':True,'capabilities':self.capabilities.summary(),'tasks':self.tasks.state,'transactions':self.transactions.state,'index':self.index.summary(),'skills':self.skills.list(),'maintenance':self.maintenance.history(),'audit_events':len(self.audit.tail(100000))}
    def route(self,candidates):
        result=self.capabilities.route(candidates); self.audit.event(kind='route',candidates=list(candidates),result=result); return result
    def plan(self,goal,steps,scope=None):
        task=self.tasks.start(goal,steps,scope); self.audit.event(kind='plan',goal=goal,task_id=task['id'],scope=scope or [],steps=steps); return task
    def transaction_begin(self,paths,label='task'):
        tx=self.transactions.begin(paths,label); self.audit.event(kind='transaction_begin',transaction_id=tx['id'],files=paths,label=label); return tx
    def execute(self,task_id,node_id,candidates,operation,args=None,transaction_id=None):
        args=args or {}; route=self.route(candidates); tool=route.get('selected')
        if not tool: return {'ok':False,'error':'no healthy capability'}
        started=__import__('time').perf_counter(); error=None; result=None
        try:
            if operation=='read': result=read(args['path'])
            elif operation=='search': result=search(args['query'],args.get('path','.'),args.get('limit',100),args.get('regex',False),args.get('case_sensitive',False))
            elif operation=='analyze': result=analyze(args.get('path','.'))
            elif operation=='write': result=write(args['path'],args['content'],declared_scope=args.get('scope',[]))
            elif operation=='patch': result=patch(args['path'],args['old'],args['new'],args.get('replace_all',False),declared_scope=args.get('scope',[]))
            elif operation=='test': result=test(args.get('command','pytest -q'),args.get('cwd','.'),args.get('timeout',170))
            elif operation=='build': result=build(args.get('command','python -m compileall -q .'),args.get('cwd','.'),args.get('timeout',170))
            elif operation=='lint': result=lint(args.get('command'),args.get('cwd','.'),args.get('timeout',170))
            elif operation=='git_status': result=git_status(args.get('path','.'))
            elif operation=='git_diff': result=git_diff(args.get('path','.'))
            elif operation=='snapshot': result=snapshot(args.get('paths',[]),args.get('label','control-plane'))
            elif operation=='restore': result=restore_snapshot(args['snapshot'])
            else: raise ValueError('unsupported operation')
        except Exception as exc: error=str(exc)
        latency=(__import__('time').perf_counter()-started)*1000; self.capabilities.probe(tool,error is None,latency,error or '')
        if transaction_id: self.transactions.step(transaction_id,operation,tool=tool,input_data=args,result=result,error=error)
        row=self.tasks.update(node_id,'completed' if error is None else 'failed',output=result,error=error,checkpoint={'tool':tool,'latency_ms':latency})
        self.audit.event(kind='execution',task_id=task_id,node_id=node_id,tool=tool,operation=operation,input=args,output=result,error=error)
        return {'ok':error is None,'selected_tool':tool,'route':route,'result':result,'error':error,'task':row,'latency_ms':round(latency,2)}
    def verify(self,goal,result,tests=None,files=None,commit=None): return self.audit.event(kind='verification',goal=goal,result=result,tests=tests or [],files=files or [],commit=commit)
