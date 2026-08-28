from __future__ import annotations
from typing import Any
from .capability_manager import CapabilityManager
from .transaction_engine import TransactionEngine
from .task_engine import TaskEngine
from .audit_engine import AuditEngine
from .project_index import ProjectIndex
from .maintenance import MaintenanceManager
from .skill_manager import SkillManager
from .reliability import REGISTRY
from .supervisor import Supervisor
from .store import load_json, save_json, now
from coding import analyze, read, search, write, patch, test, build, lint, git_status, git_diff, snapshot, restore_snapshot, safe_path

AUTONOMY_FILE = 'autonomous-workflow.json'


class ControlPlane:
    def __init__(self):
        self.capabilities=CapabilityManager(); self.transactions=TransactionEngine(); self.tasks=TaskEngine(); self.audit=AuditEngine(); self.index=ProjectIndex(); self.maintenance=MaintenanceManager(); self.skills=SkillManager(); self.supervisor=Supervisor()
    def bootstrap(self, tool_names, schemas=None):
        names=list(tool_names); caps=self.capabilities.discover(names,schemas); idx=self.index.refresh(); skills=self.skills.refresh(); self.audit.event(kind='control_plane_bootstrap',tool_count=len(names),index=idx,skill_count=len(skills.get('skills',{}))); return {'ok':True,'capabilities':caps,'index':idx,'skills':len(skills.get('skills',{}))}
    def status(self): return {'ok':True,'capabilities':self.capabilities.summary(),'reliability':REGISTRY.summary(),'tasks':self.tasks.state,'transactions':self.transactions.state,'index':self.index.summary(),'skills':self.skills.list(),'maintenance':self.maintenance.history(),'supervisor':self.supervisor.snapshot(),'autonomous_workflow':load_json(AUTONOMY_FILE, {'active':False,'phase':'idle','task_id':None}),'audit_events':len(self.audit.tail(100000))}
    def route(self,candidates):
        result=self.capabilities.route(candidates); self.audit.event(kind='route',candidates=list(candidates),result=result); return result
    def plan(self,goal,steps,scope=None):
        task=self.tasks.start(goal,steps,scope); self.audit.event(kind='plan',goal=goal,task_id=task['id'],scope=scope or [],steps=steps); return task
    def transaction_begin(self,paths,label='task'):
        tx=self.transactions.begin(paths,label); self.audit.event(kind='transaction_begin',transaction_id=tx['id'],files=paths,label=label); return tx
    def execute(self,task_id,node_id,candidates,operation,args=None,transaction_id=None,finalize=True):
        args=args or {}; route=self.route(candidates); tool=route.get('selected')
        if tool is None and candidates:
            self.capabilities.discover(candidates)
            route=self.route(candidates); tool=route.get('selected')
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
        latency=(__import__('time').perf_counter()-started)*1000; self.capabilities.probe(tool,error is None,latency,error or ''); REGISTRY.record(tool,error is None,latency,error or '')
        if transaction_id: self.transactions.step(transaction_id,operation,tool=tool,input_data=args,result=result,error=error)
        row=self.tasks.update(node_id,'completed' if error is None else 'failed',output=result,error=error,checkpoint={'tool':tool,'latency_ms':latency}) if finalize else self.tasks.read(task_id)
        self.audit.event(kind='execution',task_id=task_id,node_id=node_id,tool=tool,operation=operation,input=args,output=result,error=error)
        return {'ok':error is None,'selected_tool':tool,'route':route,'result':result,'error':error,'task':row,'latency_ms':round(latency,2)}

    def _classify_failure(self, error):
        text = str(error or '').lower()
        if not text:
            return 'unknown'
        if 'timeout' in text or 'timed out' in text:
            return 'timeout'
        if 'permission' in text or 'forbidden' in text or 'scope' in text:
            return 'permission'
        if 'network' in text or 'connection' in text or 'http' in text:
            return 'network'
        if 'not found' in text or 'missing' in text:
            return 'dependency'
        if 'invalid' in text or 'schema' in text:
            return 'input'
        return 'tool_error'

    def _default_candidates(self, operation):
        return {
            'read':['computer_file_read','computer_browser_text'],
            'search':['computer_file_search','computer_research'],
            'analyze':['computer_project_analyze'],
            'write':['computer_file_write'],
            'patch':['computer_file_patch'],
            'test':['computer_test_run'],
            'build':['computer_build_run'],
            'lint':['computer_lint'],
            'browser_open':['computer_browser_open'],
            'browser_state':['computer_browser_state'],
            'browser_screenshot':['computer_browser_screenshot','computer_screenshot'],
            'screenshot':['computer_screenshot'],
            'git_status':['computer_git_status'],
            'git_diff':['computer_git_diff'],
            'snapshot':['computer_snapshot'],
            'restore':['computer_restore_snapshot'],
        }.get(operation,[])

    def autonomous_goal(self, goal, steps=None, scope=None, max_time=900, max_iterations=25, max_retries=3, max_tool_calls=100, max_parallel_tasks=1, resume=True):
        persisted = load_json(AUTONOMY_FILE, {'version':1,'active':False,'goal':None,'task_id':None,'phase':'idle','step_index':0,'iteration':0,'tool_calls':0,'retries':0,'history':[],'result':None})
        can_resume = bool(resume and persisted.get('active') and persisted.get('goal') == goal and persisted.get('task_id'))
        task = self.tasks.read(persisted.get('task_id')) if can_resume else None
        if task is not None:
            normalized = task.get('nodes', [])
            step_index = int(persisted.get('step_index',0))
            iteration = int(persisted.get('iteration',0))
            tool_calls = int(persisted.get('tool_calls',0))
            retries_total = int(persisted.get('retries',0))
            history = list(persisted.get('history',[]))
            started_at = persisted.get('started_at',now())
        else:
            raw_steps = steps or [
                {'id':'understand','title':'understand goal','operation':'analyze','args':{'path':'.'}},
                {'id':'plan','title':'inspect available skills and capabilities','operation':'search','args':{'query':goal,'path':'skills','limit':20}},
                {'id':'verify','title':'verify workspace baseline','operation':'read','args':{'path':'.ai/control_plane/reliability.json'}},
            ]
            normalized=[]
            for i, step in enumerate(raw_steps):
                if isinstance(step,str):
                    step={'id':f'step{i+1}','title':step,'operation':'analyze','args':{'path':'.'}}
                normalized.append(dict(step))
            task=self.tasks.start(goal,normalized,scope or [])
            step_index=0; iteration=0; tool_calls=0; retries_total=0; history=[]; started_at=now()
        state={'version':1,'active':True,'goal':goal,'task_id':task['id'],'phase':'execute','step_index':step_index,'iteration':iteration,'tool_calls':tool_calls,'retries':retries_total,'history':history,'result':None,'started_at':started_at,'limits':{'max_time':int(max_time),'max_iterations':int(max_iterations),'max_retries':int(max_retries),'max_tool_calls':int(max_tool_calls),'max_parallel_tasks':int(max_parallel_tasks)}}
        save_json(AUTONOMY_FILE,state)
        import time as _time
        deadline=_time.monotonic()+max(1,int(max_time))
        while state['step_index'] < len(task['nodes']):
            if _time.monotonic() >= deadline or state['iteration'] >= int(max_iterations) or state['tool_calls'] >= int(max_tool_calls):
                state.update({'phase':'blocked','active':True,'result':{'status':'STOP_SAFELY','reason':'resource/time governor reached'}}); save_json(AUTONOMY_FILE,state); return state
            node=task['nodes'][state['step_index']]
            if node.get('status')=='completed':
                state['step_index']+=1; save_json(AUTONOMY_FILE,state); continue
            op=node.get('operation','analyze'); args=node.get('args') or {}; candidates=node.get('candidates') or self._default_candidates(op)
            retries=0; last=None; succeeded=False
            while retries <= int(max_retries) and _time.monotonic() < deadline:
                state['iteration']+=1; state['tool_calls']+=1; state['retries'] += 1 if retries else 0; save_json(AUTONOMY_FILE,state)
                result=self.execute(task['id'],node['id'],candidates,op,args,finalize=False)
                state['history'].append({'iteration':state['iteration'],'step':node['id'],'operation':op,'result_ok':result.get('ok',False),'tool':result.get('selected_tool'),'error':result.get('error'),'classification':self._classify_failure(result.get('error'))})
                if result.get('ok'):
                    updated=self.tasks.update(node['id'],'completed',output=result.get('result'),checkpoint={'workflow_iteration':state['iteration'],'tool':result.get('selected_tool')},task_id=task['id'])
                    state['step_index']+=1; state['phase']='execute'; state['last_success']=now(); save_json(AUTONOMY_FILE,state); task=updated; succeeded=True; break
                last=result; retries+=1
                if retries <= int(max_retries): _time.sleep(min(2 ** (retries-1),8))
            if not succeeded:
                err=last.get('error') if last else 'unknown failure'
                self.tasks.update(node['id'],'failed',output=last.get('result') if last else None,error=err,checkpoint={'workflow_iteration':state['iteration']},task_id=task['id'])
                state.update({'phase':'blocked','active':True,'result':{'status':'BLOCKED','step':node['id'],'error':err,'classification':self._classify_failure(err)}}); save_json(AUTONOMY_FILE,state); return state
        state.update({'phase':'complete','active':False,'result':{'status':'READY','task_id':task['id'],'iterations':state['iteration'],'tool_calls':state['tool_calls'],'retries':state['retries']},'completed_at':now()}); save_json(AUTONOMY_FILE,state); return state

    def verify(self,goal,result,tests=None,files=None,commit=None): return self.audit.event(kind='verification',goal=goal,result=result,tests=tests or [],files=files or [],commit=commit)
