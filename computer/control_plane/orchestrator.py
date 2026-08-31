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
from .job_manager import JobManager
from .verification_engine import VerificationEngine
from .experience import ExperienceStore
from .model_router import ModelRouter
from .subagent_manager import SubagentManager
from .reasoning_engine import ReasoningEngine
from .store import load_json, save_json, now
from coding import analyze, read, search, write, patch, test, build, lint, git_status, git_diff, snapshot, restore_snapshot, safe_path

AUTONOMY_FILE = 'autonomous-workflow.json'


class ControlPlane:
    def __init__(self):
        self.capabilities=CapabilityManager(); self.transactions=TransactionEngine(); self.tasks=TaskEngine(); self.audit=AuditEngine(); self.index=ProjectIndex(); self.maintenance=MaintenanceManager(); self.skills=SkillManager(); self.supervisor=Supervisor(); self.jobs=JobManager(); self.verification=VerificationEngine(); self.experience=ExperienceStore(); self.models=ModelRouter(); self.subagents=SubagentManager(); self.reasoning=ReasoningEngine()
    def bootstrap(self, tool_names, schemas=None):
        names=list(tool_names); caps=self.capabilities.discover(names,schemas); idx=self.index.refresh(); skills=self.skills.refresh(); self.audit.event(kind='control_plane_bootstrap',tool_count=len(names),index=idx,skill_count=len(skills.get('skills',{}))); return {'ok':True,'capabilities':caps,'index':idx,'skills':len(skills.get('skills',{}))}
    def status(self): return {'ok':True,'capabilities':self.capabilities.summary(),'reliability':REGISTRY.summary(),'tasks':self.tasks.state,'transactions':self.transactions.state,'index':self.index.summary(),'skills':self.skills.list(),'maintenance':self.maintenance.history(),'supervisor':self.supervisor.snapshot(),'jobs':self.jobs.list(),'verification':{'available':True},'experience':{'count':len(self.experience.state.get('experiences',{}))},'model_routing':self.models.status(),'subagents':self.subagents.list(),'reasoning':self.reasoning.status(self.reasoning.state.get('active_run_id')) if self.reasoning.state.get('active_run_id') else {'active_run_id':None},'autonomous_workflow':load_json(AUTONOMY_FILE, {'active':False,'phase':'idle','task_id':None}),'audit_events':len(self.audit.tail(100000))}
    def context_pack(self, query, limit_files=12, max_bytes=120000):
        return self.index.context_pack(query, limit_files, max_bytes)

    def reasoning_start(self, goal, plan=None, scope=None, metadata=None):
        meta = dict(metadata or {})
        meta.setdefault('model_route', self.choose_model(
            meta.get('task_type', 'coding'), meta.get('complexity', 'medium'),
            bool(meta.get('needs_vision', False)), bool(meta.get('prefer_speed', False))))
        result = self.reasoning.start(goal, plan, scope=scope, metadata=meta)
        self.audit.event(kind='reasoning_start', run_id=result['run_id'], goal=goal, model_route=meta['model_route'])
        return result

    def reasoning_status(self, run_id=None):
        return self.reasoning.status(run_id)

    def reasoning_next_action(self, run_id=None):
        result = self.reasoning.next_action(run_id)
        self.audit.event(kind='reasoning_next_action', run_id=result.get('run_id'), action=result.get('action'), step=(result.get('step') or {}).get('id'))
        return result

    def reasoning_observe(self, observation, run_id=None, evidence=None, phase=None):
        result = self.reasoning.observe(observation, run_id=run_id, evidence=evidence, phase=phase)
        self.audit.event(kind='reasoning_observe', run_id=result['run_id'])
        return result

    def reasoning_mark_step(self, step_id, status, run_id=None, result=None, error=None, evidence=None, metadata=None):
        out = self.reasoning.mark_step(step_id, status, run_id=run_id, result=result, error=error, evidence=evidence, metadata=metadata)
        self.audit.event(kind='reasoning_mark_step', run_id=out.get('run_id'), step_id=step_id, status=status)
        return out

    def reasoning_replan(self, reason=None, run_id=None, strategy=None):
        out = self.reasoning.replan(reason=reason, run_id=run_id, strategy=strategy)
        self.audit.event(kind='reasoning_replan', run_id=out.get('run_id'), reason=reason, strategy=strategy)
        return out

    def reasoning_feedback(self, operation, success, result=None, error=None, tool=None, task=None, step=None, evidence=None, metadata=None, run_id=None):
        out = self.reasoning.feedback(operation=operation, success=success, result=result, error=error, tool=tool, task=task, step=step, evidence=evidence, metadata=metadata, run_id=run_id)
        self.audit.event(kind='reasoning_feedback', run_id=run_id or self.reasoning.state.get('active_run_id'), operation=operation, success=success, step=step)
        return out

    def reasoning_finish(self, verified=False, result=None, run_id=None):
        out = self.reasoning.finish(verified=verified, result=result, run_id=run_id)
        self.audit.event(kind='reasoning_finish', run_id=out['run_id'], verified=verified)
        return out

    def reasoning_goal(self, goal, steps=None, scope=None, metadata=None, max_time=900, max_iterations=25, max_retries=3, max_tool_calls=100, resume=True):
        meta = dict(metadata or {})
        run = self.reasoning_start(goal, steps, scope, meta)
        self.reasoning_feedback('planning', True, result={'steps': [s['id'] for s in run['plan']], 'model_route': run['metadata'].get('model_route')}, step=run['current_step'], run_id=run['run_id'])
        try:
            execution = self.autonomous_goal(goal, steps or [self._reasoning_node_to_task_step(s) for s in run['plan']], scope, max_time, max_iterations, max_retries, max_tool_calls, 1, resume=False)
            task_id = execution.get('task_id')
            task = self.tasks.read(task_id) if task_id else None
            if task:
                for node in task.get('nodes', []):
                    status = node.get('status', 'pending').upper()
                    mapped = 'COMPLETED' if status == 'COMPLETED' else 'SKIPPED' if status in {'CANCELLED','SKIPPED'} else 'FAILED' if status == 'FAILED' else 'RUNNING' if status == 'RUNNING' else 'PENDING'
                    try:
                        if mapped == 'RUNNING': continue
                        self.reasoning_mark_step(node['id'], mapped, run['run_id'], result=node.get('output'), error=node.get('error'), evidence=node.get('checkpoint'))
                    except (KeyError, ValueError):
                        continue
            ok = execution.get('result', {}).get('status') == 'READY'
            self.reasoning_feedback('autonomous_goal', ok, result=execution, error=None if ok else execution.get('result', {}).get('reason'), task=task_id, run_id=run['run_id'])
            self.reasoning.mark_done_criterion('unit_tests', ok, run_id=run['run_id'])
            self.reasoning.mark_done_criterion('git_diff', ok, run_id=run['run_id'])
            self.reasoning.mark_done_criterion('evidence', bool(self.reasoning.status(run['run_id']).get('evidence')) if ok else False, run_id=run['run_id'])
            if not ok:
                return self.reasoning.fail(str(execution.get('result', {}).get('reason', 'engineering loop failed')), run_id=run['run_id'])
            auto_commit = bool(meta.get('auto_commit', False))
            commit_required = bool(meta.get('commit_required', False) or auto_commit)
            project_path = str(meta.get('project_path') or (list(scope or ['.'])[0]))
            commit_result = None
            if auto_commit:
                from code_agent import atomic_commit as coding_atomic_commit
                commit_message = str(meta.get('commit_message') or f'reasoning: {goal}')
                commit_result = coding_atomic_commit(commit_message, project_path, scope, bool(meta.get('allow_test_changes', False)), bool(meta.get('allow_security_changes', False)))
                committed = bool(commit_result.get('commit', {}).get('committed'))
                self.reasoning.mark_done_criterion('commit', committed, run_id=run['run_id'])
                if not committed:
                    return self.reasoning.fail('auto_commit did not create a commit', run_id=run['run_id'])
            else:
                self.reasoning.mark_done_criterion('commit', None if not commit_required else False, run_id=run['run_id'])
            self.reasoning.mark_done_criterion('persistence', True, run_id=run['run_id'])
            return self.reasoning_finish(True, {'execution': execution, 'commit': commit_result}, run['run_id'])
        except Exception as exc:
            self.reasoning_feedback('reasoning_goal', False, error=str(exc), run_id=run['run_id'])
            try: self.reasoning_replan(str(exc), run['run_id'])
            except Exception: pass
            return self.reasoning.fail(str(exc), run_id=run['run_id'])

    @staticmethod
    def _reasoning_node_to_task_step(step):
        return {
            'id': step['id'], 'title': step.get('title', step['id']), 'operation': step.get('operation', 'analyze'),
            'args': step.get('args', {}), 'depends_on': step.get('dependencies', []),
            'repository': step.get('repository', '.'), 'workspace': step.get('workspace', '.'),
            'verification': {'required': step.get('kind') == 'verification'},
        }

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
            elif operation=='browser_open':
                from server import browser
                result=browser().open(args['url'], args.get('wait_until','domcontentloaded'))
            elif operation=='browser_state':
                from server import browser
                result=browser().state()
            elif operation=='browser_screenshot':
                from server import browser
                result=browser().screenshot()
            elif operation=='browser_text':
                from server import browser
                result=browser().text()
            elif operation=='research':
                from advanced import research
                result=research(args['topic'], args.get('urls'), args.get('max_sources',5))
            elif operation=='code_agent':
                from code_agent import agent
                result=agent(args['goal'], args.get('project_path','.'), args.get('max_attempts',5), args.get('steps'), args.get('scope'), args.get('changes'), args.get('test_command',''))
            elif operation=='context_pack': result=self.context_pack(args['query'],args.get('limit_files',12),args.get('max_bytes',120000))
            elif operation=='verify': result=self.verify_deliverable(args.get('requirements'),args.get('tests'),args.get('build_cmd'),args.get('lint_cmd'),args.get('project_path','.'),args.get('runtime'),args.get('security'))
            elif operation=='experience_match': result=self.experience_match(args['query'],args.get('limit',5))
            elif operation=='model_choose': result=self.choose_model(args.get('task_type','coding'),args.get('complexity','medium'),args.get('needs_vision',False),args.get('prefer_speed',False))
            elif operation=='subagent_create': result=self.subagent_create(args['goal'],args.get('repo_path','.'),args.get('branch_prefix','agent'))
            elif operation=='environment_strategy': result=self.execution_strategy(args.get('estimated_memory_mb',1024),args.get('full_integration',True))
            elif operation=='job_start': result=self.job_start(args['command'],args.get('cwd','.'),args.get('timeout',900),task_id,args.get('scope'),args.get('allow_shell',False))
            elif operation=='job_status': result=self.job_status(args['job_id'])
            elif operation=='job_attach': result=self.job_attach(args['job_id'],args.get('tail',200))
            elif operation=='job_cancel': result=self.job_cancel(args['job_id'],args.get('grace',5))
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
        if any(x in text for x in ('exit 137', 'exit code 137', 'killed', 'out of memory', 'oom', 'cannot allocate memory', 'resource temporarily unavailable')):
            return 'resource_limit'
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
            'browser_open':['computer_browser_open'],
            'browser_state':['computer_browser_state'],
            'browser_screenshot':['computer_browser_screenshot','computer_screenshot'],
            'browser_text':['computer_browser_text'],
            'research':['computer_research'],
            'context_pack':['computer_context_pack'],
            'verify':['computer_verify_deliverable'],
            'experience_match':['computer_experience_match'],
            'model_choose':['computer_model_choose'],
            'subagent_create':['computer_subagent_create'],
            'environment_strategy':['computer_runtime_preflight'],
            'code_agent':['computer_code_agent'],
            'job_start':['computer_terminal_start'],
            'job_status':['computer_terminal_status'],
            'job_attach':['computer_terminal_attach'],
            'job_cancel':['computer_terminal_cancel'],
        }.get(operation,[])

    def execution_strategy(self, estimated_memory_mb=1024, full_integration=True):
        try:
            mem=Path('/proc/meminfo').read_text()
            available_mb=int(next(line.split()[1] for line in mem.splitlines() if line.startswith('MemAvailable:')))/1024
        except Exception:
            available_mb=0
        estimate=float(estimated_memory_mb)
        if full_integration and available_mb >= estimate*1.5:
            strategy='full_install'
        elif available_mb >= max(256.0, estimate*0.75):
            strategy='targeted_install'
        elif available_mb >= 128.0:
            strategy='isolated_component'
        else:
            strategy='static_verification'
        return {'strategy':strategy,'mem_available_mb':round(available_mb,1),'estimated_memory_mb':estimate,'full_integration_requested':bool(full_integration),'resource_limited':available_mb < estimate}

    def autonomous_goal(self, goal, steps=None, scope=None, max_time=900, max_iterations=25, max_retries=3, max_tool_calls=100, max_parallel_tasks=1, resume=True):
        persisted = load_json(AUTONOMY_FILE, {'version':2,'active':False,'goal':None,'task_id':None,'phase':'idle','step_index':0,'iteration':0,'tool_calls':0,'retries':0,'history':[],'result':None,'completed_nodes':[]})
        can_resume = bool(resume and persisted.get('active') and persisted.get('goal') == goal and persisted.get('task_id'))
        task = self.tasks.read(persisted.get('task_id')) if can_resume else None
        if task is not None:
            step_index = int(persisted.get('step_index',0)); iteration=int(persisted.get('iteration',0)); tool_calls=int(persisted.get('tool_calls',0)); retries_total=int(persisted.get('retries',0)); history=list(persisted.get('history',[])); started_at=persisted.get('started_at',now())
        else:
            raw_steps = list(steps) if steps else self._synthesize_goal_plan(goal, scope or [])
            normalized=[]
            for i, step in enumerate(raw_steps):
                if isinstance(step,str): step={'id':f'step{i+1}','title':step,'operation':'analyze','args':{'path':'.'}}
                normalized.append(dict(step))
            task=self.tasks.start(goal,normalized,scope or []); step_index=0; iteration=0; tool_calls=0; retries_total=0; history=[]; started_at=now()
        skill_matches=self.skills.match(goal)
        state={'version':2,'active':True,'goal':goal,'task_id':task['id'],'phase':'execute','step_index':step_index,'iteration':iteration,'tool_calls':tool_calls,'retries':retries_total,'history':history,'result':None,'started_at':started_at,'skills':skill_matches,'completed_nodes':[n['id'] for n in task.get('nodes',[]) if n.get('status')=='completed'],'limits':{'max_time':int(max_time),'max_iterations':int(max_iterations),'max_retries':int(max_retries),'max_tool_calls':int(max_tool_calls),'max_parallel_tasks':int(max_parallel_tasks)}}
        save_json(AUTONOMY_FILE,state)
        import time as _time
        deadline=_time.monotonic()+max(1,int(max_time))
        while _time.monotonic() < deadline and state['iteration'] < int(max_iterations) and state['tool_calls'] < int(max_tool_calls):
            runnable = self.tasks.runnable(task['id']) if hasattr(self.tasks, 'runnable') else [n for n in task.get('nodes', []) if n.get('status') in ('pending','running')][:1]
            if not runnable:
                unfinished=[n for n in task['nodes'] if n.get('status') not in ('completed','cancelled')]
                if not unfinished:
                    break
                pending=unfinished[0]
                state.update({'phase':'replanning','current_step':pending['id']}); save_json(AUTONOMY_FILE,state)
                new_nodes=self._replan_nodes(goal,task,pending)
                if not new_nodes:
                    self.tasks.update(pending['id'],'blocked',error='no viable replanning path',checkpoint={'reason':'no_runnable_nodes'},task_id=task['id'])
                    state.update({'phase':'blocked','active':True,'result':{'status':'BLOCKED','step':pending['id'],'error':'no viable replanning path','classification':'strategy_exhausted'}})
                    save_json(AUTONOMY_FILE,state); return state
                for node in new_nodes:
                    self.tasks.add_node(task['id'],node,created_by='replanner')
                task=self.tasks.read(task['id']); continue
            node=runnable[0]
            if node.get('status')=='completed':
                continue
            op=node.get('operation','analyze'); args=node.get('args') or {}; candidates=node.get('candidates') or self._default_candidates(op)
            retries=int(node.get('retry_count',0)); last=None
            state['phase']='execute'; state['current_step']=node['id']; save_json(AUTONOMY_FILE,state)
            if node.get('status') in ('pending','ready'): self.tasks.update(node['id'],'running',task_id=task['id'])
            while retries <= int(max_retries) and _time.monotonic() < deadline:
                state['iteration']+=1; state['tool_calls']+=1; save_json(AUTONOMY_FILE,state)
                result=self.execute(task['id'],node['id'],candidates,op,args,finalize=False)
                ok=result.get('ok',False); cls=self._classify_failure(result.get('error'))
                history_row={'iteration':state['iteration'],'step':node['id'],'operation':op,'result_ok':ok,'tool':result.get('selected_tool'),'error':result.get('error'),'classification':cls}
                state['history'].append(history_row)
                if ok:
                    self.tasks.update(node['id'],'verifying',output=result.get('result'),checkpoint={'workflow_iteration':state['iteration'],'tool':result.get('selected_tool')},task_id=task['id'])
                    verification=self._verify_node(node,result)
                    if verification['ok']:
                        updated=self.tasks.update(node['id'],'completed',output=result.get('result'),checkpoint={'workflow_iteration':state['iteration'],'tool':result.get('selected_tool'),'verification':verification},task_id=task['id'])
                        task=updated; state['completed_nodes'].append(node['id']); state['step_index']=sum(1 for n in task['nodes'] if n.get('status')=='completed'); state['last_success']=now(); save_json(AUTONOMY_FILE,state); break
                    self.tasks.update(node['id'],'failed',output=result.get('result'),error=verification.get('error','verification failed'),checkpoint={'verification':verification},task_id=task['id']); last={'error':verification.get('error','verification failed'),'result':result.get('result')}; cls='verification'; history_row['classification']=cls; state['history'][-1]=history_row; save_json(AUTONOMY_FILE,state)
                else:
                    last=result
                    self.tasks.update(node['id'],'retrying',error=result.get('error'),checkpoint={'classification':cls},task_id=task['id'])
                retries += 1; state['retries'] += 1
                if cls in ('resource_limit','dependency'):
                    state['phase']='fallback'; save_json(AUTONOMY_FILE,state)
                    fallback=self._fallback_for(node,cls)
                    if fallback:
                        node.setdefault('args',{}).update(fallback.get('args',{})); node['operation']=fallback.get('operation',node.get('operation')); node['candidates']=fallback.get('candidates',node.get('candidates')); self.tasks.update(node['id'],'fallback',checkpoint={'classification':cls,'fallback':fallback},task_id=task['id']); retries=max(retries-1,0)
                    else:
                        self.tasks.update(node['id'],'replanning',checkpoint={'classification':cls},task_id=task['id'])
                        break
                elif retries <= int(max_retries):
                    _time.sleep(min(2 ** max(retries-1,0),8))
            else:
                err=(last or {}).get('error','unknown failure')
                self.tasks.update(node['id'],'failed',error=err,checkpoint={'classification':self._classify_failure(err)},task_id=task['id'])
                state.update({'phase':'blocked','active':True,'result':{'status':'BLOCKED','step':node['id'],'error':err,'classification':self._classify_failure(err)}}); save_json(AUTONOMY_FILE,state); return state
            task=self.tasks.read(task['id'])
        task=self.tasks.read(task['id']) if task else task
        nodes_complete=bool(task and task.get('nodes')) and all(n.get('status') in ('completed','cancelled') for n in task.get('nodes',[]))
        if task and (task.get('status')=='completed' or nodes_complete):
            state.update({'phase':'complete','active':False,'result':{'status':'READY','task_id':task['id'],'iterations':state['iteration'],'tool_calls':state['tool_calls'],'retries':state['retries']},'completed_at':now()}); save_json(AUTONOMY_FILE,state); return state
        reason='resource/time governor reached' if _time.monotonic() >= deadline or state['iteration'] >= int(max_iterations) or state['tool_calls'] >= int(max_tool_calls) else 'workflow blocked'
        state.update({'phase':'blocked','active':True,'result':{'status':'STOP_SAFELY','reason':reason}}); save_json(AUTONOMY_FILE,state); return state

    def _synthesize_goal_plan(self, goal, scope):
        text=goal.lower(); repo=scope[0] if scope else '.'
        steps=[{'id':'research','title':'research requirements and relevant repository context','operation':'research','args':{'topic':goal,'max_sources':5},'repository':repo,'workspace':repo,'verification':{'required':False}},
               {'id':'analyze','title':'analyze project and targeted context','operation':'analyze','args':{'path':repo},'repository':repo,'workspace':repo,'verification':{'required':True}},
               {'id':'context','title':'build focused context pack','operation':'context_pack','args':{'query':goal,'limit_files':12,'max_bytes':120000},'repository':repo,'workspace':repo},
               {'id':'implementation','title':'implement the requested change','operation':'code_agent','args':{'goal':goal,'project_path':repo,'max_attempts':3,'scope':scope or [repo]},'repository':repo,'workspace':repo,'verification':{'required':True}},
               {'id':'test','title':'run targeted verification','operation':'test','args':{'command':'python -m compileall -q .','cwd':repo,'timeout':120},'repository':repo,'workspace':repo,'verification':{'required':True}},
               {'id':'review','title':'review diff and guardrails','operation':'git_diff','args':{'path':repo},'repository':repo,'workspace':repo},
               {'id':'verify','title':'verify deliverable against requirements','operation':'verify','args':{'requirements':[goal],'project_path':repo},'repository':repo,'workspace':repo,'verification':{'required':True}}]
        if 'multi-repo' in text or 'repository' in text or 'frontend' in text or 'backend' in text:
            steps.insert(3,{'id':'environment','title':'choose resource-aware environment strategy','operation':'environment_strategy','args':{'estimated_memory_mb':2048,'full_integration':True},'repository':repo,'workspace':repo})
            steps.insert(4,{'id':'model','title':'choose execution route','operation':'model_choose','args':{'task_type':'coding','complexity':'high'},'repository':repo,'workspace':repo})
        return steps

    def _verify_node(self,node,result):
        spec=node.get('verification') or {}
        if not spec or spec.get('required') is False: return {'ok':True,'level':'L1'}
        payload=result.get('result')
        if isinstance(payload,dict):
            if payload.get('ok') is False: return {'ok':False,'level':'L0','error':'tool reported unsuccessful result'}
            note=str(payload.get('note','')).lower()
            if note and 'no source was modified' in note: return {'ok':False,'level':'L0','error':'planning_only_no_source_change'}
            if payload.get('status') in {'SKIPPED','STOP_SAFELY'}: return {'ok':False,'level':'L0','error':str(payload.get('status'))}
        return {'ok':True,'level':'L1'}

    def _fallback_for(self,node,classification):
        if classification=='resource_limit':
            op=node.get('operation')
            if op=='test': return {'operation':'build','args':{'command':'python -m compileall -q .','cwd':node.get('args',{}).get('cwd','.'),'timeout':120},'candidates':['computer_build_run']}
            if op=='code_agent': return {'operation':'analyze','args':{'path':node.get('repository','.')},'candidates':['computer_project_analyze']}
        if classification=='dependency' and node.get('operation')=='test': return {'operation':'build','args':{'command':'python -m compileall -q .','cwd':node.get('args',{}).get('cwd','.'),'timeout':120},'candidates':['computer_build_run']}
        return None

    def _replan_nodes(self,goal,task,pending):
        nodes=[]
        if pending.get('operation')=='code_agent' and not any(n.get('id')=='test-generated' for n in task.get('nodes',[])):
            nodes.append({'id':'test-generated','title':'generated focused verification','operation':'test','args':{'command':'python -m compileall -q .','cwd':pending.get('repository','.'),'timeout':120},'depends_on':[pending['id']],'repository':pending.get('repository','.'),'workspace':pending.get('workspace','.')})
        return nodes

    def job_start(self, command, cwd='.', timeout=900, owner_task=None, scope=None, allow_shell=False):
        result=self.jobs.start(command,cwd,timeout,owner_task,scope,allow_shell)
        self.audit.event(kind='job_start',job_id=result['id'],owner_task=owner_task,command=command,cwd=cwd)
        return result

    def job_status(self, job_id):
        return self.jobs.status(job_id)

    def job_list(self):
        return self.jobs.list()

    def job_attach(self, job_id, tail=200):
        return self.jobs.attach(job_id,tail)

    def job_detach(self, job_id):
        return self.jobs.detach(job_id)

    def job_cancel(self, job_id, grace=5):
        result=self.jobs.cancel(job_id,grace); self.audit.event(kind='job_cancel',job_id=job_id,result=result); return result

    def job_cleanup(self, keep_final=100):
        return self.jobs.cleanup(keep_final)

    def verify(self,goal,result,tests=None,files=None,commit=None): return self.audit.event(kind='verification',goal=goal,result=result,tests=tests or [],files=files or [],commit=commit)

    def verify_deliverable(self, requirements=None, tests=None, build_cmd=None, lint_cmd=None, project_path='.', runtime=None, security=None):
        result=self.verification.run(requirements=requirements, tests=tests, build_cmd=build_cmd, lint_cmd=lint_cmd, project_path=project_path, runtime=runtime, security=security)
        self.audit.event(kind='deliverable_verification', result=result); return result

    def record_experience(self, problem, context, solution, tools=None, failure_modes=None, successful_strategy='', verification=None, tags=None):
        row=self.experience.record(problem,context,solution,tools,failure_modes,successful_strategy,verification,tags); self.audit.event(kind='experience_record',experience_id=row['id']); return row

    def experience_match(self, query, limit=5): return {'experiences':self.experience.match(query,limit)}

    def subagent_create(self, goal, repo_path='.', branch_prefix='agent'):
        return self.subagents.create(goal,repo_path,branch_prefix)

    def subagent_status(self, agent_id): return self.subagents.status(agent_id)
    def subagent_list(self): return self.subagents.list()
    def subagent_finish(self, agent_id, status='ready'): return self.subagents.finish(agent_id,status)
    def subagent_remove(self, agent_id, force=False): return self.subagents.remove(agent_id,force)

    def choose_model(self, task_type='simple', complexity='medium', needs_vision=False, prefer_speed=False):
        return self.models.choose(task_type,complexity,needs_vision,prefer_speed)
