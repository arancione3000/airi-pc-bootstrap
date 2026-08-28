import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "computer"))
import control_plane.orchestrator as mod


class FakeTasks:
    def __init__(self):
        self.tasks = {}
    def start(self, goal, nodes, scope):
        tid='t1'
        self.tasks[tid]={'id':tid,'goal':goal,'scope':scope,'nodes':[],'current':None,'status':'running'}
        for n in nodes:
            row=dict(n); row.update({'status':'running' if not self.tasks[tid]['nodes'] else 'pending','output':None,'error':None,'retry_count':0,'checkpoint':None})
            self.tasks[tid]['nodes'].append(row)
        self.tasks[tid]['current']=self.tasks[tid]['nodes'][0]['id'] if nodes else None
        return self.tasks[tid]
    def read(self, tid=None):
        return self.tasks.get(tid)
    def update(self, node_id, status, output=None, error=None, checkpoint=None, task_id=None):
        row=self.tasks[task_id]
        node=next(n for n in row['nodes'] if n['id']==node_id)
        node.update({'status':status,'output':output,'error':error,'checkpoint':checkpoint})
        if status=='completed':
            for nxt in row['nodes']:
                if nxt['status']=='pending':
                    nxt['status']='running'; row['current']=nxt['id']; break
        if all(n['status']=='completed' for n in row['nodes']): row['status']='completed'
        return row
    def finish(self, status='completed'):
        row=next(iter(self.tasks.values())); row['status']=status; return row


def test_goal_executes_and_persists(tmp_path, monkeypatch):
    f=tmp_path/'autonomy.json'; monkeypatch.setattr(mod,'AUTONOMY_FILE',f)
    cp=mod.ControlPlane(); cp.tasks=FakeTasks()
    calls=[]
    def execute(*args, **kwargs):
        calls.append(args[3]); return {'ok':True,'selected_tool':'fake','result':{'ok':True},'error':None}
    monkeypatch.setattr(cp,'execute',execute)
    out=cp.autonomous_goal('demo',steps=[{'id':'a','operation':'analyze','args':{}},{'id':'b','operation':'test','args':{}}],max_time=30)
    assert out['phase']=='complete' and out['result']['status']=='READY'
    assert calls==['analyze','test']
    assert json.loads(f.read_text())['active'] is False


def test_goal_retries_then_succeeds(tmp_path, monkeypatch):
    f=tmp_path/'autonomy.json'; monkeypatch.setattr(mod,'AUTONOMY_FILE',f)
    cp=mod.ControlPlane(); cp.tasks=FakeTasks(); n={'count':0}
    def execute(*args, **kwargs):
        n['count']+=1
        if n['count']==1: return {'ok':False,'selected_tool':'fake','result':None,'error':'timeout'}
        return {'ok':True,'selected_tool':'fake','result':{'ok':True},'error':None}
    monkeypatch.setattr(cp,'execute',execute)
    out=cp.autonomous_goal('retry',steps=[{'id':'a','operation':'analyze','args':{}}],max_time=30,max_retries=2)
    assert out['phase']=='complete' and out['retries']==1
    assert out['history'][0]['classification']=='timeout'


def test_goal_stops_on_governor(tmp_path, monkeypatch):
    f=tmp_path/'autonomy.json'; monkeypatch.setattr(mod,'AUTONOMY_FILE',f)
    cp=mod.ControlPlane(); cp.tasks=FakeTasks()
    monkeypatch.setattr(cp,'execute',lambda *a,**k: {'ok':True,'selected_tool':'fake','result':{},'error':None})
    out=cp.autonomous_goal('limit',steps=[{'id':'a','operation':'analyze','args':{}},{'id':'b','operation':'analyze','args':{}}],max_time=30,max_iterations=1)
    assert out['result']['status']=='STOP_SAFELY' and out['active'] is True


def test_goal_resumes_persisted_state(tmp_path, monkeypatch):
    f=tmp_path/'autonomy.json'; monkeypatch.setattr(mod,'AUTONOMY_FILE',f)
    cp=mod.ControlPlane(); cp.tasks=FakeTasks();
    monkeypatch.setattr(cp,'execute',lambda *a,**k: {'ok':True,'selected_tool':'fake','result':{},'error':None})
    first=cp.autonomous_goal('resume',steps=[{'id':'a','operation':'analyze','args':{}},{'id':'b','operation':'analyze','args':{}}],max_time=30,max_iterations=1)
    assert first['active'] is True
    # create a fresh coordinator instance over the same persisted task/controller state
    cp2=mod.ControlPlane(); cp2.tasks=cp.tasks
    monkeypatch.setattr(cp2,'execute',lambda *a,**k: {'ok':True,'selected_tool':'fake','result':{},'error':None})
    out=cp2.autonomous_goal('resume',max_time=30,max_iterations=10,resume=True)
    assert out['phase']=='complete'


def test_capability_discovered_tools_are_routable_before_first_probe(monkeypatch):
    from control_plane.capability_manager import CapabilityManager
    m=CapabilityManager()
    monkeypatch.setattr(m, 'data', {'version':1,'capabilities':{}})
    m.discover(['computer_file_read'])
    routed=m.route(['computer_file_read'])
    assert routed['selected']=='computer_file_read'
    assert m.data['capabilities']['computer_file_read']['health']=='unknown'


def test_skill_matching_reuses_existing_metadata(monkeypatch):
    from control_plane.skill_manager import SkillManager
    m=SkillManager(); m.state={'version':1,'skills':{'coding-task':{'name':'coding-task','version':'1.0','description':'structured coding workflow','required_tools':['computer_code_agent'],'status':'valid','dependencies':['pytest']}}}
    assert m.match('run a structured coding workflow')['matches'][0]['name']=='coding-task'


def test_control_plane_dispatches_browser_and_code_agent(monkeypatch):
    cp=mod.ControlPlane(); cp.capabilities.data={'version':1,'capabilities':{}}
    def fake_execute(*args, **kwargs): return {'ok':True,'selected_tool':'fake','result':{}}
    monkeypatch.setattr(cp,'execute',fake_execute)
    assert cp.autonomous_goal('browser smoke',steps=[{'id':'a','operation':'browser_state','args':{}}],max_time=5,max_iterations=2,max_tool_calls=2,resume=False)['phase']=='complete'


def test_dynamic_goal_plan_includes_execution_pipeline(monkeypatch):
    cp=mod.ControlPlane()
    plan=cp._synthesize_goal_plan('Implement backend and frontend feature across repositories', ['repo'])
    ops=[n['operation'] for n in plan]
    assert ops[0]=='research'
    assert 'analyze' in ops and 'context_pack' in ops and 'code_agent' in ops
    assert 'model_choose' in ops and 'test' in ops and 'verify' in ops
    assert all('repository' in n and 'workspace' in n for n in plan)


def test_code_agent_planning_only_is_not_task_completion(tmp_path, monkeypatch):
    f=tmp_path/'autonomy.json'; monkeypatch.setattr(mod,'AUTONOMY_FILE',f)
    cp=mod.ControlPlane(); cp.tasks=FakeTasks()
    monkeypatch.setattr(cp,'execute',lambda *a,**k: {'ok':True,'selected_tool':'computer_code_agent','result':{'ok':True,'note':'No concrete changes supplied; plan created and context loaded, no source was modified.'},'error':None})
    out=cp.autonomous_goal('implement feature',steps=[{'id':'a','operation':'code_agent','args':{},'verification':{'required':True}}],max_time=5,max_iterations=3,max_retries=0,max_tool_calls=3,resume=False)
    assert out['phase']=='blocked'
    assert out['result']['status'] in {'BLOCKED','STOP_SAFELY'}
    assert any(h['classification']=='verification' for h in out['history'])


def test_execution_strategy_reports_resource_constraints():
    cp=mod.ControlPlane()
    out=cp.execution_strategy(2048, True)
    assert out['strategy'] in {'full_install','targeted_install','isolated_component','static_verification'}
    assert {'mem_available_mb','estimated_memory_mb','resource_limited'} <= set(out)


def test_task_graph_tracks_multiple_repositories():
    from control_plane.task_engine import TaskEngine
    e=TaskEngine()
    row=e.start('multi-repo', [
        {'id':'backend','repository':'repo-backend','workspace':'repo-backend'},
        {'id':'frontend','repository':'repo-frontend','workspace':'repo-frontend','depends_on':['backend']},
    ])
    assert row['repositories']==['repo-backend','repo-frontend']
    e.update('backend','completed',task_id=row['id'])
    assert e.read(row['id'])['current']=='frontend'
    assert e.read(row['id'])['nodes'][1]['status']=='running'
