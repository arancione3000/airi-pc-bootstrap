import time

def test_dynamic_task_graph_persists():
    from control_plane import task_engine
    t=task_engine.TaskEngine(); row=t.start('test-only-dynamic-graph',[{'id':'a','title':'a'}],scope=['tests'])
    n=t.add_node(row['id'], {'id':'b','title':'b','depends_on':['a'],'operation':'test'}, created_by='replan')
    assert n['created_by']=='replan'
    t.update('a','completed',output={'ok':1},task_id=row['id'])
    assert t.read(row['id'])['nodes'][1]['status']=='running'

def test_experience_and_model_router(monkeypatch):
    from control_plane.experience import ExperienceStore
    from control_plane.model_router import ModelRouter
    e=ExperienceStore(); row=e.record('pytest timeout','python repo','reduce scope',['pytest'],['timeout'],'focused test',{'tests':1},['python'])
    assert any(x['id']==row['id'] for x in e.match('python timeout', limit=100))
    r=ModelRouter(); r.register_provider('local-coder',['strong'],True,'low')
    assert r.choose('coding',complexity='high')['selected']=='local-coder'

def test_orchestrator_supports_job_operations(monkeypatch):
    from control_plane.orchestrator import ControlPlane
    cp=ControlPlane()
    captured={}
    monkeypatch.setattr(cp, 'job_start', lambda *a, **k: {'id':'j1','status':'running'})
    monkeypatch.setattr(cp, 'job_status', lambda jid: {'id':jid,'status':'completed','exit_code':0})
    monkeypatch.setattr(cp.capabilities,'route', lambda candidates: {'selected':candidates[0] if candidates else None,'candidates':list(candidates)})
    import control_plane.orchestrator as mod
    monkeypatch.setattr(mod.REGISTRY,'record',lambda *a,**k: None)
    # exercise operation dispatch with a tiny task
    t=cp.tasks.start('job-op',[{'id':'n1','operation':'job_start','args':{'command':'true'},'candidates':['computer_terminal_start']}],scope=['tests'])
    r=cp.execute(t['id'],'n1',['computer_terminal_start'],'job_start',{'command':'true'},finalize=True)
    assert r['ok'] and r['result']['id']=='j1'

def test_context_pack_and_verification_engine():
    from control_plane.project_index import ProjectIndex
    from control_plane.verification_engine import VerificationEngine
    idx=ProjectIndex(); idx.refresh(['computer/control_plane'])
    pack=idx.context_pack('TaskEngine dynamic task',limit_files=5,max_bytes=20000)
    assert pack['files'] and pack['bytes']<=20000
    v=VerificationEngine().run(requirements=['goal','tests'], tests='python -m py_compile computer/control_plane/verification_engine.py', project_path='.')
    assert v['tests']=='PASS' and v['ready'] is True
