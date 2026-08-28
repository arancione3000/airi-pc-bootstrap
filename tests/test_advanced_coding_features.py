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
    assert any(x['id']==row['id'] for x in e.match('python timeout', limit=10))
    r=ModelRouter(); r.register_provider('local-coder',['strong'],True,'low')
    assert r.choose('coding',complexity='high')['selected']=='local-coder'
