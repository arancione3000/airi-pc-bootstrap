import sys
import pytest
sys.path.insert(0, 'computer')
from control_plane import local_agent
from control_plane.model_router import ModelRouter
def test_chatgpt_only_surface_has_no_operational_provider():
    status=local_agent.provider_status()
    assert local_agent.CHATGPT_ONLY is True
    assert local_agent.REASONING_AUTHORITY == 'chatgpt'
    assert status['available'] is False and status['provider']=='disabled' and status['model'] is None
def test_local_agent_provider_calls_fail_closed():
    with pytest.raises(RuntimeError, match='sole reasoning authority'): local_agent._provider_request([{'role':'user','content':'x'}],'ignored')
    with pytest.raises(RuntimeError, match='sole reasoning authority'): local_agent.ask_local_model('goal')
    with pytest.raises(RuntimeError, match='sole reasoning authority'): local_agent.ask_local_model_changes('goal','context')
def test_openrouter_cannot_be_registered_operationally(monkeypatch):
    monkeypatch.setenv('AIRI_MODEL_PROVIDER','openrouter'); monkeypatch.setenv('OPENROUTER_API_KEY','TEST_ONLY_NOT_A_REAL_KEY')
    router=ModelRouter(); assert set(router.state['providers']) == {'chatgpt'}
    assert router.choose(task_type='coding')['selected']=='chatgpt'
    result=router.register_provider('openrouter',['coding'],available=True); assert result['available'] is False and 'openrouter' not in router.state['providers']
