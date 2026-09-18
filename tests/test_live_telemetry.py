from __future__ import annotations


def test_live_topic_is_stable_across_commits():
    from control_plane.live_telemetry import STABLE_TOPIC, topic_for
    a = topic_for('a' * 40)
    b = topic_for('b' * 40)
    assert a == b == STABLE_TOPIC
    assert a.startswith('airi-live-')
    assert len(a) >= len('airi-live-') + 32


def test_live_telemetry_redacts_secrets():
    from control_plane.live_telemetry import _safe
    text = _safe('token=abc123 password:xyz authorization=Bearer-777')
    assert 'abc123' not in text and 'xyz' not in text and 'Bearer-777' not in text
    assert '[redacted]' in text


def test_task_engine_emits_live_state(tmp_path, monkeypatch):
    import control_plane.task_engine as mod
    events = []
    monkeypatch.setattr(mod, 'load_json', lambda *a, **k: {'tasks': {}, 'active': None})
    monkeypatch.setattr(mod, 'save_json', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'live_emit', lambda *a, **k: events.append((a, k)) or True)
    e = mod.TaskEngine()
    row = e.start('demo goal', [{'id': 'one', 'title': 'Analyze', 'operation': 'analyze'}])
    e.update('one', 'completed', task_id=row['id'])
    assert any(args[1] == 'Task started' for args, _ in events)
    assert any(args[1] == 'Step completed' for args, _ in events)
    assert any(args[1] == 'Task completed' for args, _ in events)
