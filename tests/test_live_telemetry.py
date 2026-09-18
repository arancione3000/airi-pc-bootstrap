from __future__ import annotations


def test_live_topic_is_stable_across_commits_and_matches_manifest():
    import json
    from control_plane.live_telemetry import CONFIG_PATH, STABLE_TOPIC, session_id, topic_for

    manifest = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    a = topic_for('a' * 40)
    b = topic_for('b' * 40)
    assert a == b == STABLE_TOPIC == manifest['topic']
    assert int(manifest['protocol']) == 2
    assert a.startswith('airi-live-')
    assert len(a) >= len('airi-live-') + 32
    assert session_id()


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
