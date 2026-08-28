import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

sys.path.insert(0, 'computer')

from control_plane import local_agent
from control_plane.model_router import ModelRouter

class Handler(BaseHTTPRequestHandler):
    seen_auth = None
    def do_POST(self):
        Handler.seen_auth = self.headers.get('Authorization')
        length = int(self.headers.get('Content-Length', '0'))
        _ = self.rfile.read(length)
        body = {'choices': [{'message': {'content': '{"changes": [{"path": "sample.py", "operation": "patch", "old": "A", "new": "B", "test_command": "python -m pytest -q", "scope": ["sample.py"]}]}'}}]}
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *_): pass

def test_missing_key():
    old = dict(os.environ)
    try:
        os.environ['AIRI_MODEL_PROVIDER'] = 'openrouter'
        os.environ.pop('OPENROUTER_API_KEY', None)
        local_agent.PROVIDER = 'openrouter'
        try:
            local_agent._provider_request([{'role':'user','content':'x'}], 'inclusionai/ling-3.0-flash-fin:free')
        except RuntimeError as exc:
            assert 'OPENROUTER_API_KEY is missing' in str(exc)
        else:
            raise AssertionError('missing key did not fail safely')
    finally:
        os.environ.clear(); os.environ.update(old)

def test_structured_openrouter_request_without_real_secret():
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    old = dict(os.environ)
    try:
        os.environ['AIRI_MODEL_PROVIDER'] = 'openrouter'
        os.environ['OPENROUTER_API_KEY'] = 'TEST_ONLY_NOT_A_REAL_KEY'
        os.environ['OPENROUTER_MODEL'] = 'inclusionai/ling-3.0-flash-fin:free'
        os.environ['OPENROUTER_URL'] = f'http://127.0.0.1:{server.server_port}/chat'
        local_agent.PROVIDER = 'openrouter'
        result = local_agent.ask_local_model_changes('add feature', 'context', root='.')
        assert result and result[0]['path'] == 'sample.py'
        assert Handler.seen_auth == 'Bearer TEST_ONLY_NOT_A_REAL_KEY'
        assert 'TEST_ONLY_NOT_A_REAL_KEY' not in json.dumps(result)
    finally:
        server.shutdown(); thread.join(timeout=2); os.environ.clear(); os.environ.update(old)

def test_router_openrouter_available_requires_key():
    old = dict(os.environ)
    try:
        os.environ['AIRI_MODEL_PROVIDER'] = 'openrouter'
        os.environ['OPENROUTER_API_KEY'] = 'TEST_ONLY_NOT_A_REAL_KEY'
        router = ModelRouter()
        assert any(p.get('name') == 'openrouter' and p.get('available') for p in router.state['providers'].values())
    finally:
        os.environ.clear(); os.environ.update(old)
