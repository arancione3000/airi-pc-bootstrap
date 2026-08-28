from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'computer'))
import server


def test_browser_open_retries_after_worker_failure(monkeypatch):
    manager=server._BrowserManager()
    class Response: status=200
    class Page:
        url='http://127.0.0.1:9010/self-test'
        def title(self): return 'Airi-PC Self Test'
        def __init__(self): self.calls=0
        def goto(self,*args,**kwargs):
            self.calls+=1
            if self.calls == 1: raise RuntimeError('browser worker crashed')
            return Response()
    page=Page(); ensures=[]
    monkeypatch.setattr(manager,'_ensure',lambda: ensures.append(True) or page)
    monkeypatch.setattr(manager,'_close_worker',lambda: None)
    monkeypatch.setattr(server.time,'sleep',lambda *_: None)
    result=manager.open('http://127.0.0.1:9010/self-test')
    assert result['ok'] is True
    assert result['attempt']==2
    assert page.calls==2
    assert len(ensures)>=2
