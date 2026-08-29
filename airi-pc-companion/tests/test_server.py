import json, threading, urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer
from companion.server import Companion, Handler
from companion.protocol import Request

def test_dispatch_system(tmp_path: Path):
 c=Companion(tmp_path); r=Request.from_dict({'capability':'system','action':'info'}); out=c.dispatch(r); assert out['success'] and out['result']['os']
