from __future__ import annotations
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .auth import AuthStore
from .permissions import authorize
from .protocol import Request, ok, err
from .platform import PlatformControl

class Companion:
    VERSION='0.1.0'
    def __init__(self, state_dir: Path):
        self.state_dir=state_dir; self.auth=AuthStore(state_dir); self.control=PlatformControl(state_dir/'sandbox'); self.paired=self.auth._hash!=''; self.started=time.time(); self.stop_event=threading.Event(); self.disable_file=state_dir/'DISABLED'
    def status(self): return {'online':not self.disable_file.exists(),'version':self.VERSION,'paired':self.auth._hash!='','authenticated':self.auth._hash!='','disabled':self.disable_file.exists(),'capabilities':['screen','system','mouse','keyboard','windows','active_window','applications','filesystem','processes','process_control'],'os':self.control.info()['os'],'last_connection':time.time()}
    def dispatch(self, req: Request, confirmed=False):
        cap,act=req.capability,req.action
        level={'screen':'READ_ONLY','system':'READ_ONLY','windows':'READ_ONLY','processes':'READ_ONLY','mouse':'LOW_RISK','keyboard':'LOW_RISK','applications':'LOW_RISK','filesystem':'HIGH_RISK','process_control':'HIGH_RISK'}.get(cap,'DESTRUCTIVE')
        allowed,reason=authorize(cap,act,level,confirmed)
        if not allowed: return err(req,'PERMISSION_DENIED',reason)
        try:
            if self.disable_file.exists(): return err(req,'DISABLED','companion kill switch active')
            a=req.arguments
            if cap=='system': return ok(req,self.control.info())
            if cap=='screen' and act=='screenshot':
                img=self.control.screenshot(); path=self.state_dir/f'last-screen-{int(time.time())}.png'; img.save(path); return ok(req,{'artifact':path.name,'width':img.width,'height':img.height})
            if cap=='windows' and act=='list': return ok(req,self.control.windows())
            if cap=='windows' and act=='active': return ok(req,self.control.active_window())
            if cap=='processes' and act=='list': return ok(req,self.control.processes())
            if cap=='mouse': return ok(req,self.control.mouse(act,**a))
            if cap=='keyboard' and act in {'type','key','hotkey'}: return ok(req,self.control.keyboard(text=a.get('text') if act=='type' else None,key=a.get('key') or a.get('combo')))
            if cap=='applications' and act=='launch': return ok(req,self.control.launch(a.get('argv')))
            if cap=='filesystem' and act=='write': return ok(req,self.control.filesystem_write(a['path'],a.get('content','')))
            if cap=='filesystem' and act=='read': return ok(req,self.control.filesystem_read(a['path']))
            if cap=='filesystem' and act=='delete_test_file': return ok(req,self.control.filesystem_delete_test_file(a['path']))
            return err(req,'UNSUPPORTED_ACTION','unsupported capability/action')
        except PermissionError as e: return err(req,'PERMISSION_DENIED',str(e))
        except Exception as e: return err(req,'EXECUTION_ERROR',type(e).__name__)

class Handler(BaseHTTPRequestHandler):
    server_version='AiriCompanion/0.1'
    def _json(self, code, body):
        raw=json.dumps(body,separators=(',',':')).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/health': return self._json(200,{'online':not self.server.companion.disable_file.exists(),'version':self.server.companion.VERSION,'disabled':self.server.companion.disable_file.exists()})
        if self.path=='/status':
            token=self.headers.get('Authorization','');
            if not self.server.companion.auth.verify(token.removeprefix('Bearer ').strip()): return self._json(401,{'error':'unauthorized'})
            return self._json(200,self.server.companion.status())
        self._json(404,{'error':'not_found'})
    def do_POST(self):
        token=self.headers.get('Authorization','')
        if not self.server.companion.auth.verify(token.removeprefix('Bearer ').strip()): return self._json(401,{'error':'unauthorized'})
        if self.server.companion.disable_file.exists(): return self._json(423,{'error':'disabled'})
        if self.path!='/v1/execute': return self._json(404,{'error':'not_found'})
        try:
            n=int(self.headers.get('Content-Length','0')); d=json.loads(self.rfile.read(n)); req=Request.from_dict(d); confirmed=bool((d.get('meta') or {}).get('confirmed')); return self._json(200,self.server.companion.dispatch(req,confirmed=confirmed))
        except Exception as e: return self._json(400,{'error':'invalid_request','message':type(e).__name__})
    def log_message(self,*args): pass

def serve(host='127.0.0.1',port=17894,state_dir=None):
    root=Path(state_dir or os.environ.get('AIRIPC_COMPANION_STATE',str(Path.home()/'.airi-pc-companion')))
    c=Companion(root); srv=ThreadingHTTPServer((host,port),Handler); srv.companion=c; srv.timeout=1
    try:
        while not c.stop_event.is_set(): srv.handle_request()
    finally: srv.server_close()
