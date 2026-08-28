from __future__ import annotations

import hmac
import secrets
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse


def build_router(backend, root: Path) -> APIRouter:
    router = APIRouter()
    token_path = root / '.ai' / 'state' / 'viewer_token'
    token_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not token_path.exists():
            token_path.write_text(secrets.token_urlsafe(32), encoding='utf-8')
            token_path.chmod(0o600)
    except OSError:
        pass

    def auth(request: Request) -> None:
        try:
            expected = token_path.read_text(encoding='utf-8').strip()
        except OSError:
            expected = ''
        supplied = request.headers.get('x-airi-viewer-token', '')
        if not expected or not supplied or not hmac.compare_digest(expected, supplied):
            raise HTTPException(status_code=401, detail='viewer authentication required')

    page = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Airi-PC Secure GUI</title>
<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0}header{position:sticky;top:0;padding:10px;background:#1d1d1d;display:flex;gap:10px;align-items:center;z-index:2}main{padding:10px}button,input{font:inherit;padding:8px;border-radius:7px;border:1px solid #555;background:#222;color:#eee}.status{margin-left:auto}.ok{color:#80e5a5}.bad{color:#ff8b8b}#screen{display:block;max-width:100%;width:1280px;height:auto;border:1px solid #444;cursor:crosshair;user-select:none}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.grow{flex:1;min-width:220px}.hint{opacity:.72;font-size:.9rem;margin-top:10px}</style>
</head><body><header><b>Airi-PC</b><span id="status" class="status">checking…</span><button onclick="stopAiri()">STOP AIRI</button><button onclick="refresh()">↻</button></header>
<main><img id="screen" alt="Airi-PC live display"><div class="row"><input id="url" class="grow" placeholder="https://chatgpt.com/"><button onclick="openUrl()">Apri URL</button></div>
<div class="row"><input id="text" class="grow" placeholder="Testo da digitare"><button onclick="typeText()">Digita</button><button onclick="key('ENTER')">ENTER</button></div>
<div class="hint">Clicca sullo schermo per inviare un click. Il token resta nel frammento # dell'URL e non viene inviato al server come query parameter.</div></main>
<script>const token=new URLSearchParams(location.hash.slice(1)).get('token')||''; const headers={'X-Airi-Viewer-Token':token,'Content-Type':'application/json'};
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers,body:body?JSON.stringify(body):undefined});if(!r.ok)throw new Error(await r.text());return r.json()}
async function refresh(){try{const s=await api('/viewer/state');const el=document.getElementById('status');el.textContent=(s.browser.available?'browser OK':'browser offline')+' • '+(s.url||'');el.className='status '+(s.ok?'ok':'bad');if(s.screenshot?.data_base64)document.getElementById('screen').src='data:image/png;base64,'+s.screenshot.data_base64}catch(e){document.getElementById('status').textContent='viewer error'}}
async function openUrl(){await api('/viewer/action',{action:'browser_open',payload:{url:document.getElementById('url').value,wait_until:'domcontentloaded'}});await refresh()}
async function typeText(){await api('/viewer/action',{action:'type',payload:{text:document.getElementById('text').value}});await refresh()}
async function key(k){await api('/viewer/action',{action:'key',payload:{key:k}});await refresh()}
async function stopAiri(){await api('/viewer/stop',{});document.getElementById('status').textContent='STOPPED'}
document.getElementById('screen').addEventListener('click',async e=>{const r=e.currentTarget.getBoundingClientRect();const x=Math.round((e.clientX-r.left)*1280/r.width);const y=Math.round((e.clientY-r.top)*800/r.height);await api('/viewer/action',{action:'click',payload:{x,y}});await refresh()});
refresh();setInterval(refresh,2000)</script></body></html>"""
    @router.get('/viewer', response_class=HTMLResponse)
    def viewer_page():
        return HTMLResponse(page)

    @router.get('/viewer/state')
    def viewer_state(request: Request):
        auth(request)
        bs = backend.browser().status()
        shot = backend.browser().screenshot()
        return {'ok': bool(bs.get('available') and shot.get('ok')), 'url': bs.get('url'), 'browser': bs, 'screenshot': shot}

    @router.post('/viewer/action')
    def viewer_action(request: Request, req: dict):
        auth(request)
        action = req.get('action')
        allowed = {'screenshot','observe','move','click','double_click','drag','scroll','key','hotkey','type','wait','browser_status','browser_open','browser_screenshot','browser_state'}
        if action not in allowed:
            raise HTTPException(status_code=403, detail='viewer action not allowed')
        return backend.perform(action, req.get('payload') or {})

    @router.post('/viewer/stop')
    def viewer_stop(request: Request):
        auth(request)
        stop = root / '.ai' / 'STOP'
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_text('viewer stop\n', encoding='utf-8')
        try: stop.chmod(0o600)
        except OSError: pass
        script = root / 'scripts' / 'airi-stop'
        if script.exists():
            subprocess.run(['sh', str(script)], cwd=root, capture_output=True, text=True, timeout=10, check=False)
        return {'ok':True,'stopped':True,'checkpoint_preserved':True}

    return router
