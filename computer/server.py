import base64, io, os, re, subprocess, time, difflib, threading, atexit, urllib.parse, secrets, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageGrab, ImageChops

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from cleanup import scan as cleanup_scan, cleanup_safe
except Exception:
    cleanup_scan = None
    cleanup_safe = None

app = FastAPI(title='Airi Computer', version='2.0')
from coding import analyze as code_analyze, read as code_read, search as code_search, write as code_write, patch as code_patch, test as code_test, build as code_build, lint as code_lint, shell as code_shell, git_status as code_git_status, git_diff as code_git_diff, git_log as code_git_log, git_commit as code_git_commit, project_context as code_project_context, load_project_context as code_load_project_context, scope_check as code_scope_check, diff_summary as code_diff_summary, guardrail_check as code_guardrail_check, snapshot as code_snapshot, restore_snapshot as code_restore_snapshot
from skills import list_skills, load_skill, create_skill, update_skill, test_skill, delete_skill, memory_read, memory_update, task_start, task_read, task_update, task_finish, session_event
from code_agent import apply_fix as code_apply_fix, verify_change as code_verify_change, plan as code_plan, agent as code_agent, autonomous_change_cycle as code_autonomous_change_cycle, prepare_commit as code_prepare_commit, atomic_commit as code_atomic_commit
from advanced import (health_check, checkpoint, recovery_read, recovery_finish, record_decision, decisions,
                    persistence_status, persist_current, research, scheduler_status, schedule_job, cancel_job,
                    permission_check, runtime_preflight, integration_status, AI, ROOT)
from control_plane import ControlPlane

# Optional remote MCP authentication. Local requests remain unchanged when the
# token is unset; public deployments should set AIRI_MCP_TOKEN.
_mcp_env_token = os.environ.get('AIRI_MCP_TOKEN', '').strip()
_mcp_file = Path('/home/user/airi/.mcp_token')
AIRI_MCP_TOKEN = _mcp_env_token or (_mcp_file.read_text().strip() if _mcp_file.exists() else '')
AIRI_BOOTSTRAP_SHA = os.environ.get('AIRI_BOOTSTRAP_SHA', '').strip()
CONTROL_PLANE = ControlPlane()
CONTROL_PLANE_BOOTSTRAPPED = False

@app.middleware('http')
async def mcp_auth_middleware(request: Request, call_next):
    if AIRI_MCP_TOKEN and request.url.path == '/mcp':
        auth = request.headers.get('authorization', '')
        if auth != f'Bearer {AIRI_MCP_TOKEN}':
            return JSONResponse(status_code=401, content={'error': 'unauthorized'})
    return await call_next(request)

# Minimal OAuth 2.0 + PKCE for the remote MCP connector.
OAUTH_STATE = AI / 'state' / 'oauth.json'
OAUTH_LOCK = threading.RLock()

def _oauth_load():
    try:
        return json.loads(OAUTH_STATE.read_text(encoding='utf-8'))
    except Exception:
        return {'clients': {}, 'codes': {}, 'tokens': {}}

def _oauth_save(data):
    OAUTH_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OAUTH_STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(OAUTH_STATE)

def _oauth_redirect_ok(uri):
    try:
        u=urllib.parse.urlparse(uri)
        return u.scheme == 'https' and u.netloc == 'chatgpt.com' and u.path.startswith('/connector/oauth/')
    except Exception:
        return False

def _pkce(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()

def _oauth_token_ok(token):
    if not token:
        return False
    with OAUTH_LOCK:
        data=_oauth_load(); row=data.get('tokens',{}).get(token)
        if not row:
            return False
        if row.get('expires_at',0) <= time.time():
            data['tokens'].pop(token,None); _oauth_save(data); return False
        return True

@app.get('/.well-known/oauth-authorization-server')
async def oauth_metadata(request: Request):
    proto=request.headers.get('x-forwarded-proto') or request.url.scheme
    host=request.headers.get('x-forwarded-host') or request.headers.get('host')
    if host and host not in {'127.0.0.1:9010','localhost:9010'} and proto == 'http':
        proto='https'
    base=f'{proto}://{host}'.rstrip('/')
    return {'issuer':base,'authorization_endpoint':base+'/oauth/authorize','token_endpoint':base+'/oauth/token','registration_endpoint':base+'/oauth/register','response_types_supported':['code'],'grant_types_supported':['authorization_code'],'code_challenge_methods_supported':['S256'],'token_endpoint_auth_methods_supported':['none']}

@app.get('/.well-known/oauth-protected-resource')
async def oauth_resource_metadata(request: Request):
    proto=request.headers.get('x-forwarded-proto') or request.url.scheme
    host=request.headers.get('x-forwarded-host') or request.headers.get('host')
    if host and host not in {'127.0.0.1:9010','localhost:9010'} and proto == 'http':
        proto='https'
    base=f'{proto}://{host}'.rstrip('/')
    return {'resource':base+'/mcp','authorization_servers':[base]}

@app.post('/oauth/register')
async def oauth_register(request: Request):
    body=await request.json(); uris=body.get('redirect_uris') or []
    if not uris or not all(_oauth_redirect_ok(u) for u in uris):
        return JSONResponse(status_code=400, content={'error':'invalid_redirect_uri'})
    cid=body.get('client_id') or 'airi-'+secrets.token_urlsafe(18)
    with OAUTH_LOCK:
        data=_oauth_load(); data.setdefault('clients',{})[cid]={'redirect_uris':uris}; _oauth_save(data)
    return {'client_id':cid,'redirect_uris':uris,'token_endpoint_auth_method':'none'}

@app.get('/oauth/authorize')
async def oauth_authorize(request: Request):
    q=dict(request.query_params); cid=q.get('client_id',''); redirect_uri=q.get('redirect_uri',''); challenge=q.get('code_challenge','')
    if q.get('response_type')!='code' or q.get('code_challenge_method')!='S256' or not challenge or not _oauth_redirect_ok(redirect_uri):
        return JSONResponse(status_code=400, content={'error':'invalid_request'})
    with OAUTH_LOCK:
        data=_oauth_load(); client=data.setdefault('clients',{}).get(cid)
        if client is None:
            data['clients'][cid]={'redirect_uris':[redirect_uri]}
        elif redirect_uri not in client.get('redirect_uris',[]):
            return JSONResponse(status_code=400, content={'error':'redirect_uri_mismatch'})
        ticket=secrets.token_urlsafe(24)
        data.setdefault('codes',{})['approval:'+ticket]={'client_id':cid,'redirect_uri':redirect_uri,'code_challenge':challenge,'scope':q.get('scope','mcp'),'state':q.get('state',''),'created_at':time.time()}
        _oauth_save(data)
    approve_url=str(request.base_url).rstrip('/')+'/oauth/authorize/approve?ticket='+urllib.parse.quote(ticket,safe='')
    html='<!doctype html><html><body style="font-family:sans-serif;max-width:640px;margin:10vh auto;padding:24px"><h2>Airi-PC authorization</h2><p>Allow ChatGPT to access this Airi-PC instance?</p><form method="post" action="'+approve_url+'"><button style="font-size:18px;padding:12px 24px">Approve</button></form></body></html>'
    return HTMLResponse(html)

@app.post('/oauth/authorize/approve')
async def oauth_approve(request: Request):
    q=urllib.parse.parse_qs(urllib.parse.urlparse(str(request.url)).query); ticket=(q.get('ticket') or [''])[0]
    with OAUTH_LOCK:
        data=_oauth_load(); pending=data.get('codes',{}).pop('approval:'+ticket,None)
        if not pending: return JSONResponse(status_code=400, content={'error':'invalid_ticket'})
        code=secrets.token_urlsafe(32); data.setdefault('codes',{})[code]=pending; _oauth_save(data)
    sep='&' if '?' in pending['redirect_uri'] else '?'
    red=pending['redirect_uri']+sep+'code='+urllib.parse.quote(code,safe='')+'&state='+urllib.parse.quote(pending.get('state',''),safe='')
    return HTMLResponse('<html><body><script>location.replace('+json.dumps(red)+')</script><a href="'+red+'">Continue</a></body></html>')

@app.post('/oauth/token')
async def oauth_token(request: Request):
    body=urllib.parse.parse_qs((await request.body()).decode('utf-8')); code=(body.get('code') or [''])[0]; verifier=(body.get('code_verifier') or [''])[0]; grant=(body.get('grant_type') or [''])[0]
    with OAUTH_LOCK:
        data=_oauth_load(); row=data.get('codes',{}).pop(code,None)
        if not row or grant!='authorization_code' or time.time()-row.get('created_at',0)>600 or _pkce(verifier)!=row.get('code_challenge'):
            return JSONResponse(status_code=400, content={'error':'invalid_grant'})
        token=secrets.token_urlsafe(32); data.setdefault('tokens',{})[token]={'client_id':row['client_id'],'scope':row.get('scope','mcp'),'expires_at':time.time()+86400}; _oauth_save(data)
    return {'access_token':token,'token_type':'Bearer','expires_in':86400,'scope':row.get('scope','mcp')}

@app.middleware('http')
async def mcp_auth_middleware(request: Request, call_next):
    if request.url.path=='/mcp' and (request.client.host if request.client else '') not in {'127.0.0.1','::1'}:
        auth=request.headers.get('authorization','')
        token=auth[7:] if auth.lower().startswith('bearer ') else ''
        if not _oauth_token_ok(token):
            proto=request.headers.get('x-forwarded-proto') or request.url.scheme
            host=request.headers.get('x-forwarded-host') or request.headers.get('host')
            base=f'{proto}://{host}'.rstrip('/')
            return JSONResponse(status_code=401, content={'error':'unauthorized','authorization_uri':base+'/oauth/authorize'})
    return await call_next(request)

DISPLAY = os.environ.get('DISPLAY', ':99')
SAFE_ACTIONS = {
    'screenshot', 'observe', 'move', 'click', 'double_click', 'drag', 'scroll',
    'key', 'hotkey', 'type', 'wait', 'browser_status', 'browser_open',
    'browser_screenshot', 'browser_state', 'windows', 'mouse_position'
}
RISKY_ACTIONS = {'run_shell', 'delete_file', 'upload_data', 'send_message'}
_BROWSER = None
_PAGE = None
_BROWSER_ERROR = None

class _BrowserManager:
    """Own the Playwright sync API on one dedicated thread.

    Playwright sync objects are thread-affine; every browser operation is routed
    through this single worker so FastAPI request threads never touch them directly.
    """
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='airi-playwright')
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._lock = threading.Lock()
        self.last_error = None
        self.auth_profile = None

    def _run(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs).result()

    def _auth_path(self):
        if not self.auth_profile: return None
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', self.auth_profile): raise ValueError('invalid auth profile name')
        path=AI/'auth'/f'{self.auth_profile}.json'; path.parent.mkdir(parents=True,exist_ok=True)
        try: path.parent.chmod(0o700)
        except OSError: pass
        return path

    def _ensure(self):
        global _BROWSER, _PAGE, _BROWSER_ERROR
        from playwright.sync_api import sync_playwright
        if self._page is not None:
            try:
                _ = self._page.url
                return self._page
            except Exception:
                self._close_worker()
        if self._pw is None:
            self._pw = sync_playwright().start()
        launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        last_error = None
        mode = os.environ.get('AIRI_BROWSER_HEADLESS', 'auto').strip().lower()
        if mode in {'1', 'true', 'yes', 'on', 'headless'}:
            modes = [True]
        elif mode in {'0', 'false', 'no', 'off', 'headed'}:
            modes = [False] if gui_available() else [True]
        else:
            modes = [False, True] if gui_available() else [True]
        for headless in modes:
            # A failed headed launch gets one bounded attempt before headless fallback.
            for _attempt in range(1):
                try:
                    self._browser = self._pw.chromium.launch(headless=headless, args=launch_args, timeout=15000)
                    auth_file = self._auth_path() if self.auth_profile else None
                    self._context = self._browser.new_context(viewport={'width':1280,'height':800}, storage_state=str(auth_file) if auth_file and auth_file.exists() else None)
                    self._page = self._context.new_page()
                    self.last_error = None
                    _BROWSER, _PAGE, _BROWSER_ERROR = self._browser, self._page, None
                    return self._page
                except Exception as exc:
                    last_error = exc
                    self._close_worker()
                    time.sleep(0.5)
        self.last_error = str(last_error)
        _BROWSER_ERROR = self.last_error
        raise RuntimeError(f'Browser unavailable: {last_error}')

    def _close_worker(self):
        global _BROWSER, _PAGE
        try:
            if self._page is not None: self._page.close()
        except Exception: pass
        try:
            if self._context is not None: self._context.close()
        except Exception: pass
        try:
            if self._browser is not None: self._browser.close()
        except Exception: pass
        self._page = None
        self._context = None
        self._browser = None
        _BROWSER, _PAGE = None, None

    def call(self, fn, *args, **kwargs):
        with self._lock:
            return self._run(fn, *args, **kwargs)

    def status(self):
        def _status():
            try:
                page=self._ensure()
                return {'available':True,'open':page is not None,'url':page.url,'error':None}
            except Exception as exc:
                return {'available':False,'open':False,'url':None,'error':str(exc)}
        return self.call(_status)

    def state(self):
        def _state():
            try:
                page=self._ensure()
                return {'ok':True,'url':page.url,'title':page.title(),'status':page.locator('#status').inner_text(timeout=3000) if page.locator('#status').count() else None,'field':page.locator('#field').input_value(timeout=3000) if page.locator('#field').count() else None,'scroll_y':page.evaluate('window.scrollY'),'drag_box':page.locator('#drag').bounding_box(timeout=3000) if page.locator('#drag').count() else None,'click_box':page.locator('#click').bounding_box(timeout=3000) if page.locator('#click').count() else None}
            except Exception as exc:
                self._close_worker()
                return {'ok':False,'url':None,'title':'','error':str(exc),'browser_error':self.last_error}
        return self.call(_state)

    def open(self, url, wait_until='domcontentloaded'):
        def _open():
            last_error = None
            for attempt in range(2):
                try:
                    page=self._ensure(); response=page.goto(url, wait_until=wait_until, timeout=30000)
                    status=response.status if response is not None else None
                    classification='auth_or_permission' if status in {401,403} else 'ok'
                    return {'ok':True,'url':page.url,'title':page.title(),'status':status,'classification':classification,'auth_profile':self.auth_profile,'attempt':attempt+1}
                except Exception as exc:
                    last_error=str(exc)
                    low=last_error.lower()
                    if 'err_blocked_by_administrator' in low: cls='policy_block'
                    elif 'timeout' in low: cls='timeout'
                    elif 'net::' in low or 'connection' in low: cls='network_error'
                    elif 'certificate' in low or 'ssl' in low: cls='tls_error'
                    else: cls='browser_error'
                    self._close_worker()
                    if attempt == 0:
                        try: self._ensure()
                        except Exception as restart_exc: last_error=str(restart_exc)
                        time.sleep(0.25)
            return {'ok':False,'error':last_error or 'browser_open_failed','url':None,'status':None,'classification':cls if 'cls' in locals() else 'browser_error','auth_profile':self.auth_profile,'attempt':2}
        return self.call(_open)

    def auth_set(self, name):
        def _set():
            if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', str(name)): raise ValueError('invalid auth profile name')
            path=AI/'auth'/f'{name}.json'; path.parent.mkdir(parents=True,exist_ok=True); path.parent.chmod(0o700)
            self.auth_profile=str(name); self._close_worker()
            return {'ok':True,'profile':self.auth_profile,'exists':path.exists()}
        return self.call(_set)

    def auth_save(self, name=None):
        def _save():
            profile=str(name or self.auth_profile or '').strip()
            if not profile or not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', profile): raise ValueError('valid auth profile name required')
            self.auth_profile=profile; self._ensure(); path=self._auth_path(); self._context.storage_state(path=str(path))
            try: path.chmod(0o600)
            except OSError: pass
            return {'ok':True,'profile':profile,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sensitive':True}
        return self.call(_save)

    def auth_status(self, name):
        profile=str(name or '').strip()
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', profile): raise ValueError('invalid auth profile name')
        path=AI/'auth'/f'{profile}.json'
        return {'ok':True,'profile':profile,'exists':path.exists(),'bytes':path.stat().st_size if path.exists() else 0,'current':self.auth_profile==profile}

    def _human_challenge_info(self):
        page = self._ensure()
        title = (page.title() or '').strip()
        try:
            body = page.locator('body').inner_text(timeout=3000)[:12000]
        except Exception:
            body = ''
        try:
            frame_urls = [f.url for f in page.frames if f.url]
        except Exception:
            frame_urls = []
        hay = ' '.join([title, body, ' '.join(frame_urls)]).lower()
        markers = ['captcha','recaptcha','hcaptcha','turnstile','verify you are human','i am not a robot','checking your browser','performing security verification','security challenge','human verification','attention required']
        matched = [m for m in markers if m in hay]
        selectors = []
        for sel in ['iframe[src*=\"captcha\"]','iframe[src*=\"recaptcha\"]','iframe[src*=\"hcaptcha\"]','iframe[src*=\"challenges.cloudflare.com\"]','[data-sitekey]','[class*=\"captcha\"]','[id*=\"captcha\"]','[class*=\"turnstile\"]','[id*=\"turnstile\"]']:
            try:
                if page.locator(sel).count(): selectors.append(sel)
            except Exception:
                pass
        detected = bool(matched or selectors)
        return {'detected':detected,'url':page.url,'title':title,'matched_markers':matched,'matched_selectors':selectors,'action_url':page.url if detected else None,'message':'Human verification detected; solve it in the Airi-PC browser, then resume.' if detected else 'No human verification detected.'}

    def human_challenge_status(self):
        def _status():
            try: return {'ok':True,**self._human_challenge_info()}
            except Exception as exc: return {'ok':False,'detected':False,'error':str(exc),'url':None}
        return self.call(_status)

    def wait_for_human(self, timeout=300, poll=1.0):
        timeout=max(1,min(int(timeout),900)); poll=max(0.25,min(float(poll),5.0)); started=time.time(); last=None
        while time.time()-started < timeout:
            last=self.human_challenge_status()
            if last.get('ok') and not last.get('detected'):
                last['waited_seconds']=round(time.time()-started,2); last['resumed']=True; return last
            time.sleep(poll)
        if last is None: last=self.human_challenge_status()
        last['waited_seconds']=round(time.time()-started,2); last['resumed']=False; last['timed_out']=True; return last

    def screenshot(self):
        def _screenshot():
            last_error = None
            for attempt in range(2):
                try:
                    page=self._ensure()
                    b=page.screenshot(type='png', timeout=10000)
                    return {'ok':True,'format':'png','width':1280,'height':800,'data_base64':base64.b64encode(b).decode(),'method':'playwright'}
                except Exception as exc:
                    last_error=str(exc)
                    self._close_worker()
                    if attempt == 0:
                        try: self._ensure()
                        except Exception as restart_exc: last_error=str(restart_exc)
                        time.sleep(0.25)
            try:
                img=screenshot_image(); b=io.BytesIO(); img.save(b,format='PNG')
                return {'ok':True,'format':'png','width':img.width,'height':img.height,'data_base64':base64.b64encode(b.getvalue()).decode(),'method':'x11_fallback','warning':'Playwright page screenshot failed; returned the live Airi-PC display capture.','browser_error':last_error}
            except Exception as fallback_exc:
                self.last_error=f'{last_error}; fallback={fallback_exc}'
                return {'ok':False,'error':self.last_error}
        return self.call(_screenshot)

    def text(self):
        def _text():
            page=self._ensure(); text=page.locator('body').inner_text(timeout=5000); paragraphs=page.locator('p'); first=paragraphs.first.inner_text().strip() if paragraphs.count() else next((x.strip() for x in text.split('\\n') if x.strip()), '')
            return {'url':page.url,'title':page.title(),'text':text,'paragraphs':paragraphs.all_inner_texts() if paragraphs.count() else [],'first_paragraph':first}
        return self.call(_text)

    def links(self, limit=20):
        def _links():
            page=self._ensure(); out=[]
            for a in page.locator('a').all()[:100]:
                try:
                    href=a.get_attribute('href'); text=(a.inner_text() or '').strip()
                    if href and href.startswith(('http://','https://')): out.append({'url':href,'title':text})
                    if len(out)>=int(limit): break
                except Exception: continue
            return out
        return self.call(_links)

    def close(self):
        try:
            self.call(self._close_worker)
        except Exception:
            pass
        self.executor.shutdown(wait=False, cancel_futures=True)

_browser_manager = _BrowserManager()
atexit.register(_browser_manager.close)

def browser():
    return _browser_manager

class Move(BaseModel):
    x:int
    y:int
class Click(BaseModel):
    x:int
    y:int
    button:str='left'
    clicks:int=1
class Scroll(BaseModel):
    amount:int
class Key(BaseModel):
    key:str
class Hotkey(BaseModel):
    keys:List[str]
class TypeText(BaseModel):
    text:str
    interval:float=0.02
class Wait(BaseModel):
    seconds:float=Field(ge=0,le=30)
class FindText(BaseModel):
    text:str
    click:bool=False
class BrowserOpen(BaseModel):
    url:str
    wait_until:str='domcontentloaded'
class ActionRequest(BaseModel):
    action:str
    payload:Dict[str,Any]=Field(default_factory=dict)
    confirmation:Optional[str]=None

def _pyautogui():
    import pyautogui
    pyautogui.PAUSE = 0.05
    pyautogui.FAILSAFE = False
    return pyautogui

def gui_available() -> bool:
    return bool(os.environ.get('DISPLAY'))

def screenshot_image() -> Image.Image:
    if not gui_available():
        raise RuntimeError('GUI display is unavailable in this sandbox')
    last=None
    for _ in range(2):
        try:
            return ImageGrab.grab()
        except Exception as exc:
            last=exc
            time.sleep(0.15)
    try:
        return _pyautogui().screenshot()
    except Exception as exc:
        raise RuntimeError(f'GUI screenshot failed: {last}; pyautogui fallback: {exc}') from exc

def image_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format='PNG'); return base64.b64encode(buf.getvalue()).decode()

def ocr(img: Image.Image) -> List[Dict[str, Any]]:
    if pytesseract is None:
        return []
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        out = []
        for i, text in enumerate(data['text']):
            text = (text or '').strip()
            if not text: continue
            out.append({'text': text, 'x': int(data['left'][i]), 'y': int(data['top'][i]),
                        'w': int(data['width'][i]), 'h': int(data['height'][i]),
                        'confidence': float(data['conf'][i]) if str(data['conf'][i]).strip() not in ('', '-1') else -1})
        return out
    except Exception:
        return []

def windows_info() -> List[Dict[str, Any]]:
    try:
        raw = subprocess.check_output(['xprop','-root','_NET_CLIENT_LIST'], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    ids = [x.strip() for x in raw.split('=')[-1].split(',') if x.strip()]
    out=[]
    for wid in ids:
        try:
            name = subprocess.check_output(['xprop','-id',wid,'_NET_WM_NAME'], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            name=''
        out.append({'id': wid, 'name': name})
    return out

def mouse_position() -> Dict[str, int]:
    pos = _pyautogui().position()
    return {'x': int(pos.x), 'y': int(pos.y)}

def perform(action: str, p: Dict[str, Any]) -> Any:
    if action == 'click_element':
        query=p['text']; img=screenshot_image(); matches=[x for x in ocr(img) if query.casefold() in x['text'].casefold()]
        if matches:
            m=matches[0]; _pyautogui().click(m['x']+m['w']//2,m['y']+m['h']//2); return {'ok':True,'match':m,'method':'ocr'}
        # The browser can be visible to Playwright while desktop OCR does not
        # reliably capture its rendered text. Use the same browser session as a
        # deterministic fallback, then verify through the normal browser state.
        try:
            def _dom_click():
                page=browser()._ensure()
                locator=page.get_by_text(query, exact=True).first
                if locator.count() == 0:
                    locator=page.get_by_text(query, exact=False).first
                if locator.count() == 0:
                    locator=page.get_by_placeholder(query, exact=True).first
                if locator.count() == 0:
                    locator=page.get_by_role('textbox', name=query, exact=True).first
                if locator.count() == 0:
                    locator=page.locator(f'#{query}').first
                if locator.count() == 0:
                    raise RuntimeError(f'Element text/placeholder/role/id not found: {query}')
                locator.scroll_into_view_if_needed()
                locator.click(timeout=5000)
                return {'ok':True,'method':'dom','text':query}
            return browser().call(_dom_click)
        except Exception as exc:
            raise RuntimeError(f'Element text not found: {query}; OCR and DOM fallback failed: {exc}')
    if action == 'screenshot':
        img=screenshot_image(); return {'format':'png','width':img.width,'height':img.height,'data_base64':image_b64(img)}
    pa = _pyautogui()
    if action == 'move': pa.moveTo(p['x'], p['y']); return {'ok':True}
    if action == 'click': pa.click(button=p.get('button','left'), clicks=p.get('clicks',1)); return {'ok':True}
    if action == 'double_click': pa.doubleClick(button=p.get('button','left')); return {'ok':True}
    if action == 'drag': pa.moveTo(p['x1'],p['y1']); pa.dragTo(p['x2'],p['y2'],duration=p.get('duration',0.25)); return {'ok':True}
    if action == 'scroll': pa.scroll(p['amount']); return {'ok':True}
    if action == 'key': pa.press(p['key']); return {'ok':True}
    if action == 'hotkey': pa.hotkey(*p['keys']); return {'ok':True}
    if action == 'type': pa.write(p['text'], interval=p.get('interval',0.02)); return {'ok':True}
    if action == 'wait': time.sleep(p['seconds']); return {'ok':True}
    if action == 'windows': return {'windows':windows_info()}
    if action == 'mouse_position': return mouse_position()
    if action == 'browser_status':
        return browser().status()
    if action == 'browser_state':
        return browser().state()
    if action == 'browser_text':
        return browser().text()
    if action == 'browser_open':
        return browser().open(p['url'], p.get('wait_until','domcontentloaded'))
    if action == 'browser_screenshot':
        return browser().screenshot()
    raise ValueError(f'Unsupported action: {action}')

def observe() -> Dict[str, Any]:
    img = screenshot_image()
    return {'ok':True,'display':DISPLAY,'resolution':f'{img.width}x{img.height}',
            'screenshot': image_b64(img),'ocr':ocr(img),'windows':windows_info(),
            'browser': _browser_manager.status()}

@app.get('/status')
def status():
    if gui_available():
        try:
            img=screenshot_image(); return {'ok':True,'display':DISPLAY,'gui_available':True,'resolution':f'{img.width}x{img.height}','version':'2.0','source_sha':AIRI_BOOTSTRAP_SHA or None}
        except Exception: pass
    return {'ok':True,'display':DISPLAY if gui_available() else None,'gui_available':False,'resolution':None,'version':'2.0','mode':'headless','source_sha':AIRI_BOOTSTRAP_SHA or None}

@app.get('/ready')
def ready():
    checks = {'status': False, 'gui': False, 'browser': False, 'mcp': False, 'source_match': True}
    try:
        st = status(); checks['status'] = bool(st.get('ok')); checks['gui'] = bool(st.get('gui_available')) and st.get('resolution') == '1280x800'
    except Exception as exc:
        return JSONResponse(status_code=503, content={'ready': False, 'checks': checks, 'error': str(exc)})
    try:
        bs = browser().status(); checks['browser'] = bool(bs.get('available')) and bool(bs.get('open')) and not bs.get('error')
    except Exception:
        checks['browser'] = False
    try:
        names = tools()['tools']; checks['mcp'] = len(names) >= 20 and 'computer_status' in names and 'computer_browser_state' in names
    except Exception:
        checks['mcp'] = False
    expected_sha = os.environ.get('AIRI_EXPECTED_SHA', '').strip()
    if expected_sha and AIRI_BOOTSTRAP_SHA:
        checks['source_match'] = AIRI_BOOTSTRAP_SHA == expected_sha
    elif expected_sha and not AIRI_BOOTSTRAP_SHA:
        checks['source_match'] = False
    ready_now = all(checks.values())
    payload = {'ready': ready_now, 'checks': checks, 'source_sha': AIRI_BOOTSTRAP_SHA or None}
    return JSONResponse(status_code=200 if ready_now else 503, content=payload)


@app.get('/screenshot')
def screenshot():
    img=screenshot_image(); return JSONResponse({'format':'png','width':img.width,'height':img.height,'data_base64':image_b64(img)})

@app.get('/observe')
def observe_endpoint(): return observe()

@app.post('/find-text')
def find_text(req: FindText):
    img=screenshot_image(); items=ocr(img); target=req.text.casefold(); matches=[x for x in items if target in x['text'].casefold()]
    if req.click and matches:
        m=matches[0]; _pyautogui().click(m['x']+m['w']//2,m['y']+m['h']//2)
    return {'found':bool(matches),'matches':matches}

@app.post('/act-verify')
def act_verify(req: ActionRequest):
    if req.action in RISKY_ACTIONS and req.confirmation != 'ALLOW': raise HTTPException(403,'Confirmation required')
    before=screenshot_image(); before_b64=image_b64(before); result=perform(req.action, req.payload); time.sleep(0.15); after=screenshot_image(); after_b64=image_b64(after); diff=ImageChops.difference(before,after).convert('L'); bbox=diff.getbbox(); changed=sum(1 for v in diff.getdata() if v>10); total=diff.width*diff.height
    return {'ok':True,'action':req.action,'result':result,'changed_ratio':changed/total,'changed_bbox':bbox,'before_screenshot':{'format':'png','width':before.width,'height':before.height,'data_base64':before_b64},'after_screenshot':{'format':'png','width':after.width,'height':after.height,'data_base64':after_b64}}

@app.post('/action')
def action(req: ActionRequest):
    if req.action in RISKY_ACTIONS and req.confirmation != 'ALLOW': raise HTTPException(403,'Confirmation required')
    return perform(req.action, req.payload)

@app.post('/mouse/move')
def mouse_move(m: Move): return perform('move',m.model_dump())
@app.post('/mouse/click')
def mouse_click(c: Click): return perform('click',c.model_dump())
@app.post('/mouse/scroll')
def mouse_scroll(s: Scroll): return perform('scroll',s.model_dump())
@app.post('/keyboard/key')
def key(k: Key): return perform('key',k.model_dump())
@app.post('/keyboard/type')
def type_text(t: TypeText): return perform('type',t.model_dump())
@app.get('/windows')
def windows(): return {'windows':windows_info()}

@app.get('/mouse/position')
def mouse_position_endpoint(): return mouse_position()

@app.get('/cleanup/scan')
def cleanup_scan_endpoint():
    if cleanup_scan is None: raise HTTPException(503, 'Cleanup module unavailable')
    return cleanup_scan()

@app.post('/cleanup/clean-safe')
def cleanup_safe_endpoint(req: Dict[str,Any] = {}):
    if cleanup_safe is None: raise HTTPException(503, 'Cleanup module unavailable')
    return cleanup_safe(req.get('max_bytes'))

SELF_TEST_HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Airi-PC Self Test</title><style>body{font-family:sans-serif;padding:40px;height:2400px}button,input{font-size:28px;margin:12px;padding:16px}.drag{display:inline-block;font-size:24px;margin:12px;padding:30px;border:4px dashed #333}.spacer{height:1000px}#status{font-size:32px;margin-top:24px;position:sticky;top:10px;background:#fff}</style></head><body><h1>Airi-PC Self Test</h1><button id="click" onclick="status.textContent='AIRI_CLICK_OK'">AIRI_CLICK_TARGET</button><button id="drag" class="drag">AIRI_DRAG_TARGET</button><input id="field" placeholder="AIRI_INPUT_FIELD"><button id="enter" onclick="status.textContent='AIRI_BUTTON_OK'">AIRI_BUTTON</button><div class="spacer"></div><div id="status">AIRI_READY</div><script>const status=document.getElementById('status');const field=document.getElementById('field');const drag=document.getElementById('drag');window.addEventListener('keydown',e=>{if(e.key==='Enter')status.textContent='AIRI_ENTER_OK';if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key.toLowerCase()==='a')status.textContent='AIRI_HOTKEY_OK'});field.addEventListener('input',()=>{status.textContent='AIRI_TYPE_OK'});let dragging=false;drag.addEventListener('mousedown',()=>{dragging=true;status.textContent='AIRI_DRAG_START'});window.addEventListener('mouseup',()=>{if(dragging){dragging=false;status.textContent='AIRI_DRAG_OK'}});window.addEventListener('scroll',()=>{if(window.scrollY>50)status.textContent='AIRI_SCROLL_OK'});</script></body></html>'''


@app.get('/self-test', response_class=HTMLResponse)
def self_test(): return HTMLResponse(SELF_TEST_HTML)


@app.post('/code/project-analyze')
def code_project_analyze(req: Dict[str,Any] = {}): return code_analyze(req.get('path','.'))
@app.post('/code/tree')
def code_project_tree(req: Dict[str,Any] = {}): return code_analyze(req.get('path','.'))['tree']
@app.post('/code/file-read')
def code_file_read(req: Dict[str,Any]): return code_read(req['path'])
@app.post('/code/file-search')
def code_file_search(req: Dict[str,Any]): return code_search(req['query'],req.get('path','.'),req.get('limit',100))
def _direct_scope(req: Dict[str,Any]) -> list[str]:
    scope = req.get('scope')
    if not scope: raise HTTPException(403, 'declared scope is required for direct coding mutations')
    return [str(x) for x in scope]

def _guarded_file_change(req: Dict[str,Any], mode: str):
    scope = _direct_scope(req); path = req['path']; snap = code_snapshot([path], label=f'direct-{mode}:{path}')
    try:
        if mode == 'write': result = code_write(path, req['content'], declared_scope=scope)
        else: result = code_patch(path, req['old'], req['new'], req.get('replace_all',False), declared_scope=scope)
        test_command = req.get('test_command','')
        verification = code_test(test_command, req.get('cwd','.'), req.get('timeout',170)) if test_command else {'returncode':0,'stdout':'no test command','stderr':''}
        if verification.get('returncode') != 0:
            code_restore_snapshot(snap['snapshot']); raise HTTPException(409, {'error':'verification_failed','verification':verification,'rolled_back':True})
        guard = code_guardrail_check(req.get('cwd','.'), scope)
        if not guard.get('ok'):
            code_restore_snapshot(snap['snapshot']); raise HTTPException(409, {'error':'guardrail_violation','guardrails':guard,'rolled_back':True})
        checkpoint(req.get('goal',f'direct {mode}'), scope, int(req.get('step',0)), f'{mode} completed', [path])
        return {'ok':True,'result':result,'verification':verification,'guardrails':guard,'snapshot':snap}
    except HTTPException: raise
    except Exception:
        try: code_restore_snapshot(snap['snapshot'])
        except Exception: pass
        raise

@app.post('/code/file-write')
def code_file_write(req: Dict[str,Any]): return _guarded_file_change(req,'write')
@app.post('/code/file-patch')
def code_file_patch(req: Dict[str,Any]): return _guarded_file_change(req,'patch')
@app.post('/code/terminal-run')
def code_terminal_run(req: Dict[str,Any]):
    scope = _direct_scope(req); code_scope_check([req.get('cwd','.')], scope)
    result = code_shell(req['command'],req.get('cwd','.'),req.get('timeout',120),req.get('allow_shell',False))
    if result.get('returncode') != 0: return result
    guard = code_guardrail_check(req.get('cwd','.'), scope)
    if not guard.get('ok'): raise HTTPException(409, {'error':'guardrail_violation','guardrails':guard})
    return {'result':result,'guardrails':guard}
@app.post('/code/test-run')
def code_test_run(req: Dict[str,Any] = {}): return code_test(req.get('command','pytest -q'),req.get('cwd','.'),req.get('timeout',170))
@app.post('/code/build-run')
def code_build_run(req: Dict[str,Any] = {}): return code_build(req.get('command','python -m compileall -q .'),req.get('cwd','.'),req.get('timeout',170))
@app.post('/code/lint')
def code_lint_endpoint(req: Dict[str,Any] = {}): return code_lint(req.get('command','python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")'),req.get('cwd','.'),req.get('timeout',170))
@app.post('/code/git-status')
def code_git_status_endpoint(req: Dict[str,Any] = {}): return code_git_status(req.get('path','.'))
@app.post('/code/git-diff')
def code_git_diff_endpoint(req: Dict[str,Any] = {}): return code_git_diff(req.get('path','.'))
@app.post('/code/git-log')
def code_git_log_endpoint(req: Dict[str,Any] = {}): return code_git_log(req.get('path','.'),req.get('n',10))
@app.post('/code/git-commit')
def code_git_commit_endpoint(req: Dict[str,Any]): return code_git_commit(req['message'],req.get('path','.'))
@app.get('/code/skills')
def code_skills(): return list_skills()
@app.post('/code/skill-load')
def code_skill_load(req: Dict[str,Any]): return load_skill(req['name'])
@app.post('/code/skill-create')
def code_skill_create(req: Dict[str,Any]): return create_skill(req['name'],req['description'],req['instructions'],req.get('tools'))
@app.post('/code/skill-update')
def code_skill_update(req: Dict[str,Any]): return update_skill(req['name'],req['content'])
@app.post('/code/skill-test')
def code_skill_test(req: Dict[str,Any]): return test_skill(req['name'])
@app.post('/code/skill-delete')
def code_skill_delete(req: Dict[str,Any]): return delete_skill(req['name'])
@app.get('/code/memory')
def code_memory_read(): return memory_read()
@app.post('/code/memory')
def code_memory_update(req: Dict[str,Any]): return memory_update(req['entry'])
@app.post('/code/apply-fix')
def code_apply_fix_endpoint(req: Dict[str,Any]): return code_apply_fix(req['path'],req['old'],req['new'],req.get('test_command',''))
@app.post('/code/verify-change')
def code_verify_change_endpoint(req: Dict[str,Any]): return code_verify_change(req['path'],req['test_command'])
@app.post('/code/agent')
def code_agent_endpoint(req: Dict[str,Any]): return code_agent(req['goal'],req.get('project_path','.'),req.get('max_attempts',5),req.get('steps'),req.get('scope'),req.get('changes'),req.get('test_command',''))
@app.get('/code/project-context')
def code_project_context_endpoint(req: Dict[str,Any] = {}): return code_project_context(req.get('path','.'))
@app.post('/code/task-start')
def code_task_start(req: Dict[str,Any]): return task_start(req['goal'],req['steps'],req.get('scope'))
@app.get('/code/task')
def code_task_read_endpoint(): return task_read()
@app.post('/code/task-update')
def code_task_update_endpoint(req: Dict[str,Any]): return task_update(req['index'],req['status'],req.get('note',''))
@app.post('/code/task-finish')
def code_task_finish_endpoint(req: Dict[str,Any] = {}): return task_finish(req.get('outcome','done'),req.get('note',''))
@app.post('/code/scope-check')
def code_scope_check_endpoint(req: Dict[str,Any]): return code_scope_check(req['paths'],req['scope'])
@app.post('/code/diff-summary')
def code_diff_summary_endpoint(req: Dict[str,Any] = {}): return code_diff_summary(req.get('path','.'),req.get('scope'),req.get('allow_test_changes',False),req.get('allow_security_changes',False))
@app.post('/code/guardrails')
def code_guardrails_endpoint(req: Dict[str,Any] = {}): return code_guardrail_check(req.get('path','.'),req.get('scope'),req.get('allow_test_changes',False),req.get('allow_security_changes',False))
@app.post('/code/snapshot')
def code_snapshot_endpoint(req: Dict[str,Any]): return code_snapshot(req['paths'],req.get('label','task'))
@app.post('/code/restore-snapshot')
def code_restore_snapshot_endpoint(req: Dict[str,Any]): return code_restore_snapshot(req['snapshot'])
@app.post('/code/prepare-commit')
def code_prepare_commit_endpoint(req: Dict[str,Any] = {}): return code_prepare_commit(req.get('path','.'),req.get('scope'))
@app.post('/code/commit')
def code_commit_endpoint(req: Dict[str,Any]): return code_atomic_commit(req['message'],req.get('project_path','.'),req.get('scope'),req.get('allow_test_changes',False),req.get('allow_security_changes',False))
@app.post('/code/autonomous-cycle')
def code_autonomous_cycle_endpoint(req: Dict[str,Any]): return code_autonomous_change_cycle(req['changes'],req.get('project_path','.'),req.get('test_command',''),req.get('scope'),req.get('max_attempts',5))

@app.get('/health')
def health_endpoint(): return health_check()
@app.get('/recovery')
def recovery_endpoint(): return recovery_read()
@app.post('/recovery/checkpoint')
def recovery_checkpoint_endpoint(req: Dict[str,Any]): return checkpoint(req['goal'],req.get('scope',[]),req.get('step',0),req.get('note',''),req.get('artifacts'),req.get('status','active'))
@app.post('/recovery/finish')
def recovery_finish_endpoint(req: Dict[str,Any] = {}): return recovery_finish(req.get('status','done'),req.get('note',''))
@app.post('/memory/decision')
def decision_record_endpoint(req: Dict[str,Any]): return record_decision(req['decision'],req.get('reason',''),req.get('evidence'),req.get('files'),req.get('commit'),req.get('result',''))
@app.get('/memory/decisions')
def decisions_endpoint(limit: int = 50): return decisions(limit)
@app.get('/persistence/status')
def persistence_status_endpoint(): return persistence_status()
@app.post('/persistence/persist')
def persistence_persist_endpoint(req: Dict[str,Any]): return persist_current(req['message'],req.get('branch'),req.get('push',True),req.get('scope'))
def _research_with_browser(req: Dict[str,Any]):
    topic=req['topic']; urls=req.get('urls'); max_sources=req.get('max_sources',5)
    if urls: return research(topic, urls, max_sources)
    search_url='https://html.duckduckgo.com/html/?q='+urllib.parse.quote_plus(topic)
    opened=browser().open(search_url,'domcontentloaded')
    if not opened.get('ok'):
        return {'ok':False,'topic':topic,'sources':[],'errors':[{'url':search_url,'error':opened.get('error','browser search failed')}],'source_count':0}
    discovered=[]; seen=set()
    for item in browser().links(max_sources*4):
        url=item.get('url','')
        if not url.startswith(('http://','https://')) or url in seen: continue
        if urllib.parse.urlparse(url).netloc.lower().endswith('duckduckgo.com'): continue
        seen.add(url); discovered.append(url)
        if len(discovered)>=max_sources: break
    return research(topic, discovered, max_sources)

@app.post('/research')
def research_endpoint(req: Dict[str,Any]): return _research_with_browser(req)
@app.post('/browser/auth/set')
def browser_auth_set_endpoint(req: Dict[str,Any]): return browser().auth_set(req['profile'])
@app.post('/browser/auth/save')
def browser_auth_save_endpoint(req: Dict[str,Any]): return browser().auth_save(req.get('profile'))
@app.get('/browser/auth/status')
def browser_auth_status_endpoint(profile: str): return browser().auth_status(profile)
@app.get('/browser/human/status')
def browser_human_status_endpoint(): return browser().human_challenge_status()
@app.post('/browser/human/wait')
def browser_human_wait_endpoint(req: Dict[str,Any] = {}): return browser().wait_for_human(req.get('timeout',300), req.get('poll',1.0))
@app.get('/permissions')
def permission_endpoint(): return permission_check()
@app.get('/runtime/preflight')
def runtime_preflight_endpoint(): return runtime_preflight()
@app.get('/integrations')
def integrations_endpoint(): return integration_status()
@app.post('/cleanup/prune-backups')
def cleanup_prune_backups_endpoint(req: Dict[str,Any] = {}):
    from cleanup import prune_backups
    return prune_backups(req.get('max_entries',50), req.get('max_age_days',30), req.get('dry_run',True))
@app.get('/scheduler')
def scheduler_status_endpoint(): return scheduler_status()
@app.post('/scheduler/schedule')
def scheduler_schedule_endpoint(req: Dict[str,Any]): return schedule_job(req['name'],req['action'],req['interval_seconds'],req.get('run_now',False))
@app.post('/scheduler/cancel')
def scheduler_cancel_endpoint(req: Dict[str,Any]): return cancel_job(req['name'])

def _control_plane_bootstrap():
    global CONTROL_PLANE_BOOTSTRAPPED
    if not CONTROL_PLANE_BOOTSTRAPPED:
        CONTROL_PLANE.bootstrap(tools()["tools"])
        CONTROL_PLANE_BOOTSTRAPPED = True
    return CONTROL_PLANE


@app.get('/control-plane')
def control_plane_status_endpoint():
    return _control_plane_bootstrap().status()


@app.get('/tools')
def tools():
    names=['computer_status','computer_observe','computer_screenshot','computer_find_text','computer_click_element','computer_click','computer_double_click','computer_move','computer_drag','computer_scroll','computer_key','computer_hotkey','computer_type','computer_wait','computer_windows','computer_mouse_position','computer_browser_open','computer_browser_status','computer_browser_screenshot','computer_browser_state','computer_browser_text','computer_act_verify','computer_cleanup_scan','computer_cleanup_safe','computer_project_analyze','computer_project_tree','computer_file_read','computer_file_search','computer_file_write','computer_file_patch','computer_terminal_run','computer_test_run','computer_build_run','computer_lint','computer_git_status','computer_git_diff','computer_git_log','computer_git_commit','computer_skill_list','computer_skill_load','computer_skill_create','computer_skill_update','computer_skill_test','computer_skill_delete','computer_project_memory_read','computer_project_memory_update','computer_code_apply_fix','computer_code_verify_change','computer_code_agent','computer_project_context','computer_task_start','computer_task_read','computer_task_update','computer_task_finish','computer_scope_check','computer_diff_summary','computer_guardrails','computer_snapshot','computer_restore_snapshot','computer_prepare_commit','computer_code_commit','computer_autonomous_cycle','computer_health','computer_recovery_read','computer_recovery_checkpoint','computer_recovery_finish','computer_decision_record','computer_decisions','computer_persistence_status','computer_persist','computer_research','computer_backup_prune','computer_scheduler_status','computer_scheduler_schedule','computer_scheduler_cancel','computer_browser_auth_set','computer_browser_auth_save','computer_browser_auth_status','computer_browser_human_status','computer_browser_human_wait','computer_permission_check','computer_runtime_preflight','computer_integrations','computer_control_plane','computer_autonomous_goal','computer_terminal_start','computer_terminal_status','computer_terminal_list','computer_terminal_attach','computer_terminal_detach','computer_terminal_cancel','computer_terminal_cleanup','computer_task_add_node','computer_task_add_dependency','computer_task_skip','computer_task_runnable','computer_context_pack','computer_verify_deliverable','computer_experience_record','computer_experience_match','computer_model_choose','computer_reasoning_start','computer_reasoning_status','computer_reasoning_next_action','computer_reasoning_observe','computer_reasoning_mark_step','computer_reasoning_replan','computer_reasoning_feedback','computer_reasoning_finish','computer_reasoning_goal']
    return {'name':'Airi Computer','version':'2.0','tools':names}

@app.post('/mcp')
def mcp(req: Dict[str,Any]):
    method=req.get('method'); rid=req.get('id')
    if method=='initialize': return {'jsonrpc':'2.0','id':rid,'result':{'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{'name':'Airi Computer','version':'2.0'}}}
    if method in ('notifications/initialized','ping'): return {'jsonrpc':'2.0','id':rid,'result':{}}
    if method=='tools/list':
        names=tools()['tools']; reasoning_schema={'type':'object','properties':{'goal':{'type':'string'},'plan':{'type':'array'},'steps':{'type':'array'},'scope':{'type':'array'},'metadata':{'type':'object'},'run_id':{'type':'string'},'step_id':{'type':'string'},'status':{'type':'string'},'observation':{},'operation':{'type':'string'},'success':{'type':'boolean'},'result':{},'error':{'type':'string'},'tool':{'type':'string'},'task':{'type':'string'},'evidence':{'type':'array'},'phase':{'type':'string'},'strategy':{'type':'string'},'verified':{'type':'boolean'},'max_time':{'type':'integer'},'max_iterations':{'type':'integer'},'max_retries':{'type':'integer'},'max_tool_calls':{'type':'integer'},'resume':{'type':'boolean'}}}; cp_schema={'type':'object','properties':{'action':{'type':'string'},'candidates':{'type':'array'},'goal':{'type':'string'},'steps':{'type':'array'},'scope':{'type':'array'},'task_id':{'type':'string'},'node_id':{'type':'string'},'tool':{'type':'string'},'input_data':{},'output':{},'error':{'type':'string'},'paths':{'type':'array'},'label':{'type':'string'},'transaction_id':{'type':'string'},'limit':{'type':'integer'},'query':{'type':'string'},'name':{'type':'string'},'operation':{'type':'string'},'args':{'type':'object'},'candidates':{'type':'array'},'transaction_id':{'type':'string'},'confirm':{'type':'string'},'level':{'type':'string'},'result':{},'tests':{'type':'array'},'files':{'type':'array'},'commit':{'type':'string'}},'required':['action']}; return {'jsonrpc':'2.0','id':rid,'result':{'tools':[{'name':n,'description':'Airi Computer action','inputSchema':cp_schema if n=='computer_control_plane' else reasoning_schema if n.startswith('computer_reasoning_') else {'type':'object','properties':{}}} for n in names]}}
    if method=='tools/call':
        params=req.get('params',{}); name=params.get('name',''); args=params.get('arguments',{})
        mapping={
          'computer_status':('status',{}),'computer_observe':('observe',{}),'computer_screenshot':('screenshot',{}),
          'computer_find_text':('find-text',args),'computer_click_element':('action',{'action':'click_element','payload':args}),'computer_click':('action',{'action':'click','payload':args}),
          'computer_double_click':('action',{'action':'double_click','payload':args}),'computer_move':('action',{'action':'move','payload':args}), 'computer_drag':('action',{'action':'drag','payload':args}),
          'computer_scroll':('action',{'action':'scroll','payload':args}), 'computer_key':('action',{'action':'key','payload':args}), 'computer_hotkey':('action',{'action':'hotkey','payload':args}), 'computer_type':('action',{'action':'type','payload':args}),
          'computer_wait':('action',{'action':'wait','payload':args}), 'computer_windows':('windows',{}),'computer_mouse_position':('mouse_position',{}),'computer_browser_open':('action',{'action':'browser_open','payload':args}),
          'computer_browser_status':('action',{'action':'browser_status','payload':args}),'computer_browser_screenshot':('action',{'action':'browser_screenshot','payload':args}),'computer_browser_state':('action',{'action':'browser_state','payload':args}),'computer_browser_text':('action',{'action':'browser_text','payload':args}), 'computer_act_verify':('act-verify',args),'computer_cleanup_scan':('cleanup_scan',{}),'computer_cleanup_safe':('cleanup_safe',args),'computer_project_analyze':('code_analyze',args),'computer_project_tree':('code_tree',args),'computer_file_read':('code_read',args),'computer_file_search':('code_search',args),'computer_file_write':('code_write',args),'computer_file_patch':('code_patch',args),'computer_terminal_run':('code_shell',args),'computer_test_run':('code_test',args),'computer_build_run':('code_build',args),'computer_lint':('code_lint',args),'computer_git_status':('code_git_status',args),'computer_git_diff':('code_git_diff',args),'computer_git_log':('code_git_log',args),'computer_git_commit':('code_git_commit',args),'computer_skill_list':('skill_list',{}),'computer_skill_load':('skill_load',args),'computer_skill_create':('skill_create',args),'computer_skill_update':('skill_update',args),'computer_skill_test':('skill_test',args),'computer_skill_delete':('skill_delete',args),'computer_project_memory_read':('memory_read',{}),'computer_project_memory_update':('memory_update',args),'computer_code_apply_fix':('code_apply_fix',args),'computer_code_verify_change':('code_verify_change',args),'computer_code_agent':('code_agent',args),'computer_project_context':('code_project_context',args),'computer_task_start':('task_start',args),'computer_task_read':('task_read',{}),'computer_task_update':('task_update',args),'computer_task_finish':('task_finish',args),'computer_scope_check':('scope_check',args),'computer_diff_summary':('diff_summary',args),'computer_guardrails':('guardrails',args),'computer_snapshot':('snapshot',args),'computer_restore_snapshot':('restore_snapshot',args),'computer_prepare_commit':('prepare_commit',args),'computer_code_commit':('code_commit',args),'computer_autonomous_cycle':('autonomous_cycle',args),'computer_health':('health',{}),'computer_recovery_read':('recovery_read',{}),'computer_recovery_checkpoint':('recovery_checkpoint',args),'computer_recovery_finish':('recovery_finish',args),'computer_decision_record':('decision_record',args),'computer_decisions':('decisions',args),'computer_persistence_status':('persistence_status',{}),'computer_persist':('persist',args),'computer_research':('research',args),'computer_backup_prune':('backup_prune',args),'computer_scheduler_status':('scheduler_status',{}),'computer_scheduler_schedule':('scheduler_schedule',args),'computer_scheduler_cancel':('scheduler_cancel',args),'computer_browser_auth_set':('browser_auth_set',args),'computer_browser_auth_save':('browser_auth_save',args),'computer_browser_auth_status':('browser_auth_status',args),'computer_browser_human_status':('browser_human_status',args),'computer_browser_human_wait':('browser_human_wait',args),'computer_permission_check':('permission_check',{}),'computer_runtime_preflight':('runtime_preflight',{}),'computer_integrations':('integrations',{}),'computer_control_plane':('control_plane',args),'computer_autonomous_goal':('autonomous_goal',args),'computer_terminal_start':('job_start',args),'computer_terminal_status':('job_status',args),'computer_terminal_list':('job_list',args),'computer_terminal_attach':('job_attach',args),'computer_terminal_detach':('job_detach',args),'computer_terminal_cancel':('job_cancel',args),'computer_terminal_cleanup':('job_cleanup',args),'computer_task_add_node':('task_add_node',args),'computer_task_add_dependency':('task_add_dependency',args),'computer_task_skip':('task_skip',args),'computer_task_runnable':('task_runnable',args),'computer_context_pack':('context_pack',args),'computer_verify_deliverable':('verify_deliverable',args),'computer_experience_record':('experience_record',args),'computer_experience_match':('experience_match',args),'computer_model_choose':('model_choose',args),'computer_subagent_create':('subagent_create',args),'computer_subagent_status':('subagent_status',args),'computer_subagent_list':('subagent_list',args),'computer_subagent_finish':('subagent_finish',args),'computer_subagent_remove':('subagent_remove',args),'computer_reasoning_start':('reasoning_start',args),'computer_reasoning_status':('reasoning_status',args),'computer_reasoning_next_action':('reasoning_next_action',args),'computer_reasoning_observe':('reasoning_observe',args),'computer_reasoning_mark_step':('reasoning_mark_step',args),'computer_reasoning_replan':('reasoning_replan',args),'computer_reasoning_feedback':('reasoning_feedback',args),'computer_reasoning_finish':('reasoning_finish',args),'computer_reasoning_goal':('reasoning_goal',args)}
        if name not in mapping: return {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Tool not found'}}
        path, payload=mapping[name]
        try:
            if path=='autonomous_goal':
                result=_control_plane_bootstrap().autonomous_goal(payload['goal'],payload.get('steps'),payload.get('scope'),payload.get('max_time',900),payload.get('max_iterations',25),payload.get('max_retries',3),payload.get('max_tool_calls',100),payload.get('max_parallel_tasks',1),payload.get('resume',True))
            elif path=='control_plane':
                cp=_control_plane_bootstrap(); action=payload.get('action','status')
                if action=='status': result=cp.status()
                elif action=='bootstrap': result=cp.bootstrap(tools()['tools'])
                elif action=='route': result=cp.route(payload.get('candidates',[]))
                elif action=='task_start': result=cp.plan(payload['goal'],payload.get('steps',[]),payload.get('scope'))
                elif action=='task_read': result=cp.tasks.read(payload.get('task_id'))
                elif action=='task_update': result=cp.tasks.update(payload['node_id'],payload.get('status','completed'),payload.get('output'),payload.get('error'),payload.get('checkpoint'))
                elif action=='task_finish': result=cp.tasks.finish(payload.get('status','completed'))
                elif action=='transaction_begin': result=cp.transaction_begin(payload.get('paths',[]),payload.get('label','task'))
                elif action=='transaction_step': result=cp.transactions.step(payload['transaction_id'],payload.get('label','step'),payload.get('note',''))
                elif action=='transaction_commit': result=cp.transactions.commit(payload['transaction_id'])
                elif action=='transaction_rollback': result=cp.transactions.rollback(payload['transaction_id'])
                elif action=='execute': result=cp.execute(payload['task_id'],payload['node_id'],payload.get('candidates',[]),payload['operation'],payload.get('args'),payload.get('transaction_id'))
                elif action=='maintenance_recover': result=cp.maintenance.recover(payload.get('level','auto'),payload.get('confirm'))
                elif action=='audit': result=cp.audit.tail(payload.get('limit',100))
                elif action=='index_refresh': result=cp.index.refresh(payload.get('paths'))
                elif action=='index_search': result=cp.index.search(payload.get('query',''),payload.get('limit',50))
                elif action=='maintenance': result=cp.maintenance.run()
                elif action=='skills': result=cp.skills.refresh()
                elif action=='skill_verify': result=cp.skills.verify(payload['name'])
                elif action=='verify': result=cp.verify(payload.get('goal',''),payload.get('result'),payload.get('tests'),payload.get('files'),payload.get('commit'))
                else: raise ValueError(f'unknown control-plane action: {action}')
            elif path=='status': result=status()
            elif path=='observe': result=observe()
            elif path=='screenshot': result=perform('screenshot',{})
            elif path=='windows': result=windows()
            elif path=='mouse_position': result=mouse_position()
            elif path=='find-text': result=find_text(FindText(**payload))
            elif path=='cleanup_scan': result=cleanup_scan()
            elif path=='cleanup_safe': result=cleanup_safe(payload.get('max_bytes'))
            elif path=='code_analyze': result=code_analyze(payload.get('path','.'))
            elif path=='code_tree': result=code_analyze(payload.get('path','.'))['tree']
            elif path=='code_read': result=code_read(payload['path'])
            elif path=='code_search': result=code_search(payload['query'],payload.get('path','.'),payload.get('limit',100))
            elif path=='code_write': result=code_write(payload['path'],payload['content'],declared_scope=payload.get('scope'))
            elif path=='code_patch': result=code_patch(payload['path'],payload['old'],payload['new'],payload.get('replace_all',False),declared_scope=payload.get('scope'))
            elif path=='code_shell':
                code_scope_check([payload.get('cwd','.')], payload.get('scope') or [])
                result=code_shell(payload['command'],payload.get('cwd','.'),payload.get('timeout',120),payload.get('allow_shell',False))
            elif path=='code_test': result=code_test(payload.get('command','pytest -q'),payload.get('cwd','.'),payload.get('timeout',170))
            elif path=='code_build': result=code_build(payload.get('command','python -m compileall -q .'),payload.get('cwd','.'),payload.get('timeout',170))
            elif path=='code_lint': result=code_lint(payload.get('command','python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")'),payload.get('cwd','.'),payload.get('timeout',170))
            elif path=='code_git_status': result=code_git_status(payload.get('path','.'))
            elif path=='code_git_diff': result=code_git_diff(payload.get('path','.'))
            elif path=='code_git_log': result=code_git_log(payload.get('path','.'),payload.get('n',10))
            elif path=='code_git_commit': result=code_git_commit(payload['message'],payload.get('path','.'),declared_scope=payload.get('scope'),allow_test_changes=payload.get('allow_test_changes',False),allow_security_changes=payload.get('allow_security_changes',False))
            elif path=='skill_list': result=list_skills()
            elif path=='skill_load': result=load_skill(payload['name'])
            elif path=='skill_create': result=create_skill(payload['name'],payload['description'],payload['instructions'],payload.get('tools'))
            elif path=='skill_update': result=update_skill(payload['name'],payload['content'])
            elif path=='skill_test': result=test_skill(payload['name'])
            elif path=='skill_delete': result=delete_skill(payload['name'])
            elif path=='memory_read': result=memory_read()
            elif path=='memory_update': result=memory_update(payload['entry'])
            elif path=='code_apply_fix': result=code_apply_fix(payload['path'],payload['old'],payload['new'],payload.get('test_command',''))
            elif path=='code_verify_change': result=code_verify_change(payload['path'],payload['test_command'])
            elif path=='code_agent': result=code_agent(payload['goal'],payload.get('project_path','.'),payload.get('max_attempts',5),payload.get('steps'),payload.get('scope'),payload.get('changes'),payload.get('test_command',''))
            elif path=='code_project_context': result=code_project_context(payload.get('path','.'))
            elif path=='task_start': result=task_start(payload['goal'],payload['steps'],payload.get('scope'))
            elif path=='task_read': result=task_read()
            elif path=='task_update': result=task_update(payload['index'],payload['status'],payload.get('note',''))
            elif path=='task_finish': result=task_finish(payload.get('outcome','done'),payload.get('note',''))
            elif path=='scope_check': result=code_scope_check(payload['paths'],payload['scope'])
            elif path=='diff_summary': result=code_diff_summary(payload.get('path','.'),payload.get('scope'),payload.get('allow_test_changes',False),payload.get('allow_security_changes',False))
            elif path=='guardrails': result=code_guardrail_check(payload.get('path','.'),payload.get('scope'),payload.get('allow_test_changes',False),payload.get('allow_security_changes',False))
            elif path=='snapshot': result=code_snapshot(payload['paths'],payload.get('label','task'))
            elif path=='restore_snapshot': result=code_restore_snapshot(payload['snapshot'])
            elif path=='prepare_commit': result=code_prepare_commit(payload.get('path','.'),payload.get('scope'))
            elif path=='code_commit': result=code_atomic_commit(payload['message'],payload.get('project_path','.'),payload.get('scope'),payload.get('allow_test_changes',False),payload.get('allow_security_changes',False))
            elif path=='autonomous_cycle': result=code_autonomous_change_cycle(payload['changes'],payload.get('project_path','.'),payload.get('test_command',''),payload.get('scope'),payload.get('max_attempts',5))
            elif path=='job_start': result=_control_plane_bootstrap().job_start(payload['command'],payload.get('cwd','.'),payload.get('timeout',900),payload.get('owner_task'),payload.get('scope'),payload.get('allow_shell',False))
            elif path=='job_status': result=_control_plane_bootstrap().job_status(payload['job_id'])
            elif path=='job_list': result=_control_plane_bootstrap().job_list()
            elif path=='job_attach': result=_control_plane_bootstrap().job_attach(payload['job_id'],payload.get('tail',200))
            elif path=='job_detach': result=_control_plane_bootstrap().job_detach(payload['job_id'])
            elif path=='job_cancel': result=_control_plane_bootstrap().job_cancel(payload['job_id'],payload.get('grace',5))
            elif path=='job_cleanup': result=_control_plane_bootstrap().job_cleanup(payload.get('keep_final',100))
            elif path=='task_add_node': result=_control_plane_bootstrap().tasks.add_node(payload['task_id'],payload['node'],payload.get('created_by','dynamic'))
            elif path=='task_add_dependency': result=_control_plane_bootstrap().tasks.add_dependency(payload['task_id'],payload['node_id'],payload['depends_on'])
            elif path=='task_skip': result=_control_plane_bootstrap().tasks.skip(payload['task_id'],payload['node_id'],payload.get('reason','not needed'))
            elif path=='task_runnable': result={'runnable':_control_plane_bootstrap().tasks.runnable(payload.get('task_id'))}
            elif path=='context_pack': result=_control_plane_bootstrap().context_pack(payload['query'],payload.get('limit_files',12),payload.get('max_bytes',120000))
            elif path=='verify_deliverable': result=_control_plane_bootstrap().verify_deliverable(payload.get('requirements'),payload.get('tests'),payload.get('build_cmd'),payload.get('lint_cmd'),payload.get('project_path','.'),payload.get('runtime'),payload.get('security'))
            elif path=='experience_record': result=_control_plane_bootstrap().record_experience(payload['problem'],payload['context'],payload['solution'],payload.get('tools'),payload.get('failure_modes'),payload.get('successful_strategy',''),payload.get('verification'),payload.get('tags'))
            elif path=='experience_match': result=_control_plane_bootstrap().experience_match(payload['query'],payload.get('limit',5))
            elif path=='model_choose': result=_control_plane_bootstrap().choose_model(payload.get('task_type','simple'),payload.get('complexity','medium'),payload.get('needs_vision',False),payload.get('prefer_speed',False))
            elif path=='reasoning_start': result=_control_plane_bootstrap().reasoning_start(payload['goal'],payload.get('plan') or payload.get('steps'),payload.get('scope'),payload.get('metadata'))
            elif path=='reasoning_status': result=_control_plane_bootstrap().reasoning_status(payload.get('run_id'))
            elif path=='reasoning_next_action': result=_control_plane_bootstrap().reasoning_next_action(payload.get('run_id'))
            elif path=='reasoning_observe': result=_control_plane_bootstrap().reasoning_observe(payload.get('observation'),payload.get('run_id'),payload.get('evidence'),payload.get('phase'))
            elif path=='reasoning_mark_step': result=_control_plane_bootstrap().reasoning_mark_step(payload['step_id'],payload['status'],payload.get('run_id'),payload.get('result'),payload.get('error'),payload.get('evidence'),payload.get('metadata'))
            elif path=='reasoning_replan': result=_control_plane_bootstrap().reasoning_replan(payload.get('reason'),payload.get('run_id'),payload.get('strategy'))
            elif path=='reasoning_feedback': result=_control_plane_bootstrap().reasoning_feedback(payload['operation'],payload.get('success',False),payload.get('result'),payload.get('error'),payload.get('tool'),payload.get('task'),payload.get('step'),payload.get('evidence'),payload.get('metadata'),payload.get('run_id'))
            elif path=='reasoning_finish': result=_control_plane_bootstrap().reasoning_finish(payload.get('verified',False),payload.get('result'),payload.get('run_id'))
            elif path=='reasoning_goal': result=_control_plane_bootstrap().reasoning_goal(payload['goal'],payload.get('steps') or payload.get('plan'),payload.get('scope'),payload.get('metadata'),payload.get('max_time',900),payload.get('max_iterations',25),payload.get('max_retries',3),payload.get('max_tool_calls',100),payload.get('resume',True))
            elif path=='health': result=health_check()
            elif path=='recovery_read': result=recovery_read()
            elif path=='recovery_checkpoint': result=checkpoint(payload['goal'],payload.get('scope',[]),payload.get('step',0),payload.get('note',''),payload.get('artifacts'),payload.get('status','active'))
            elif path=='recovery_finish': result=recovery_finish(payload.get('status','done'),payload.get('note',''))
            elif path=='decision_record': result=record_decision(payload['decision'],payload.get('reason',''),payload.get('evidence'),payload.get('files'),payload.get('commit'),payload.get('result',''))
            elif path=='decisions': result=decisions(payload.get('limit',50))
            elif path=='persistence_status': result=persistence_status()
            elif path=='persist': result=persist_current(payload['message'],payload.get('branch'),payload.get('push',True),payload.get('scope'))
            elif path=='research': result=_research_with_browser(payload)
            elif path=='backup_prune':
                from cleanup import prune_backups
                result=prune_backups(payload.get('max_entries',50),payload.get('max_age_days',30),payload.get('dry_run',True))
            elif path=='scheduler_status': result=scheduler_status()
            elif path=='scheduler_schedule': result=schedule_job(payload['name'],payload['action'],payload['interval_seconds'],payload.get('run_now',False))
            elif path=='scheduler_cancel': result=cancel_job(payload['name'])
            elif path=='browser_auth_set': result=browser().auth_set(payload['profile'])
            elif path=='browser_auth_save': result=browser().auth_save(payload.get('profile'))
            elif path=='browser_auth_status': result=browser().auth_status(payload['profile'])
            elif path=='browser_human_status': result=browser().human_challenge_status()
            elif path=='browser_human_wait': result=browser().wait_for_human(payload.get('timeout',300),payload.get('poll',1.0))
            elif path=='permission_check': result=permission_check()
            elif path=='runtime_preflight': result=runtime_preflight()
            elif path=='integrations': result=integration_status()
            elif path=='action': result=action(ActionRequest(**payload))
            else: result=act_verify(ActionRequest(**payload))
            return {'jsonrpc':'2.0','id':rid,'result':{'content':[{'type':'text','text':str(result)}],'structuredContent':result}}
        except Exception as e:
            return {'jsonrpc':'2.0','id':rid,'error':{'code':-32000,'message':str(e)}}
    return {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Method not found'}}
