import base64, io, os, subprocess, time, difflib, threading, atexit, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
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
                    persistence_status, persist_current, research, scheduler_status, schedule_job, cancel_job)

# Optional remote MCP authentication. Local requests remain unchanged when the
# token is unset; public deployments should set AIRI_MCP_TOKEN.
AIRI_MCP_TOKEN = os.environ.get('AIRI_MCP_TOKEN', '').strip() or Path('/home/user/airi/.mcp_token').read_text().strip() if Path('/home/user/airi/.mcp_token').exists() else ''

@app.middleware('http')
async def mcp_auth_middleware(request, call_next):
    if AIRI_MCP_TOKEN and request.url.path == '/mcp':
        auth = request.headers.get('authorization', '')
        if auth != f'Bearer {AIRI_MCP_TOKEN}':
            return JSONResponse(status_code=401, content={'error': 'unauthorized'})
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
        self._page = None
        self._lock = threading.Lock()
        self.last_error = None

    def _run(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs).result()

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
        modes = [False, True] if gui_available() else [True]
        for headless in modes:
            for _attempt in range(3):
                try:
                    self._browser = self._pw.chromium.launch(headless=headless, args=launch_args, timeout=15000)
                    self._page = self._browser.new_page(viewport={'width':1280,'height':800})
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
            if self._page is not None:
                self._page.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        self._page = None
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
            try:
                page=self._ensure(); page.goto(url, wait_until=wait_until, timeout=30000); return {'ok':True,'url':page.url,'title':page.title()}
            except Exception as exc:
                self._close_worker(); return {'ok':False,'error':str(exc),'url':None}
        return self.call(_open)

    def screenshot(self):
        def _screenshot():
            try:
                page=self._ensure(); b=page.screenshot(type='png', timeout=10000); return {'ok':True,'format':'png','width':1280,'height':800,'data_base64':base64.b64encode(b).decode()}
            except Exception as exc:
                self._close_worker(); return {'ok':False,'error':str(exc)}
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
    return ImageGrab.grab()

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
    pa = _pyautogui()
    if action == 'click_element':
        query=p['text']; img=screenshot_image(); matches=[x for x in ocr(img) if query.casefold() in x['text'].casefold()]
        if not matches: raise RuntimeError(f'Element text not found: {query}')
        m=matches[0]; _pyautogui().click(m['x']+m['w']//2,m['y']+m['h']//2); return {'ok':True,'match':m}
    if action == 'screenshot':
        img=screenshot_image(); return {'format':'png','width':img.width,'height':img.height,'data_base64':image_b64(img)}
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
            img=screenshot_image(); return {'ok':True,'display':DISPLAY,'gui_available':True,'resolution':f'{img.width}x{img.height}','version':'2.0'}
        except Exception: pass
    return {'ok':True,'display':DISPLAY if gui_available() else None,'gui_available':False,'resolution':None,'version':'2.0','mode':'headless'}

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
        seen.add(url); discovered.append(url)
        if len(discovered)>=max_sources: break
    return research(topic, discovered, max_sources)

@app.post('/research')
def research_endpoint(req: Dict[str,Any]): return _research_with_browser(req)
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

@app.get('/tools')
def tools():
    names=['computer_status','computer_observe','computer_screenshot','computer_find_text','computer_click_element','computer_click','computer_double_click','computer_move','computer_drag','computer_scroll','computer_key','computer_hotkey','computer_type','computer_wait','computer_windows','computer_mouse_position','computer_browser_open','computer_browser_status','computer_browser_screenshot','computer_browser_state','computer_browser_text','computer_act_verify','computer_cleanup_scan','computer_cleanup_safe','computer_project_analyze','computer_project_tree','computer_file_read','computer_file_search','computer_file_write','computer_file_patch','computer_terminal_run','computer_test_run','computer_build_run','computer_lint','computer_git_status','computer_git_diff','computer_git_log','computer_git_commit','computer_skill_list','computer_skill_load','computer_skill_create','computer_skill_update','computer_skill_test','computer_skill_delete','computer_project_memory_read','computer_project_memory_update','computer_code_apply_fix','computer_code_verify_change','computer_code_agent','computer_project_context','computer_task_start','computer_task_read','computer_task_update','computer_task_finish','computer_scope_check','computer_diff_summary','computer_guardrails','computer_snapshot','computer_restore_snapshot','computer_prepare_commit','computer_code_commit','computer_autonomous_cycle','computer_health','computer_recovery_read','computer_recovery_checkpoint','computer_recovery_finish','computer_decision_record','computer_decisions','computer_persistence_status','computer_persist','computer_research','computer_backup_prune','computer_scheduler_status','computer_scheduler_schedule','computer_scheduler_cancel']
    return {'name':'Airi Computer','version':'2.0','tools':names}

@app.post('/mcp')
def mcp(req: Dict[str,Any]):
    method=req.get('method'); rid=req.get('id')
    if method=='initialize': return {'jsonrpc':'2.0','id':rid,'result':{'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{'name':'Airi Computer','version':'2.0'}}}
    if method in ('notifications/initialized','ping'): return {'jsonrpc':'2.0','id':rid,'result':{}}
    if method=='tools/list':
        names=tools()['tools']; return {'jsonrpc':'2.0','id':rid,'result':{'tools':[{'name':n,'description':'Airi Computer action'} for n in names]}}
    if method=='tools/call':
        params=req.get('params',{}); name=params.get('name',''); args=params.get('arguments',{})
        mapping={
          'computer_status':('status',{}),'computer_observe':('observe',{}),'computer_screenshot':('screenshot',{}),
          'computer_find_text':('find-text',args),'computer_click_element':('action',{'action':'click_element','payload':args}),'computer_click':('action',{'action':'click','payload':args}),
          'computer_double_click':('action',{'action':'double_click','payload':args}),'computer_move':('action',{'action':'move','payload':args}), 'computer_drag':('action',{'action':'drag','payload':args}),
          'computer_scroll':('action',{'action':'scroll','payload':args}), 'computer_key':('action',{'action':'key','payload':args}), 'computer_hotkey':('action',{'action':'hotkey','payload':args}), 'computer_type':('action',{'action':'type','payload':args}),
          'computer_wait':('action',{'action':'wait','payload':args}), 'computer_windows':('windows',{}),'computer_mouse_position':('mouse_position',{}),'computer_browser_open':('action',{'action':'browser_open','payload':args}),
          'computer_browser_status':('action',{'action':'browser_status','payload':args}),'computer_browser_screenshot':('action',{'action':'browser_screenshot','payload':args}),'computer_browser_state':('action',{'action':'browser_state','payload':args}),'computer_browser_text':('action',{'action':'browser_text','payload':args}), 'computer_act_verify':('act-verify',args),'computer_cleanup_scan':('cleanup_scan',{}),'computer_cleanup_safe':('cleanup_safe',args),'computer_project_analyze':('code_analyze',args),'computer_project_tree':('code_tree',args),'computer_file_read':('code_read',args),'computer_file_search':('code_search',args),'computer_file_write':('code_write',args),'computer_file_patch':('code_patch',args),'computer_terminal_run':('code_shell',args),'computer_test_run':('code_test',args),'computer_build_run':('code_build',args),'computer_lint':('code_lint',args),'computer_git_status':('code_git_status',args),'computer_git_diff':('code_git_diff',args),'computer_git_log':('code_git_log',args),'computer_git_commit':('code_git_commit',args),'computer_skill_list':('skill_list',{}),'computer_skill_load':('skill_load',args),'computer_skill_create':('skill_create',args),'computer_skill_update':('skill_update',args),'computer_skill_test':('skill_test',args),'computer_skill_delete':('skill_delete',args),'computer_project_memory_read':('memory_read',{}),'computer_project_memory_update':('memory_update',args),'computer_code_apply_fix':('code_apply_fix',args),'computer_code_verify_change':('code_verify_change',args),'computer_code_agent':('code_agent',args),'computer_project_context':('code_project_context',args),'computer_task_start':('task_start',args),'computer_task_read':('task_read',{}),'computer_task_update':('task_update',args),'computer_task_finish':('task_finish',args),'computer_scope_check':('scope_check',args),'computer_diff_summary':('diff_summary',args),'computer_guardrails':('guardrails',args),'computer_snapshot':('snapshot',args),'computer_restore_snapshot':('restore_snapshot',args),'computer_prepare_commit':('prepare_commit',args),'computer_code_commit':('code_commit',args),'computer_autonomous_cycle':('autonomous_cycle',args),'computer_health':('health',{}),'computer_recovery_read':('recovery_read',{}),'computer_recovery_checkpoint':('recovery_checkpoint',args),'computer_recovery_finish':('recovery_finish',args),'computer_decision_record':('decision_record',args),'computer_decisions':('decisions',args),'computer_persistence_status':('persistence_status',{}),'computer_persist':('persist',args),'computer_research':('research',args),'computer_backup_prune':('backup_prune',args),'computer_scheduler_status':('scheduler_status',{}),'computer_scheduler_schedule':('scheduler_schedule',args),'computer_scheduler_cancel':('scheduler_cancel',args)}
        if name not in mapping: return {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Tool not found'}}
        path, payload=mapping[name]
        try:
            if path=='status': result=status()
            elif path=='observe': result=observe()
            elif path=='screenshot': result=screenshot()
            elif path=='windows': result=windows()
            elif path=='mouse_position': result=mouse_position()
            elif path=='find-text': result=find_text(FindText(**payload))
            elif path=='cleanup_scan': result=cleanup_scan()
            elif path=='cleanup_safe': result=cleanup_safe(payload.get('max_bytes'))
            elif path=='code_analyze': result=code_analyze(payload.get('path','.'))
            elif path=='code_tree': result=code_analyze(payload.get('path','.'))['tree']
            elif path=='code_read': result=code_read(payload['path'])
            elif path=='code_search': result=code_search(payload['query'],payload.get('path','.'),payload.get('limit',100))
            elif path=='code_write': result=code_write(payload['path'],payload['content'])
            elif path=='code_patch': result=code_patch(payload['path'],payload['old'],payload['new'],payload.get('replace_all',False))
            elif path=='code_shell': result=code_shell(payload['command'],payload.get('cwd','.'),payload.get('timeout',120),payload.get('allow_shell',False))
            elif path=='code_test': result=code_test(payload.get('command','pytest -q'),payload.get('cwd','.'),payload.get('timeout',170))
            elif path=='code_build': result=code_build(payload.get('command','python -m compileall -q .'),payload.get('cwd','.'),payload.get('timeout',170))
            elif path=='code_lint': result=code_lint(payload.get('command','python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")'),payload.get('cwd','.'),payload.get('timeout',170))
            elif path=='code_git_status': result=code_git_status(payload.get('path','.'))
            elif path=='code_git_diff': result=code_git_diff(payload.get('path','.'))
            elif path=='code_git_log': result=code_git_log(payload.get('path','.'),payload.get('n',10))
            elif path=='code_git_commit': result=code_git_commit(payload['message'],payload.get('path','.'))
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
            elif path=='health': result=health_check()
            elif path=='recovery_read': result=recovery_read()
            elif path=='recovery_checkpoint': result=checkpoint(payload['goal'],payload.get('scope',[]),payload.get('step',0),payload.get('note',''),payload.get('artifacts'),payload.get('status','active'))
            elif path=='recovery_finish': result=recovery_finish(payload.get('status','done'),payload.get('note',''))
            elif path=='decision_record': result=record_decision(payload['decision'],payload.get('reason',''),payload.get('evidence'),payload.get('files'),payload.get('commit'),payload.get('result',''))
            elif path=='decisions': result=decisions(payload.get('limit',50))
            elif path=='persistence_status': result=persistence_status()
            elif path=='persist': result=persist_current(payload['message'],payload.get('branch'),payload.get('push',True))
            elif path=='research': result=_research_with_browser(payload)
            elif path=='backup_prune':
                from cleanup import prune_backups
                result=prune_backups(payload.get('max_entries',50),payload.get('max_age_days',30),payload.get('dry_run',True))
            elif path=='scheduler_status': result=scheduler_status()
            elif path=='scheduler_schedule': result=schedule_job(payload['name'],payload['action'],payload['interval_seconds'],payload.get('run_now',False))
            elif path=='scheduler_cancel': result=cancel_job(payload['name'])
            elif path=='action': result=action(ActionRequest(**payload))
            else: result=act_verify(ActionRequest(**payload))
            return {'jsonrpc':'2.0','id':rid,'result':{'content':[{'type':'text','text':str(result)}],'structuredContent':result}}
        except Exception as e:
            return {'jsonrpc':'2.0','id':rid,'error':{'code':-32000,'message':str(e)}}
    return {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Method not found'}}
