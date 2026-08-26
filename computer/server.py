import base64, io, os, subprocess, time, difflib
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

def browser():
    global _BROWSER, _PAGE
    from playwright.sync_api import sync_playwright
    if _PAGE is not None:
        try:
            _ = _PAGE.url
            return _PAGE
        except Exception:
            _PAGE = None
    if not hasattr(browser, '_pw'):
        browser._pw = sync_playwright().start()
    launch_args = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    last_error = None
    for headless in (not gui_available(), True):
        try:
            _BROWSER = browser._pw.chromium.launch(headless=headless, args=launch_args)
            _PAGE = _BROWSER.new_page(viewport={'width':1280,'height':800})
            return _PAGE
        except Exception as exc:
            last_error = exc
            _BROWSER = None
            _PAGE = None
    raise RuntimeError(f'Browser unavailable: {last_error}')

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
    if action == 'browser_status': return {'available': True, 'open': _PAGE is not None, 'url': _PAGE.url if _PAGE else None}
    if action == 'browser_state':
        page=browser(); return {'url':page.url,'title':page.title(),'status':page.locator('#status').inner_text() if page.locator('#status').count() else None,'field':page.locator('#field').input_value() if page.locator('#field').count() else None,'scroll_y':page.evaluate('window.scrollY'),'drag_box':page.locator('#drag').bounding_box() if page.locator('#drag').count() else None,'click_box':page.locator('#click').bounding_box() if page.locator('#click').count() else None}
    if action == 'browser_open':
        page=browser(); page.goto(p['url'], wait_until=p.get('wait_until','domcontentloaded')); return {'ok':True,'url':page.url,'title':page.title()}
    if action == 'browser_screenshot':
        page=browser(); b=page.screenshot(type='png'); return {'format':'png','width':1280,'height':800,'data_base64':base64.b64encode(b).decode()}
    raise ValueError(f'Unsupported action: {action}')

def observe() -> Dict[str, Any]:
    img = screenshot_image()
    return {'ok':True,'display':DISPLAY,'resolution':f'{img.width}x{img.height}',
            'screenshot': image_b64(img),'ocr':ocr(img),'windows':windows_info(),
            'browser': {'open': _PAGE is not None, 'url': _PAGE.url if _PAGE else None}}

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
    before=screenshot_image(); result=perform(req.action, req.payload); time.sleep(0.15); after=screenshot_image(); diff=ImageChops.difference(before,after).convert('L'); bbox=diff.getbbox(); changed=sum(1 for v in diff.getdata() if v>10); total=diff.width*diff.height
    return {'ok':True,'action':req.action,'result':result,'changed_ratio':changed/total,'changed_bbox':bbox}

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

@app.get('/tools')
def tools():
    names=['computer_status','computer_observe','computer_screenshot','computer_find_text','computer_click_element','computer_click','computer_double_click','computer_move','computer_drag','computer_scroll','computer_key','computer_hotkey','computer_type','computer_wait','computer_windows','computer_mouse_position','computer_browser_open','computer_browser_status','computer_browser_screenshot','computer_browser_state','computer_act_verify','computer_cleanup_scan','computer_cleanup_safe']
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
          'computer_browser_status':('action',{'action':'browser_status','payload':args}),'computer_browser_screenshot':('action',{'action':'browser_screenshot','payload':args}),'computer_browser_state':('action',{'action':'browser_state','payload':args}), 'computer_act_verify':('act-verify',args),'computer_cleanup_scan':('cleanup_scan',{}),'computer_cleanup_safe':('cleanup_safe',args)}
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
            elif path=='action': result=action(ActionRequest(**payload))
            else: result=act_verify(ActionRequest(**payload))
            return {'jsonrpc':'2.0','id':rid,'result':{'content':[{'type':'text','text':str(result)}],'structuredContent':result}}
        except Exception as e:
            return {'jsonrpc':'2.0','id':rid,'error':{'code':-32000,'message':str(e)}}
    return {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Method not found'}}
