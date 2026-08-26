#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any
ROOT=Path(os.environ.get('AIRIPC_WORKSPACE_ROOT','/home/user/airi')).expanduser()
STATUS_URL=os.environ.get('AIRIPC_STATUS_URL','http://127.0.0.1:9010/status')
MCP_URL=os.environ.get('AIRIPC_MCP_URL','http://127.0.0.1:9010/mcp')
START=ROOT/'computer'/'start.sh'
GUIDE=ROOT/'AIRI_COMPUTER_FUTURE_AGENT.md'
CONTROL=ROOT/'scripts'/'airi-control'
def req(url,method='GET',body=None,timeout=8):
    data=None; headers={'Accept':'application/json'}
    if body is not None:
        data=json.dumps(body).encode(); headers['Content-Type']='application/json'
    try:
        with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers,method=method),timeout=timeout) as r:
            raw=r.read().decode('utf-8','replace')
            try:return True,json.loads(raw)
            except json.JSONDecodeError:return True,raw
    except (urllib.error.URLError,TimeoutError,OSError) as e:return False,str(e)
def start():
    if not START.exists(): return False,f'missing {START}'
    env=os.environ.copy(); env.setdefault('DISPLAY',':99')
    try:
        p=subprocess.run(['/bin/sh',str(START)],cwd=str(ROOT/'computer'),env=env,text=True,capture_output=True,timeout=120,check=False)
        return p.returncode==0,(p.stdout+'\n'+p.stderr).strip()[-3000:]
    except Exception as e:return False,str(e)
def mcp():
    init={'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'airi-agent','version':'1.1'}}}
    ok,v=req(MCP_URL,'POST',init)
    if not ok:return {'reachable':False,'error':v}
    ok,v=req(MCP_URL,'POST',{'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})
    if not ok:return {'reachable':True,'initialized':True,'tools_ok':False,'error':v}
    result=v.get('result',{}) if isinstance(v,dict) else {}
    tools=result.get('tools',[]) if isinstance(result,dict) else []
    names=[x.get('name') for x in tools if isinstance(x,dict)]
    return {'reachable':True,'initialized':True,'tools_ok':True,'tool_count':len(names),'tools':names,'has_computer_status':'computer_status' in names}

def control(cmd, timeout=45):
    try:
        p=subprocess.run(['/bin/sh',str(CONTROL),*cmd],cwd=str(ROOT),env={**os.environ,'DISPLAY':os.environ.get('DISPLAY',':99')},text=True,capture_output=True,timeout=timeout,check=False)
        return p.returncode, (p.stdout+'\n'+p.stderr).strip()[-5000:]
    except Exception as e:
        return 99, str(e)


def ensure_browser():
    cache=Path.home()/'.cache'/'ms-playwright'
    candidates=list(cache.glob('chromium-*/chrome-linux*/chrome'))+list(cache.glob('chromium-*/chrome'))+list(cache.glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
    if any(x.exists() for x in candidates): return True, 'browser_present'
    try:
        p=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','playwright','install','chromium'],cwd=str(ROOT),env=os.environ.copy(),text=True,capture_output=True,timeout=170,check=False)
        return p.returncode==0, (p.stdout+'\n'+p.stderr).strip()[-2000:]
    except subprocess.TimeoutExpired:
        return False, 'playwright_install_timeout'
    except Exception as e:
        return False, str(e)

def functional_self_test():
    import base64 as _base64
    out={k:False for k in ['status','observe','screenshot','windows','mouse_move','mouse_click','typing','hotkey','keyboard','drag','scroll','browser','browser_navigation','browser_screenshot','browser_preflight']}
    code,text=control(['status']); out['status']=code==0
    ok,obs=req(STATUS_URL.replace('/status','/observe')); out['observe']=ok and isinstance(obs,dict) and obs.get('ok') is True
    ok,shot=req(STATUS_URL.replace('/status','/screenshot')); out['screenshot']=ok and isinstance(shot,dict) and shot.get('width')==1280 and shot.get('height')==800 and len(shot.get('data_base64',''))>100
    ok,wins=req(STATUS_URL.replace('/status','/windows')); out['windows']=ok and isinstance(wins,dict) and 'windows' in wins
    code,text=control(['move','320','240']); okp,pos=req(STATUS_URL.replace('/status','/mouse/position')); out['mouse_move']=code==0 and okp and pos.get('x')==320 and pos.get('y')==240
    browser_ok,browser_info=ensure_browser(); out['browser_preflight']=browser_ok
    if not browser_ok:
        out['browser_navigation']=False; out['browser']=False; out['browser_screenshot']=False; out['browser_preflight']=False
        out['all']=False; return out
    code,text=control(['browser-open','http://127.0.0.1:9010/self-test']); out['browser_navigation']=code==0 and 'Airi-PC Self Test' in text
    code,text=control(['browser-status']); out['browser']=code==0 and '"open":true' in text.replace(' ','').lower()
    ok,bshot=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_screenshot','payload':{}}); out['browser_screenshot']=ok and isinstance(bshot,dict) and bshot.get('format')=='png' and len(bshot.get('data_base64',''))>100
    code,text=control(['click-element','AIRI_CLICK_TARGET']); ok,state=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}}); out['mouse_click']=code==0 and ok and state.get('status')=='AIRI_CLICK_OK'
    code,text=control(['click-element','AIRI_INPUT_FIELD']); code2,text2=control(['type','AIRI_TYPE_OK']); ok,state=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}}); out['typing']=code==0 and code2==0 and ok and state.get('field')=='AIRI_TYPE_OK' and state.get('status')=='AIRI_TYPE_OK'
    control(['click-element','AIRI_INPUT_FIELD']); code,text=control(['key','enter']); time.sleep(0.3); ok,state=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}}); out['keyboard']=code==0 and ok and state.get('status')=='AIRI_ENTER_OK'
    code,text=control(['hotkey','ctrl','shift','a']); ok,state=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}}); out['hotkey']=code==0 and ok and state.get('status')=='AIRI_HOTKEY_OK'
    # reset top and locate the visible drag target, then verify the page marker.
    control(['key','home']); time.sleep(0.2)
    ok,state=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}})
    try:
        import math
        box=state.get('drag_box'); click_box=state.get('click_box'); ok_obs,obs=req(STATUS_URL.replace('/status','/observe'))
        ocr_click=next(i for i in obs.get('ocr',[]) if 'AIRI_CLICK_TARGET' in i.get('text',''))
        offx=ocr_click['x']-(click_box['x']+click_box['width']/2); offy=ocr_click['y']-(click_box['y']+click_box['height']/2)
        x=int(box['x']+box['width']/2+offx); y=int(box['y']+box['height']/2+offy); code,text=control(['drag',str(x),str(y),str(x+180),str(y),'0.8']); time.sleep(0.3); ok2,state2=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}}); out['drag']=code==0 and ok2 and state2.get('status')=='AIRI_DRAG_OK'
    except Exception: out['drag']=False
    control(['key','home']); time.sleep(0.2); control(['scroll','-5']); ok,state=req(STATUS_URL.replace('/status','/action'),'POST',{'action':'browser_state','payload':{}}); out['scroll']=ok and (state.get('status')=='AIRI_SCROLL_OK' or state.get('scroll_y',0)>50)
    out['all']=all(out.values())
    return out

def main():
    report={'agent':'airi-agent','workspace':str(ROOT),'workspace_exists':ROOT.exists(),'guide_exists':GUIDE.exists(),'control_exists':CONTROL.exists(),'display':os.environ.get('DISPLAY',':99')}
    if not ROOT.exists(): report.update(ready=False,stage='workspace_missing'); print(json.dumps(report,indent=2)); return 2
    initial_ok,initial=req(STATUS_URL)
    if initial_ok and isinstance(initial,dict) and initial.get('ok') is True:
        ok,out=True,'already_running'
    else: ok,out=start()
    report['computer_start']={'attempted':not (initial_ok and isinstance(initial,dict) and initial.get('ok') is True),'ok':ok,'output':out}
    for _ in range(15):
        ok_s,st=req(STATUS_URL)
        if ok_s: report['status']=st; break
        time.sleep(1)
    else: report['status']={'ok':False,'error':'status endpoint unavailable'}
    report['mcp']=mcp()
    selftest=ROOT/'scripts'/'airi-selftest'
    if selftest.exists() and report.get('mcp',{}).get('reachable') and report.get('mcp',{}).get('tools_ok'):
        try:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),str(selftest)],cwd=str(ROOT),env={**os.environ,'DISPLAY':os.environ.get('DISPLAY',':99')},text=True,capture_output=True,timeout=170,check=False)
            report['functional']={'returncode':p.returncode,'output':(p.stdout+'\n'+p.stderr).strip()[-12000:],'passed':p.returncode==0 and 'ALL= True' in p.stdout}
        except subprocess.TimeoutExpired:
            report['functional']={'returncode':124,'output':'selftest_timeout','passed':False}
    else:
        report['functional']={'returncode':2,'output':'selftest_unavailable','passed':False}
    status_ok=isinstance(report.get('status'),dict) and report['status'].get('ok') is True
    mcp_ok=isinstance(report.get('mcp'),dict) and report['mcp'].get('reachable') is True and report['mcp'].get('tools_ok') is True
    functional_ok=isinstance(report.get('functional'),dict) and report['functional'].get('passed') is True
    report['ready']=bool(status_ok and mcp_ok and functional_ok)
    report['stage']='ready' if report['ready'] else 'verification_failed'
    print(json.dumps(report,indent=2,ensure_ascii=False)); return 0 if report['ready'] else 1
if __name__=='__main__': raise SystemExit(main())
