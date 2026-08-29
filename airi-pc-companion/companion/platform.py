from __future__ import annotations
import os, platform, subprocess, sys
from pathlib import Path

class PlatformControl:
    def __init__(self, sandbox: Path): self.sandbox=sandbox.resolve(); self.sandbox.mkdir(parents=True,exist_ok=True)
    def info(self): return {'os':platform.system(),'release':platform.release(),'arch':platform.machine(),'python':platform.python_version()}
    def screenshot(self):
        try:
            from PIL import ImageGrab
            img=ImageGrab.grab(); return img
        except Exception:
            try:
                import pyautogui
                return pyautogui.screenshot()
            except Exception as e: raise RuntimeError(f'screenshot_unavailable:{type(e).__name__}')
    def mouse(self, action, **kw):
        import pyautogui
        if action=='move': pyautogui.moveTo(int(kw['x']),int(kw['y']),duration=min(float(kw.get('duration',0.15)),2.0)); return {'ok':True}
        if action=='click': pyautogui.click(int(kw['x']),int(kw['y']),clicks=2 if kw.get('double') else 1,button=kw.get('button','left')); return {'ok':True}
        if action=='drag': pyautogui.moveTo(int(kw['x']),int(kw['y'])); pyautogui.dragTo(int(kw['end_x']),int(kw['end_y']),duration=min(float(kw.get('duration',0.5)),3.0),button=kw.get('button','left')); return {'ok':True}
        if action=='scroll': pyautogui.scroll(int(kw.get('amount',1))*(1 if kw.get('direction','up')=='up' else -1)); return {'ok':True}
        raise ValueError('unsupported mouse action')
    def keyboard(self, text=None, key=None):
        import pyautogui
        if text is not None: pyautogui.write(str(text),interval=0.01); return {'ok':True}
        if key: pyautogui.hotkey(*key.split('+')) if '+' in key else pyautogui.press(key); return {'ok':True}
        raise ValueError('text or key required')
    def active_window(self):
        system=platform.system()
        if system=='Windows':
            import ctypes
            hwnd=ctypes.windll.user32.GetForegroundWindow(); buf=ctypes.create_unicode_buffer(512); ctypes.windll.user32.GetWindowTextW(hwnd,buf,512)
            return {'handle':int(hwnd),'title':buf.value}
        if system=='Darwin':
            p=subprocess.run(['osascript','-e','tell application "System Events" to get name of first process whose frontmost is true'],capture_output=True,text=True,timeout=5); return {'title':p.stdout.strip()}
        for cmd in (['xdotool','getactivewindow','getwindowname'],):
            try:
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=5,check=True); return {'title':p.stdout.strip()}
            except Exception: pass
        return {}
    def windows(self):
        system=platform.system()
        if system=='Windows':
            import ctypes
            out=[]
            EnumWindows=ctypes.windll.user32.EnumWindows; IsVisible=ctypes.windll.user32.IsWindowVisible; GetWindowText=ctypes.windll.user32.GetWindowTextW
            EnumProc=ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
            def cb(hwnd,_):
                if IsVisible(hwnd):
                    buf=ctypes.create_unicode_buffer(512); GetWindowText(hwnd,buf,512)
                    if buf.value: out.append({'handle':int(hwnd),'title':buf.value})
                return True
            EnumWindows(EnumProc(cb),0); return out
        if system=='Darwin':
            p=subprocess.run(['osascript','-e','tell application "System Events" to get name of every process whose visible is true'],capture_output=True,text=True,timeout=5); return [{'title':x.strip()} for x in p.stdout.split(',') if x.strip()]
        if sys.platform.startswith('linux'):
            for cmd in (['wmctrl','-l'],['xdotool','search','--name','']):
                try:
                    p=subprocess.run(cmd,capture_output=True,text=True,timeout=5,check=True); return [{'title':line.strip()} for line in p.stdout.splitlines() if line.strip()]
                except Exception: continue
        return []
    def launch(self, argv):
        if not isinstance(argv,list) or not argv or any(not isinstance(x,str) for x in argv): raise ValueError('argv list required')
        return {'pid':subprocess.Popen(argv,cwd=self.sandbox,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=False).pid}
    def filesystem_write(self, path, content):
        p=self._safe(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content,encoding='utf-8'); return {'path':str(p.relative_to(self.sandbox))}
    def filesystem_read(self, path):
        p=self._safe(path); return {'path':str(p.relative_to(self.sandbox)),'content':p.read_text(encoding='utf-8')}
    def filesystem_delete_test_file(self, path):
        p=self._safe(path)
        if not str(p.relative_to(self.sandbox)).startswith('test-'): raise PermissionError('only test-* files may be deleted')
        p.unlink(missing_ok=True); return {'deleted':True}
    def processes(self):
        try:
            import psutil
            return [{'pid':p.pid,'name':p.info.get('name'),'status':p.info.get('status')} for p in psutil.process_iter(['name','status'])]
        except Exception:
            cmd=['tasklist'] if platform.system()=='Windows' else ['ps','-eo','pid=,comm=']
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
            return [{'raw':x.strip()} for x in p.stdout.splitlines() if x.strip()]
    def _safe(self, path):
        p=(self.sandbox/str(path)).resolve()
        if self.sandbox not in p.parents and p!=self.sandbox: raise PermissionError('path_outside_sandbox')
        return p
