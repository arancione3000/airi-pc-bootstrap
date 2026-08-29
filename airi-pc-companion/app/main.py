from __future__ import annotations
import json, os, subprocess, sys, threading, urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

APP_DIR = Path(os.environ.get('AIRIPC_COMPANION_STATE', Path.home()/'.airi-pc-companion'))
CFG = APP_DIR / 'app.json'
DEFAULTS = {'version':'0.2.0','airi_url':''}
SERVICE_HOST='127.0.0.1'; SERVICE_PORT=17894

def load_cfg():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG.exists(): CFG.write_text(json.dumps(DEFAULTS), encoding='utf-8')
    try: return {**DEFAULTS, **json.loads(CFG.read_text(encoding='utf-8'))}
    except Exception: return DEFAULTS.copy()

def save_cfg(d):
    APP_DIR.mkdir(parents=True, exist_ok=True); CFG.write_text(json.dumps({k:v for k,v in d.items() if k!='token'}, indent=2), encoding='utf-8')
    try: os.chmod(CFG,0o600)
    except OSError: pass

def keyring_get():
    try:
        import keyring
        return keyring.get_password('Airi-PC Companion','pairing-token')
    except Exception: return None

def keyring_set(token):
    import keyring
    keyring.set_password('Airi-PC Companion','pairing-token',token)

def api_get(base,path,token=''):
    if not base: raise RuntimeError('Airi endpoint not configured')
    headers={'Authorization':f'Bearer {token}'} if token else {}
    req=urllib.request.Request(base.rstrip('/')+path,headers=headers)
    with urllib.request.urlopen(req,timeout=5) as r: return json.loads(r.read().decode())

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Airi-PC Companion'); self.geometry('540x470'); self.resizable(False,False)
        self.cfg=load_cfg(); self.server=None; self.connected=False
        self.status_var=tk.StringVar(value='OFFLINE'); self.auth_var=tk.StringVar(value='NOT CONFIGURED'); self.pc_var=tk.StringVar(value='LOCAL'); self.last_var=tk.StringVar(value='—')
        self._build(); self._tray(); self.after(500,self.refresh)
    def _build(self):
        ttk.Label(self,text='AIRI-PC COMPANION',font=('Segoe UI',18,'bold')).pack(pady=(18,3))
        ttk.Label(self,text='Secure desktop companion',font=('Segoe UI',10)).pack()
        box=ttk.LabelFrame(self,text='Connection',padding=14); box.pack(fill='x',padx=20,pady=15)
        for label,var in (('Connection',self.status_var),('Authentication',self.auth_var),('PC',self.pc_var),('Last connection',self.last_var)):
            row=ttk.Frame(box); row.pack(fill='x',pady=3); ttk.Label(row,text=label,width=18).pack(side='left'); ttk.Label(row,textvariable=var).pack(side='left')
        btns=ttk.Frame(self); btns.pack(pady=4)
        ttk.Button(btns,text='CONNETTI AIRI',command=self.connect).grid(row=0,column=0,padx=4,pady=4)
        ttk.Button(btns,text='DISCONNETTI',command=self.disconnect).grid(row=0,column=1,padx=4,pady=4)
        ttk.Button(btns,text='GAME MODE',command=self.game_mode).grid(row=1,column=0,padx=4,pady=4)
        ttk.Button(btns,text='STOP ALL',command=self.stop_all).grid(row=1,column=1,padx=4,pady=4)
        ttk.Button(btns,text='DIAGNOSTICA',command=self.diagnostics).grid(row=2,column=0,columnspan=2,padx=4,pady=4)
        ttk.Label(self,text=f"Version {self.cfg['version']}").pack(pady=12)
        ttk.Label(self,text='Pairing secret stored only in the OS credential store.',font=('Segoe UI',8)).pack()
    def _tray(self):
        try:
            import pystray
            from PIL import Image,ImageDraw
            im=Image.new('RGB',(64,64),'black'); d=ImageDraw.Draw(im); d.rectangle((12,12,52,52),outline='white',width=4)
            menu=pystray.Menu(pystray.MenuItem('Open',lambda *_: self.deiconify()),pystray.MenuItem('STOP ALL',lambda *_: self.stop_all()),pystray.MenuItem('Quit',lambda *_: self.destroy()))
            self._icon=pystray.Icon('airi-pc',im,'Airi-PC Companion',menu)
            threading.Thread(target=self._icon.run,daemon=True).start()
        except Exception: self._icon=None
    def refresh(self):
        alive=self.server and self.server.poll() is None
        self.status_var.set('SERVICE STARTED' if alive else 'OFFLINE'); self.auth_var.set('CONFIGURED' if keyring_get() else 'NOT CONFIGURED')
        self.after(1500,self.refresh)
    def ensure_server(self):
        if self.server and self.server.poll() is None: return
        root=Path(__file__).resolve().parents[1]
        env={**os.environ,'AIRIPC_COMPANION_STATE':str(APP_DIR)}
        self.server=subprocess.Popen([sys.executable,'-m','companion.cli','run'],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def connect(self):
        self.ensure_server(); token=keyring_get()
        if not token:
            root=Path(__file__).resolve().parents[1]; env={**os.environ,'AIRIPC_COMPANION_STATE':str(APP_DIR)}
            token=subprocess.check_output([sys.executable,'-m','companion.cli','pair'],cwd=root,env=env,text=True).strip(); keyring_set(token)
            token=None
        token=keyring_get()
        if self.cfg.get('airi_url'):
            try: api_get(self.cfg['airi_url'],'/health',token); self.connected=True; self.status_var.set('ONLINE'); self.auth_var.set('OK')
            except Exception as e: messagebox.showwarning('Airi non raggiungibile',str(e))
        else:
            self.status_var.set('LOCAL READY'); self.last_var.set('pending Airi transport'); messagebox.showinfo('Companion pronto','Il Companion è installato e protetto. Il trasporto Airi non è ancora configurato in questa installazione.')
    def disconnect(self): self.connected=False; self.status_var.set('DISCONNECTED')
    def game_mode(self): messagebox.showinfo('Game Mode','Game Agent foundation pronta. Il controllo reale richiede un canale Airi collegato e un GameProfile.')
    def stop_all(self):
        try:
            root=Path(__file__).resolve().parents[1]; env={**os.environ,'AIRIPC_COMPANION_STATE':str(APP_DIR)}
            subprocess.run([sys.executable,'-m','companion.cli','disable'],cwd=root,env=env,check=True,capture_output=True,text=True,timeout=3)
        except Exception: pass
        self.disconnect(); self.status_var.set('STOPPED')
    def diagnostics(self):
        self.ensure_server()
        try:
            with urllib.request.urlopen(f'http://{SERVICE_HOST}:{SERVICE_PORT}/health',timeout=3) as r: health=json.loads(r.read().decode())
            messagebox.showinfo('Diagnostica',json.dumps({'service':health,'paired':bool(keyring_get()),'airi_configured':bool(self.cfg.get('airi_url'))},indent=2))
        except Exception as e: messagebox.showerror('Diagnostica',str(e))

if __name__=='__main__': App().mainloop()
