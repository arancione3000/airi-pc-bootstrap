from __future__ import annotations
import json, os, threading, time, tkinter as tk
from http.server import ThreadingHTTPServer
from pathlib import Path
from tkinter import messagebox, ttk

from companion.server import Companion, Handler

APP_DIR = Path(os.environ.get('AIRIPC_COMPANION_STATE', Path.home() / '.airi-pc-companion'))
CFG = APP_DIR / 'app.json'
DEFAULTS = {'version': '0.2.1', 'airi_url': ''}
SERVICE_HOST = '127.0.0.1'
SERVICE_PORT = 17894


def load_cfg():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG.exists():
        CFG.write_text(json.dumps(DEFAULTS), encoding='utf-8')
    try:
        return {**DEFAULTS, **json.loads(CFG.read_text(encoding='utf-8'))}
    except Exception:
        return DEFAULTS.copy()


def save_cfg(data):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps({k: v for k, v in data.items() if k not in {'token', 'pairing_token'}}, indent=2), encoding='utf-8')
    try:
        os.chmod(CFG, 0o600)
    except OSError:
        pass


def keyring_get():
    try:
        import keyring
        return keyring.get_password('Airi-PC Companion', 'pairing-token')
    except Exception:
        return None


def keyring_set(token):
    import keyring
        
    keyring.set_password('Airi-PC Companion', 'pairing-token', token)


def api_get(base, path, token=''):
    if not base:
        raise RuntimeError('Airi endpoint not configured')
    import urllib.request
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    req = urllib.request.Request(base.rstrip('/') + path, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Airi-PC Companion')
        self.geometry('540x500')
        self.resizable(False, False)
        self.cfg = load_cfg()
        self.httpd = None
        self.http_thread = None
        self.companion = None
        self.connected = False
        self.status_var = tk.StringVar(value='OFFLINE')
        self.auth_var = tk.StringVar(value='NOT CONFIGURED')
        self.pc_var = tk.StringVar(value='LOCAL')
        self.last_var = tk.StringVar(value='—')
        self._build()
        self._tray()
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.after(500, self.refresh)

    def _build(self):
        ttk.Label(self, text='AIRI-PC COMPANION', font=('Segoe UI', 18, 'bold')).pack(pady=(18, 3))
        ttk.Label(self, text='Secure desktop companion', font=('Segoe UI', 10)).pack()
        box = ttk.LabelFrame(self, text='Connection', padding=14)
        box.pack(fill='x', padx=20, pady=15)
        for label, var in (('Connection', self.status_var), ('Authentication', self.auth_var), ('PC', self.pc_var), ('Last connection', self.last_var)):
            row = ttk.Frame(box)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, width=18).pack(side='left')
            ttk.Label(row, textvariable=var).pack(side='left')
        btns = ttk.Frame(self)
        btns.pack(pady=4)
        ttk.Button(btns, text='CONNETTI AIRI', command=self.connect).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(btns, text='DISCONNETTI', command=self.disconnect).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(btns, text='GAME MODE', command=self.game_mode).grid(row=1, column=0, padx=4, pady=4)
        ttk.Button(btns, text='STOP ALL', command=self.stop_all).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(btns, text='DIAGNOSTICA', command=self.diagnostics).grid(row=2, column=0, columnspan=2, padx=4, pady=4)
        ttk.Label(self, text=f"Version {self.cfg['version']}").pack(pady=12)
        ttk.Label(self, text='Pairing secret stored only in the OS credential store.', font=('Segoe UI', 8)).pack()

    def _tray(self):
        try:
            import pystray
            from PIL import Image, ImageDraw
            im = Image.new('RGB', (64, 64), 'black')
            d = ImageDraw.Draw(im)
            d.rectangle((12, 12, 52, 52), outline='white', width=4)
            menu = pystray.Menu(
                pystray.MenuItem('Open', lambda *_: self.deiconify()),
                pystray.MenuItem('STOP ALL', lambda *_: self.stop_all()),
                pystray.MenuItem('Quit', lambda *_: self._close()),
            )
            self._icon = pystray.Icon('airi-pc', im, 'Airi-PC Companion', menu)
            threading.Thread(target=self._icon.run, daemon=True).start()
        except Exception:
            self._icon = None

    def _start_server(self):
        if self.httpd:
            return
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self.companion = Companion(APP_DIR)
        self.httpd = ThreadingHTTPServer((SERVICE_HOST, SERVICE_PORT), Handler)
        self.httpd.companion = self.companion
        self.httpd.timeout = 1

        def loop():
            try:
                while self.companion and not self.companion.stop_event.is_set():
                    self.httpd.handle_request()
            finally:
                if self.httpd:
                    self.httpd.server_close()

        self.http_thread = threading.Thread(target=loop, name='airi-companion-server', daemon=True)
        self.http_thread.start()

    def _stop_server(self):
        if self.companion:
            self.companion.stop_event.set()
        if self.httpd:
            try:
                import socket
                with socket.create_connection((SERVICE_HOST, SERVICE_PORT), timeout=0.2):
                    pass
            except Exception:
                pass
        if self.http_thread and self.http_thread.is_alive():
            self.http_thread.join(timeout=2)
        self.httpd = None
        self.http_thread = None
        self.companion = None

    def refresh(self):
        alive = bool(self.http_thread and self.http_thread.is_alive())
        paired = bool(keyring_get())
        self.status_var.set('ONLINE' if self.connected else ('SERVICE READY' if alive else 'OFFLINE'))
        self.auth_var.set('OK' if paired else 'NOT CONFIGURED')
        self.pc_var.set('CONNECTED' if self.connected else 'LOCAL')
        self.after(1500, self.refresh)

    def connect(self):
        self._start_server()
        token = keyring_get()
        if not token and self.companion:
            token = self.companion.auth.initial_token
            if token:
                keyring_set(token)
            else:
                messagebox.showerror('Pairing', 'Pairing non disponibile. Revoca il pairing e riprova.')
                return
        if self.cfg.get('airi_url'):
            try:
                api_get(self.cfg['airi_url'], '/health', keyring_get() or '')
                self.connected = True
                self.last_var.set(time.strftime('%Y-%m-%d %H:%M:%S'))
            except Exception as exc:
                self.connected = False
                messagebox.showwarning('Airi non raggiungibile', str(exc))
        else:
            self.status_var.set('LOCAL READY')
            messagebox.showinfo('Companion pronto', 'Il Companion è installato e protetto. Il trasporto Airi non è ancora configurato in questa installazione.')

    def disconnect(self):
        self.connected = False
        self.status_var.set('DISCONNECTED')
        self.pc_var.set('LOCAL')

    def game_mode(self):
        messagebox.showinfo('Game Mode', 'Game Agent foundation pronta. Il controllo reale richiede un canale Airi collegato e un GameProfile.')

    def stop_all(self):
        if self.companion:
            try:
                self.companion.disable_file.touch(mode=0o600, exist_ok=True)
            except Exception:
                pass
        self.connected = False
        self.status_var.set('STOPPED')

    def diagnostics(self):
        self._start_server()
        try:
            import urllib.request
            with urllib.request.urlopen(f'http://{SERVICE_HOST}:{SERVICE_PORT}/health', timeout=3) as r:
                health = json.loads(r.read().decode())
            messagebox.showinfo('Diagnostica', json.dumps({
                'service': health,
                'paired': bool(keyring_get()),
                'airi_configured': bool(self.cfg.get('airi_url')),
            }, indent=2))
        except Exception as exc:
            messagebox.showerror('Diagnostica', str(exc))

    def _close(self):
        self._stop_server()
        try:
            if getattr(self, '_icon', None):
                self._icon.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
