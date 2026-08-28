from __future__ import annotations
import os, subprocess, time
from .store import now, save_json

BASE = '/home/user/airi'
COMPONENTS = {
    'mcp': 'http://127.0.0.1:9010/ready',
    'status': 'http://127.0.0.1:9010/status',
}

class Supervisor:
    def __init__(self, base: str = BASE): self.base = base

    def process_snapshot(self):
        p = subprocess.run(['bash','-lc', "pgrep -af 'contract_server|airi-watchdog|airi-tunnel-supervisor|uvicorn' || true"], cwd=self.base, text=True, capture_output=True)
        return [x for x in p.stdout.splitlines() if x.strip()]

    def http_probe(self, url: str):
        p = subprocess.run(['curl','-fsS','--max-time','5',url], cwd=self.base, text=True, capture_output=True)
        return {'ok': p.returncode == 0, 'output': p.stdout[-4000:], 'error': p.stderr[-1000:]}

    def snapshot(self):
        ready = self.http_probe(COMPONENTS['mcp'])
        status = self.http_probe(COMPONENTS['status'])
        row = {'timestamp': now(), 'ready': ready, 'status': status, 'processes': self.process_snapshot()}
        save_json('supervisor.json', row)
        return row

    def health(self):
        return self.snapshot()

    def recover(self):
        snap = self.snapshot()
        actions = []
        if not snap['ready']['ok']:
            script = os.path.join(self.base, 'computer', 'start.sh')
            subprocess.Popen(['/bin/sh', script], cwd=self.base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            actions.append('runtime_restart')
        out = self.snapshot()
        return {'ok': bool(out['ready']['ok']), 'actions': actions, 'before': snap, 'after': out}
