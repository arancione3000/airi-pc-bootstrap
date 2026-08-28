from __future__ import annotations
import json, os, shlex, signal, subprocess, time, uuid
from pathlib import Path
from typing import Any
from .store import load_json, save_json, now
from coding import ROOT, safe_path

JOBS_FILE = 'jobs.json'
LOG_DIR = ROOT / '.ai' / 'state' / 'jobs'
ACTIVE = {'queued','running','detached'}
FINAL = {'completed','failed','cancelled','lost'}

class JobManager:
    def __init__(self):
        self.state = load_json(JOBS_FILE, {'version': 1, 'jobs': {}})
        self.state.setdefault('version', 1); self.state.setdefault('jobs', {})
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _save(self):
        save_json(JOBS_FILE, self.state)

    def _row(self, job_id: str) -> dict[str, Any]:
        row = self.state['jobs'].get(job_id)
        if row is None:
            raise KeyError(job_id)
        return row

    def _refresh_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if row.get('status') not in ACTIVE:
            return row
        exit_file = Path(row['exit_file'])
        if exit_file.exists():
            try:
                rc = int(exit_file.read_text().strip())
            except Exception:
                rc = None
            if rc is not None:
                row['exit_code'] = rc
                row['status'] = 'completed' if rc == 0 else 'failed'
                row['finished_at'] = row.get('finished_at') or now()
                row['termination_reason'] = 'exit_0' if rc == 0 else 'nonzero_exit'
        else:
            pid = int(row.get('pid') or 0)
            alive = False
            if pid > 0:
                try: os.kill(pid, 0); alive = True
                except ProcessLookupError: alive = False
                except PermissionError: alive = True
            if not alive and row.get('status') in ACTIVE:
                row['status'] = 'lost'
                row['finished_at'] = row.get('finished_at') or now()
                row['termination_reason'] = 'process_missing_without_exit_record'
        log = Path(row['log_file'])
        if log.exists():
            try:
                row['last_output'] = log.read_text(errors='replace')[-20000:]
            except Exception:
                pass
        return row

    def refresh(self):
        changed = False
        for row in self.state['jobs'].values():
            before = json.dumps(row, sort_keys=True)
            self._refresh_row(row)
            changed = changed or before != json.dumps(row, sort_keys=True)
        if changed: self._save()
        return self.list()

    def start(self, command: str, cwd: str = '.', timeout: int = 900, owner_task: str | None = None,
              scope: list[str] | None = None, allow_shell: bool = False) -> dict[str, Any]:
        if not isinstance(command, str) or not command.strip():
            raise ValueError('command required')
        if not 1 <= int(timeout) <= 86400:
            raise ValueError('timeout must be 1..86400 seconds')
        base = safe_path(cwd)
        if not allow_shell:
            blocked = ('rm -rf','shutdown','reboot','mkfs','dd ','sudo ','curl |','wget -O-')
            if any(x in command for x in blocked):
                raise PermissionError('risky shell blocked; explicit allow_shell=true required')
        if scope:
            safe_path(cwd, missing=True)
        job_id = uuid.uuid4().hex[:12]
        job_dir = LOG_DIR / job_id; job_dir.mkdir(parents=True, exist_ok=True)
        log_file = job_dir / 'output.log'; exit_file = job_dir / 'exit.code'
        env = os.environ.copy(); env['AIRI_JOB_ID'] = job_id
        quoted_log, quoted_exit = shlex.quote(str(log_file)), shlex.quote(str(exit_file))
        wrapped = f"trap 'rc=$?; printf \"%s\" \"$rc\" > {quoted_exit}' EXIT; timeout --signal=TERM --kill-after=10s {int(timeout)} /bin/bash -lc {shlex.quote(command)}"
        with log_file.open('ab') as out:
            proc = subprocess.Popen(['/bin/bash','-lc',wrapped], cwd=str(base), env=env,
                                    stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        row = {
            'id': job_id, 'command': command, 'cwd': str(base), 'status':'running', 'started_at':now(),
            'pid': proc.pid, 'exit_code': None, 'last_output':'', 'log_file':str(log_file),
            'exit_file':str(exit_file), 'checkpoint': None, 'owner_task': owner_task,
            'scope': list(scope or []), 'timeout': int(timeout), 'termination_reason': None,
            'detached': True, 'created_at':now(), 'updated_at':now()
        }
        self.state['jobs'][job_id] = row; self._save()
        return row

    def status(self, job_id: str) -> dict[str, Any]:
        row = self._refresh_row(self._row(job_id)); row['updated_at'] = now(); self._save(); return row

    def list(self) -> dict[str, Any]:
        rows=[]
        for job_id in list(self.state['jobs']):
            rows.append(self.status(job_id))
        return {'count': len(rows), 'jobs': rows}

    def attach(self, job_id: str, tail: int = 200) -> dict[str, Any]:
        row = self.status(job_id)
        log = Path(row['log_file'])
        text = log.read_text(errors='replace') if log.exists() else ''
        row['attached_output'] = text[-max(1,int(tail))*200:]
        return row

    def detach(self, job_id: str) -> dict[str, Any]:
        row = self._row(job_id)
        if row.get('status') == 'running': row['status'] = 'detached'; row['updated_at'] = now(); self._save()
        return row

    def cancel(self, job_id: str, grace: int = 5) -> dict[str, Any]:
        row = self.status(job_id)
        pid = int(row.get('pid') or 0)
        if row.get('status') in ACTIVE and pid > 0:
            try: os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError: pass
            deadline = time.monotonic() + max(1, int(grace))
            while time.monotonic() < deadline:
                time.sleep(0.1)
                row = self.status(job_id)
                if row.get('status') in FINAL: break
            if row.get('status') in ACTIVE:
                try: os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError: pass
                row['status']='cancelled'; row['termination_reason']='forced_cancel'; row['finished_at']=now(); self._save()
        elif row.get('status') not in FINAL:
            row['status']='cancelled'; row['termination_reason']='already_stopped'; row['finished_at']=now(); self._save()
        return self.status(job_id)

    def cleanup(self, keep_final: int = 100) -> dict[str, Any]:
        self.refresh(); finals=[r for r in self.state['jobs'].values() if r.get('status') in FINAL]
        finals.sort(key=lambda x:x.get('finished_at') or x.get('updated_at') or 0, reverse=True)
        removed=[]
        for row in finals[max(0,int(keep_final)):]:
            jid=row['id']; removed.append(jid); self.state['jobs'].pop(jid,None)
            try:
                p=Path(row['log_file']).parent
                for f in p.glob('*'): f.unlink()
                p.rmdir()
            except OSError: pass
        self._save(); return {'ok':True,'removed':removed,'remaining':len(self.state['jobs'])}
