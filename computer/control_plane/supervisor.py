from __future__ import annotations
import argparse, fcntl, os, signal, subprocess, time
from pathlib import Path
try:
    from .store import now, save_json
except ImportError:
    from store import now, save_json

BASE = '/home/user/airi'
READY_URL = 'http://127.0.0.1:9010/ready'
STATUS_URL = 'http://127.0.0.1:9010/status'
LOOP_INTERVAL = float(os.environ.get('AIRI_SUPERVISOR_INTERVAL', '5'))
PROBE_TIMEOUT = float(os.environ.get('AIRI_SUPERVISOR_PROBE_TIMEOUT', '5'))
RECOVERY_TIMEOUT = float(os.environ.get('AIRI_SUPERVISOR_RECOVERY_TIMEOUT', '30'))
MAX_RECOVERY_RETRIES = int(os.environ.get('AIRI_SUPERVISOR_MAX_RETRIES', '3'))


class Supervisor:
    def __init__(self, base: str = BASE):
        self.base = Path(base).resolve()
        self.state_dir = self.base / '.ai' / 'state'
        self.lock_path = self.state_dir / 'supervisor.lock'
        self.pid_path = self.state_dir / 'supervisor.pid'
        self._lock_fh = None
        self._stop = False

    def acquire_single_instance(self) -> bool:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock_fh = self.lock_path.open('a+')
        try:
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_fh.close(); self._lock_fh = None
            return False
        self.pid_path.write_text(str(os.getpid()) + '\n', encoding='utf-8')
        return True

    def release_single_instance(self) -> None:
        try:
            if self.pid_path.exists() and self.pid_path.read_text(encoding='utf-8').strip() == str(os.getpid()):
                self.pid_path.unlink()
        except OSError:
            pass
        if self._lock_fh is not None:
            try: fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_fh.close(); self._lock_fh = None

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=str(self.base), text=True, capture_output=True, timeout=timeout)

    def process_snapshot(self):
        try:
            p = self._run(['/bin/bash','-lc', "pgrep -af 'contract_server|airi-watchdog|airi-tunnel-supervisor|uvicorn' || true"], 5)
        except subprocess.TimeoutExpired:
            return ['PROCESS_SNAPSHOT_TIMEOUT']
        return [x for x in p.stdout.splitlines() if x.strip()]

    def http_probe(self, url: str):
        try:
            p = self._run(['curl','-fsS','--max-time',str(PROBE_TIMEOUT),url], PROBE_TIMEOUT + 2)
            return {'ok': p.returncode == 0, 'output': p.stdout[-4000:], 'error': p.stderr[-1000:], 'returncode': p.returncode}
        except subprocess.TimeoutExpired:
            return {'ok': False, 'output': '', 'error': 'probe_timeout', 'returncode': 124}

    def snapshot(self):
        ready = self.http_probe(READY_URL)
        status = self.http_probe(STATUS_URL)
        row = {'timestamp': now(), 'ready': ready, 'status': status, 'processes': self.process_snapshot()}
        save_json('supervisor.json', row)
        return row

    def health(self):
        return self.snapshot()

    def _wait_ready(self, deadline: float):
        last = None
        while time.monotonic() < deadline and not self._stop:
            last = self.snapshot()
            if last['ready']['ok'] and last['status']['ok']:
                return last
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
        return last or self.snapshot()

    def recover(self):
        before = self.snapshot()
        if before['ready']['ok'] and before['status']['ok']:
            return {'ok': True, 'actions': [], 'attempts': 0, 'before': before, 'after': before}
        actions = []
        attempts = 0
        last = before
        script = self.base / 'computer' / 'start.sh'
        for attempts in range(1, MAX_RECOVERY_RETRIES + 1):
            if not script.exists():
                return {'ok': False, 'actions': actions, 'attempts': attempts, 'before': before, 'after': last, 'error': 'start_script_missing'}
            actions.append('runtime_restart')
            started = time.monotonic()
            try:
                proc = subprocess.Popen(['/bin/sh', str(script)], cwd=str(self.base), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            except OSError as exc:
                last = {'timestamp': now(), 'ready': {'ok': False, 'error': str(exc)}, 'status': {'ok': False}, 'processes': []}
                continue
            last = self._wait_ready(min(time.monotonic() + RECOVERY_TIMEOUT, started + RECOVERY_TIMEOUT))
            if last['ready']['ok'] and last['status']['ok']:
                return {'ok': True, 'actions': actions, 'attempts': attempts, 'before': before, 'after': last}
            if proc.poll() is None and time.monotonic() >= started + RECOVERY_TIMEOUT:
                try: proc.terminate()
                except OSError: pass
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill()
            if attempts < MAX_RECOVERY_RETRIES:
                time.sleep(min(2 ** (attempts - 1), 8))
        return {'ok': False, 'actions': actions, 'attempts': attempts, 'before': before, 'after': last, 'error': 'recovery_verification_failed'}

    def run_once(self):
        row = self.snapshot()
        if row['ready']['ok'] and row['status']['ok']:
            return {'ok': True, 'recovered': False, 'snapshot': row}
        rec = self.recover()
        return {'ok': rec['ok'], 'recovered': True, 'recovery': rec}

    def stop(self, *_args):
        self._stop = True

    def run_forever(self):
        if not self.acquire_single_instance():
            return 20
        signal.signal(signal.SIGTERM, self.stop); signal.signal(signal.SIGINT, self.stop)
        try:
            while not self._stop:
                result = self.run_once()
                save_json('supervisor-run.json', {'timestamp': now(), **result})
                if not result['ok']:
                    time.sleep(min(LOOP_INTERVAL * 2, 15))
                else:
                    time.sleep(max(0.5, LOOP_INTERVAL))
        finally:
            self.release_single_instance()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    supervisor = Supervisor()
    if args.once:
        return 0 if supervisor.run_once()['ok'] else 1
    return supervisor.run_forever()


if __name__ == '__main__':
    raise SystemExit(main())
