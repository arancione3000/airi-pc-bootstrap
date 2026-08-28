from __future__ import annotations
import json, threading, time
from dataclasses import dataclass
from .store import CP, now, atomic_json

@dataclass
class Breaker:
    failures: int = 0
    successes: int = 0
    state: str = 'closed'
    opened_at: float = 0.0
    cooldown: float = 30.0

class ReliabilityRegistry:
    def __init__(self):
        self.path = CP / 'reliability.json'
        self._lock = threading.RLock()
        self.data = self._load()

    def _load(self):
        try:
            d = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
            d.setdefault('version', 1); d.setdefault('capabilities', {})
            return d
        except Exception:
            return {'version': 1, 'capabilities': {}}

    def _row(self, name):
        caps = self.data['capabilities']
        row = caps.setdefault(name, {'calls': 0, 'failures': 0, 'successes': 0, 'latency_ms_avg': None,
                                     'state': 'closed', 'last_error': None, 'last_seen': None,
                                     'consecutive_failures': 0})
        return row

    def record(self, name: str, ok: bool, latency_ms: float | None = None, error: str = ''):
        with self._lock:
            row = self._row(name)
            row['calls'] += 1; row['last_seen'] = now()
            if ok:
                row['successes'] += 1; row['consecutive_failures'] = 0
                row['state'] = 'closed'
                if latency_ms is not None:
                    old = row['latency_ms_avg']; row['latency_ms_avg'] = round(latency_ms if old is None else old * 0.8 + latency_ms * 0.2, 2)
            else:
                row['failures'] += 1; row['consecutive_failures'] += 1; row['last_error'] = str(error)[:1000]
                if row['consecutive_failures'] >= 3:
                    row['state'] = 'open'; row['opened_at'] = now()
            atomic_json(self.path, self.data)
            return dict(row)

    def allow(self, name: str) -> bool:
        with self._lock:
            row = self._row(name)
            if row.get('state') != 'open': return True
            if now() - float(row.get('opened_at', 0)) >= 30:
                row['state'] = 'half_open'; atomic_json(self.path, self.data); return True
            return False

    def summary(self):
        with self._lock:
            rows = list(self.data['capabilities'].values())
            total = sum(r.get('calls', 0) for r in rows)
            failures = sum(r.get('failures', 0) for r in rows)
            return {'capabilities': len(rows), 'calls': total, 'failures': failures,
                    'error_rate': round(failures / total, 4) if total else 0.0,
                    'open_breakers': [k for k, v in self.data['capabilities'].items() if v.get('state') == 'open']}

REGISTRY = ReliabilityRegistry()
