from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

REPO = "arancione3000/airi-pc-bootstrap"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELAY = "https://ntfy.sh"
_ENABLED = os.environ.get("AIRI_LIVE_TELEMETRY", "1").strip().lower() not in {"0", "false", "no", "off"}
_RELAY = os.environ.get("AIRI_LIVE_RELAY_BASE", DEFAULT_RELAY).rstrip("/")
_QUEUE: queue.Queue[dict] = queue.Queue(maxsize=256)
_STARTED = False
_LOCK = threading.Lock()
_LAST: dict[str, tuple[str, float]] = {}
_SECRET = re.compile(r"(?i)(authorization|bearer|token|password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+")


def source_sha() -> str:
    env = os.environ.get("AIRI_BOOTSTRAP_SHA", "").strip()
    if env:
        return env
    marker = ROOT / ".ai" / ".runtime_source_sha"
    try:
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
    except Exception:
        return "unknown"


def topic_for(sha: str | None = None) -> str:
    digest = hashlib.sha256(f"{REPO}:{sha or source_sha()}".encode()).hexdigest()[:24]
    return f"airi-live-{digest}"


def _safe(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    text = _SECRET.sub(r"\1=[redacted]", text)
    return text[:limit]


def _worker() -> None:
    while True:
        event = _QUEUE.get()
        try:
            body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                f"{_RELAY}/{topic_for(event.get('source_sha'))}",
                data=body,
                method="POST",
                headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "Airi-PC-Live/1.0"},
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                response.read(1)
        except Exception:
            # Telemetry must never be able to break an Airi-PC task.
            pass
        finally:
            _QUEUE.task_done()


def _ensure_worker() -> None:
    global _STARTED
    if _STARTED or not _ENABLED:
        return
    with _LOCK:
        if _STARTED:
            return
        threading.Thread(target=_worker, name="airi-live-telemetry", daemon=True).start()
        _STARTED = True


def live_emit(
    kind: str,
    title: str,
    detail: object = "",
    status: str = "info",
    *,
    task_id: str | None = None,
    node_id: str | None = None,
    dedupe_key: str | None = None,
) -> bool:
    """Publish redacted operational metadata without blocking the active task."""
    # Automated tests/CI must never create user-visible live activity.
    allow_ci = os.environ.get("AIRI_LIVE_ALLOW_CI", "").strip().lower() in {"1", "true", "yes", "on"}
    if (
        not _ENABLED
        or os.environ.get("PYTEST_CURRENT_TEST")
        or (os.environ.get("GITHUB_ACTIONS", "").lower() == "true" and not allow_ci)
    ):
        return False
    sha = source_sha()
    event = {
        "v": 1,
        "id": uuid.uuid4().hex[:16],
        "ts": int(time.time()),
        "kind": _safe(kind, 32) or "event",
        "title": _safe(title, 100) or "Airi-PC",
        "detail": _safe(detail, 180),
        "status": _safe(status, 32) or "info",
        "task_id": _safe(task_id, 32) if task_id else "",
        "node_id": _safe(node_id, 64) if node_id else "",
        "source_sha": sha,
    }
    key = dedupe_key or f"{event['kind']}:{event['title']}:{event['status']}:{event['task_id']}:{event['node_id']}"
    fingerprint = json.dumps(event, sort_keys=True, separators=(",", ":"))
    now = time.monotonic()
    previous = _LAST.get(key)
    if previous and previous[0] == fingerprint and now - previous[1] < 30:
        return False
    _LAST[key] = (fingerprint, now)
    _ensure_worker()
    try:
        _QUEUE.put_nowait(event)
        return True
    except queue.Full:
        return False
