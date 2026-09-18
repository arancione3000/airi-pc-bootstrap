from __future__ import annotations

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
DEFAULT_TOPIC = "airi-live-09d926b9a4207c660a5f3efa5f50505ef5ff46e8463b3beb"
CONFIG_PATH = ROOT / ".ai" / "airi_live.json"
SESSION_PATH = ROOT / ".ai" / "state" / "live_session_id"


def _load_protocol_config() -> dict:
    data = {"protocol": 2, "relay_base": DEFAULT_RELAY, "topic": DEFAULT_TOPIC, "history": "24h"}
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data.update({k: raw[k] for k in ("protocol", "relay_base", "topic", "history") if k in raw})
    except Exception:
        pass
    env_topic = os.environ.get("AIRI_LIVE_TOPIC", "").strip()
    env_relay = os.environ.get("AIRI_LIVE_RELAY_BASE", "").strip()
    if env_topic:
        data["topic"] = env_topic
    if env_relay:
        data["relay_base"] = env_relay
    return data


_PROTOCOL = _load_protocol_config()
STABLE_TOPIC = str(_PROTOCOL.get("topic") or DEFAULT_TOPIC)
_ENABLED = os.environ.get("AIRI_LIVE_TELEMETRY", "1").strip().lower() not in {"0", "false", "no", "off"}
_RELAY = str(_PROTOCOL.get("relay_base") or DEFAULT_RELAY).rstrip("/")
def _session_identity() -> str:
    explicit = os.environ.get("AIRI_LIVE_SESSION_ID", "").strip()
    if explicit:
        return explicit
    if os.environ.get("GITHUB_RUN_ID"):
        return f"gh-{os.environ.get('GITHUB_RUN_ID')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"

    try:
        existing = SESSION_PATH.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass

    value = f"session-{uuid.uuid4().hex[:12]}"
    try:
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(SESSION_PATH), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value + "\n")
        return value
    except FileExistsError:
        try:
            existing = SESSION_PATH.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError:
            pass
    except OSError:
        pass
    return value


_SESSION_ID = _session_identity()
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
    """Return the persistent Airi Live topic.

    The optional sha argument is intentionally ignored for backwards compatibility.
    Rebuilds, new commits, new chat sessions and new runners all publish to the same
    protocol topic configured in .ai/airi_live.json.
    """
    return STABLE_TOPIC


def session_id() -> str:
    """Return the stable identifier for this one Airi-PC runtime execution."""
    return _SESSION_ID


def _safe(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    text = _SECRET.sub(r"\1=[redacted]", text)
    return text[:limit]


def _worker() -> None:
    while True:
        event = _QUEUE.get()
        try:
            body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            for attempt in range(3):
                try:
                    req = urllib.request.Request(
                        f"{_RELAY}/{topic_for()}",
                        data=body,
                        method="POST",
                        headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "Airi-PC-Live/2.0"},
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        response.read(1)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.4 * (attempt + 1))
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
        "session_id": session_id(),
        "protocol": int(_PROTOCOL.get("protocol") or 2),
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


def live_flush(timeout: float = 5.0) -> bool:
    """Wait briefly for queued telemetry to reach the relay."""
    deadline = time.monotonic() + max(0.0, timeout)
    while _QUEUE.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.05)
    return _QUEUE.unfinished_tasks == 0
