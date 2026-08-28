from __future__ import annotations
import json, os, re, time, threading, tempfile, fcntl, glob
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AIRIPC_WORKSPACE_ROOT", "/home/user/airi")).resolve()
AI = ROOT / ".ai"
CP = AI / "control_plane"
_STORE_LOCK = threading.RLock()

def ensure() -> Path:
    CP.mkdir(parents=True, exist_ok=True)
    return CP

def atomic_json(path: Path, data: Any) -> None:
    """Crash-safe, multiprocess-safe JSON replacement in the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with lock_path.open("a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            for stale in glob.glob(str(path.parent / (path.name + ".tmp.*"))):
                if stale != str(tmp):
                    try: Path(stale).unlink()
                    except OSError: pass
        finally:
            try: tmp.unlink()
            except FileNotFoundError: pass
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

def load_json(name: str, default: Any) -> Any:
    p = ensure() / name
    with _STORE_LOCK:
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            corrupt = p.with_suffix(p.suffix + ".corrupt")
            try:
                if not corrupt.exists(): p.replace(corrupt)
            except OSError:
                pass
            raise RuntimeError(f"corrupt persistent state: {p}: {exc}") from exc

def save_json(name: str, data: Any) -> None:
    with _STORE_LOCK:
        atomic_json(ensure() / name, data)

def append_jsonl(name: str, row: dict[str, Any]) -> None:
    p = ensure() / name
    with _STORE_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

def redact(value: Any) -> Any:
    if isinstance(value, dict): return {k: redact(v) for k,v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(token|secret|password|cookie|authorization|api[_-]?key)\s*[:=]\s*[^\s,}]+", r"\1=<redacted>", value)
    return value

def now() -> float: return time.time()
