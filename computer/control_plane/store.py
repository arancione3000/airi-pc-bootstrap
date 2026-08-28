from __future__ import annotations
import json, os, re, time, threading
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

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
