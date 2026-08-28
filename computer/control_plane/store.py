from __future__ import annotations
import json, os, re, time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AIRIPC_WORKSPACE_ROOT", "/home/user/airi")).resolve()
AI = ROOT / ".ai"
CP = AI / "control_plane"

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
    try: return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except Exception: return default

def save_json(name: str, data: Any) -> None:
    atomic_json(ensure() / name, data)

def append_jsonl(name: str, row: dict[str, Any]) -> None:
    p = ensure() / name
    p.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")

def redact(value: Any) -> Any:
    if isinstance(value, dict): return {k: redact(v) for k,v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(token|secret|password|cookie|authorization|api[_-]?key)\s*[:=]\s*[^\s,}]+", r"\1=<redacted>", value)
    return value

def now() -> float: return time.time()
