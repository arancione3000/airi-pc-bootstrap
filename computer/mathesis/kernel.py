from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


IMMUTABLE_KERNEL = (
    "safe_math.py",
    "verifiers.py",
    "kernel.py",
)

MUTABLE_STATE_NAMES = {
    "champion.json",
    "router.json",
    "history.jsonl",
    "history-manifest.json",
    "knowledge.json",
    "candidate_model.py",
    "champion_model.py",
    "discoveries.json",
    "discovery-history.jsonl",
    "curriculum.json",
    "health.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IntegrityKernel:
    """Hard boundary around self-improvement.

    MATHESIS can rewrite its architecture genome and learned router state, but
    it cannot promote a candidate that changed this verifier/kernel set.
    """

    def __init__(self, package_dir: str | Path | None = None):
        self.package_dir = Path(package_dir or Path(__file__).resolve().parent).resolve()

    def snapshot(self) -> dict[str, str]:
        return {
            name: _sha256(self.package_dir / name)
            for name in IMMUTABLE_KERNEL
        }

    def verify_snapshot(self, snapshot: dict[str, str]) -> dict[str, Any]:
        current = self.snapshot()
        changed = [
            name for name in IMMUTABLE_KERNEL
            if snapshot.get(name) != current.get(name)
        ]
        return {
            "ok": not changed,
            "changed": changed,
            "current": current,
        }

    def validate_state_path(self, state_dir: str | Path, path: str | Path) -> Path:
        root = Path(state_dir).resolve()
        target = Path(path).resolve()
        if target == root or root not in target.parents:
            raise PermissionError("self-rewrite target must be an authorized state file")
        if target.parent != root:
            raise PermissionError("self-rewrite target cannot use nested state paths")
        if target.name not in MUTABLE_STATE_NAMES:
            raise PermissionError(f"self-rewrite target is not authorized: {target.name}")
        return target


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
