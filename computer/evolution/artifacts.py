from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from .data import class_counts, load_records


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report(state_dir: Path, history_limit: int = 20) -> dict[str, Any]:
    state_dir = Path(state_dir)
    champion_dir = state_dir / "champion"
    metrics = {}
    provenance = {}
    genome = {}
    for name, target in (("metrics.json", metrics), ("provenance.json", provenance), ("genome.json", genome)):
        path = champion_dir / name
        if path.exists():
            try:
                target.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    records = load_records(state_dir / "data" / "verified.jsonl")
    history = []
    history_path = state_dir / "history.jsonl"
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except Exception:
                pass
    history = history[-max(1, int(history_limit)):]
    trends = []
    for row in history:
        cand = row.get("candidate") or {}
        trends.append({
            "run_id": row.get("run_id"),
            "mode": row.get("mode"),
            "promoted": row.get("promoted"),
            "f1": cand.get("f1"),
            "f1_real": cand.get("f1_real"),
            "f1_fake": cand.get("f1_fake"),
            "params": cand.get("params"),
            "latency_ms": cand.get("latency_ms"),
            "model_bytes": cand.get("model_bytes"),
        })
    edge = None
    edge_meta = state_dir / "edge" / "metadata.json"
    if edge_meta.exists():
        try:
            edge = json.loads(edge_meta.read_text(encoding="utf-8"))
        except Exception:
            edge = {"error": "invalid edge metadata"}
    try:
        from .monitoring import drift_report
        drift = drift_report(state_dir, min_window=10)
    except Exception as exc:
        drift = {"ok": False, "drift": False, "error": repr(exc)}
    return {
        "ok": True,
        "dataset_records": len(records),
        "class_counts": class_counts(records),
        "champion": {"metrics": metrics or None, "provenance": provenance or None, "genome": genome or None},
        "edge": edge,
        "drift": drift,
        "recent_runs": trends,
    }


def export_bundle(state_dir: Path, out_path: Path | None = None, include_torchscript: bool = False) -> dict[str, Any]:
    state_dir = Path(state_dir)
    champion = state_dir / "champion"
    required = [champion / "genome.json", champion / "model.pt", champion / "metrics.json", champion / "provenance.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"ok": False, "error": "champion_incomplete", "missing": missing}
    exports = state_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = exports / f"champion-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    out_path = Path(out_path)
    work = exports / ".bundle"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    files = []
    for src in required:
        dst = work / src.name
        shutil.copy2(src, dst)
        files.append(dst)
    for src in (state_dir / "edge" / "model-int8.pt", state_dir / "edge" / "metadata.json"):
        if src.exists():
            dst = work / src.name
            shutil.copy2(src, dst)
            files.append(dst)
    torchscript_error = None
    if include_torchscript:
        try:
            import torch
            from .engine import load_champion_for_prediction
            genome, model = load_champion_for_prediction(state_dir)
            model.eval()
            ids = torch.zeros((1, genome.max_len), dtype=torch.long)
            mask = torch.ones((1, genome.max_len), dtype=torch.bool)
            traced = torch.jit.trace(model, (ids, mask), strict=False)
            ts_path = work / "model.ts"
            traced.save(str(ts_path))
            files.append(ts_path)
        except Exception as exc:
            torchscript_error = repr(exc)
    manifest = {
        "format": "airi-pc-neuroevolution-bundle-v1",
        "created_at": time.time(),
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files},
        "torchscript_requested": bool(include_torchscript),
        "torchscript_error": torchscript_error,
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    files.append(manifest_path)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    shutil.rmtree(work, ignore_errors=True)
    return {"ok": True, "path": str(out_path), "bytes": out_path.stat().st_size, "sha256": _sha256(out_path), "manifest": manifest}
