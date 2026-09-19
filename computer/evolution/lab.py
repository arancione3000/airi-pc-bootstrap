from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .data import _dataset_lock, class_counts, load_records, normalize_label, text_fingerprint
from .engine import EvolutionConfig, predict_text, run_evolution

ROOT = Path(os.environ.get("AIRI_ROOT") or Path(__file__).resolve().parents[2]).resolve()
LAB_STATE = Path(os.environ.get("AIRI_EVOLUTION_LAB_STATE") or ROOT / ".ai" / "evolution-lab" / "shadow-router").resolve()
RAW = LAB_STATE / "raw" / "observations.jsonl"
DATA = LAB_STATE / "data" / "verified.jsonl"
META = LAB_STATE / "lab-meta.json"

_SECRET_RE = re.compile(
    r"(?i)(token|secret|password|cookie|authorization|api[_-]?key)\s*[:=]\s*[^\s,;}]+"
)
_URL_RE = re.compile(r"https?://\S+", re.I)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
_PATH_RE = re.compile(r"(?<!\w)(?:[A-Za-z]:\\[^\s]+|/(?:[^\s/]+/)+[^\s]*)")

def ensure_state_boundary() -> Path:
    workspace = ROOT.resolve(strict=False)
    resolved = LAB_STATE.resolve(strict=False)
    if resolved == workspace or workspace not in resolved.parents:
        raise RuntimeError(f"Evolution Lab state escaped Airi workspace: {resolved}")
    return resolved



def _json_read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_write(path: Path, value: Any) -> None:
    root = ensure_state_boundary()
    target = Path(path).resolve(strict=False)
    if target != root and root not in target.parents:
        raise RuntimeError(f"Evolution Lab blocked write outside state root: {target}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sanitize_goal(value: Any, max_chars: int = 600) -> str:
    text = str(value or "")
    text = _SECRET_RE.sub(r"\1=<redacted>", text)
    text = _URL_RE.sub("<url>", text)
    text = _EMAIL_RE.sub("<email>", text)
    text = _PATH_RE.sub("<path>", text)
    text = _LONG_NUMBER_RE.sub("<number>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max(80, int(max_chars))]


def _arg_schema(args: Any) -> str:
    if not isinstance(args, dict):
        return type(args).__name__
    bits = []
    for key in sorted(str(k) for k in args.keys())[:24]:
        value = args.get(key)
        bits.append(f"{key}:{type(value).__name__}")
    return ",".join(bits) or "none"


def route_feature(goal: str, operation: str, tool: str, args: Any = None) -> str:
    return (
        f"goal {sanitize_goal(goal)} "
        f"operation {str(operation or 'unknown')[:80]} "
        f"tool {str(tool or 'unknown')[:120]} "
        f"arg_schema {_arg_schema(args)}"
    )


def _error_class(error: Any) -> str:
    text = str(error or "").lower()
    if not text:
        return "none"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "permission" in text or "forbidden" in text or "scope" in text:
        return "permission"
    if "network" in text or "connection" in text or "http" in text:
        return "network"
    if "not found" in text or "missing" in text:
        return "dependency"
    if "memory" in text or "oom" in text or "killed" in text:
        return "resource"
    return "tool_error"


def _read_observations() -> list[dict[str, Any]]:
    if not RAW.exists():
        return []
    out = []
    with RAW.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("feature") and row.get("label") in (0, 1):
                    out.append(row)
            except Exception:
                continue
    return out


def record_execution(
    *,
    goal: str,
    operation: str,
    tool: str,
    args: Any,
    success: bool,
    latency_ms: float | None = None,
    error: Any = None,
    task_id: str | None = None,
    node_id: str | None = None,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    feature = route_feature(goal, operation, tool, args)
    stamp = time.time()
    raw_id = hashlib.sha256(
        f"{stamp}:{task_id or ''}:{node_id or ''}:{feature}:{bool(success)}".encode("utf-8")
    ).hexdigest()[:20]
    row = {
        "id": raw_id,
        "created_at": stamp,
        "feature": feature,
        "feature_id": text_fingerprint(feature),
        "label": 1 if success else 0,
        "operation": str(operation or "unknown")[:80],
        "tool": str(tool or "unknown")[:120],
        "candidate_count": len(candidates or []),
        "latency_bucket": (
            "unknown" if latency_ms is None else
            "fast" if float(latency_ms) < 250 else
            "medium" if float(latency_ms) < 1500 else
            "slow"
        ),
        "error_class": _error_class(error),
    }
    ensure_state_boundary()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with _dataset_lock(RAW, timeout=5.0, stale_after=900.0):
        with RAW.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ok": True, "recorded": True, "id": raw_id}



def compact_raw_observations(max_rows: int = 50_000) -> dict[str, Any]:
    ensure_state_boundary()
    rows = _read_observations()
    limit = max(1000, int(max_rows))
    if len(rows) <= limit:
        return {"compacted": False, "rows": len(rows)}
    kept = rows[-limit:]
    tmp = RAW.with_suffix(".compact.tmp")
    with _dataset_lock(RAW, timeout=10.0, stale_after=900.0):
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(RAW)
    return {"compacted": True, "rows": len(kept), "dropped": len(rows) - len(kept)}


def prune_lab_storage(max_runs: int = 50, max_history_lines: int = 200) -> dict[str, Any]:
    root = ensure_state_boundary()
    removed_runs = []
    runs = root / "runs"
    if runs.exists():
        dirs = sorted(
            [path for path in runs.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old in dirs[max(5, int(max_runs)):]:
            shutil.rmtree(old, ignore_errors=True)
            removed_runs.append(old.name)

    history = root / "history.jsonl"
    trimmed = 0
    if history.exists():
        lines = history.read_text(encoding="utf-8", errors="replace").splitlines()
        keep = max(20, int(max_history_lines))
        if len(lines) > keep:
            trimmed = len(lines) - keep
            tmp = history.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")
            tmp.replace(history)

    return {"removed_runs": removed_runs, "trimmed_history_lines": trimmed}

def rebuild_dataset(max_observations: int = 20_000) -> dict[str, Any]:
    ensure_state_boundary()
    compact_raw_observations()
    rows = _read_observations()
    if max_observations > 0:
        rows = rows[-int(max_observations):]

    DATA.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA.with_suffix(".tmp")
    positives = 0
    negatives = 0
    features = set()
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            label = normalize_label(row["label"])
            feature = str(row["feature"])
            feature_id = str(row.get("feature_id") or text_fingerprint(feature))
            positives += int(label == 1)
            negatives += int(label == 0)
            features.add(feature_id)
            training_row = {
                "id": str(row["id"]),
                "text": feature,
                "text_id": feature_id,
                "label": label,
                "source": "airi-shadow-route-observation",
                "evidence": json.dumps(
                    {
                        "operation": row.get("operation"),
                        "tool": row.get("tool"),
                        "latency_bucket": row.get("latency_bucket"),
                        "error_class": row.get("error_class"),
                        "candidate_count": row.get("candidate_count"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "added_at": float(row.get("created_at") or time.time()),
            }
            handle.write(json.dumps(training_row, ensure_ascii=False) + "\n")
    tmp.replace(DATA)
    return {
        "ok": True,
        "observations": len(rows),
        "unique_route_features": len(features),
        "success": positives,
        "failure": negatives,
        "dataset": str(DATA),
    }


def status() -> dict[str, Any]:
    raw = _read_observations()
    records = load_records(DATA)
    counts = class_counts(records)
    meta = _json_read(META, {})
    champion = _json_read(LAB_STATE / "champion" / "metrics.json", None)
    return {
        "ok": True,
        "mode": "shadow_only",
        "isolation": {
            "production_write_access": False,
            "network_required_for_training": False,
            "can_execute_tools": False,
            "can_modify_router": False,
            "state_dir": str(LAB_STATE),
        },
        "raw_observations": len(raw),
        "training_records": len(records),
        "success_records": counts.get("real", 0),
        "failure_records": counts.get("fake", 0),
        "unique_route_features": len({row.get("feature_id") for row in raw if row.get("feature_id")}),
        "last_cycle_raw_count": int(meta.get("last_cycle_raw_count", 0) or 0),
        "pending_observations": max(0, len(raw) - int(meta.get("last_cycle_raw_count", 0) or 0)),
        "champion": champion,
        "last_cycle": meta.get("last_cycle"),
    }


def run_cycle(
    *,
    population: int = 6,
    generations: int = 2,
    candidate_epochs: int = 1,
    finalist_epochs: int = 2,
) -> dict[str, Any]:
    ensure_state_boundary()
    rebuilt = rebuild_dataset()
    if rebuilt["observations"] < 40 or rebuilt["unique_route_features"] < 12 or min(rebuilt["success"], rebuilt["failure"]) < 4:
        result = {
            "ok": True,
            "trained": False,
            "reason": "insufficient_balanced_shadow_observations",
            "dataset": rebuilt,
        }
        _json_write(META, {
            **_json_read(META, {}),
            "last_cycle": result,
            "updated_at": time.time(),
        })
        return result

    cfg = EvolutionConfig.for_mode(
        "safe",
        population=max(4, min(12, int(population))),
        generations=max(1, min(5, int(generations))),
        candidate_epochs=max(1, min(3, int(candidate_epochs))),
        finalist_epochs=max(1, min(4, int(finalist_epochs))),
        vocab_size=4096,
        max_params=500_000,
        max_latency_ms=25.0,
        promotion_repeats=2,
        min_promotion_votes=1,
        min_f1_first_champion=0.55,
        abstain_threshold=0.65,
    )
    result = run_evolution(LAB_STATE, cfg)
    meta = _json_read(META, {})
    meta.update({
        "last_cycle_raw_count": rebuilt["observations"],
        "last_cycle": {
            "run_id": result.get("run_id"),
            "promoted": result.get("promoted"),
            "promotion_reason": result.get("promotion_reason"),
            "candidate": result.get("candidate"),
            "completed_at": time.time(),
        },
        "updated_at": time.time(),
    })
    _json_write(META, meta)
    storage = prune_lab_storage()
    return {"ok": True, "trained": True, "dataset": rebuilt, "evolution": result, "storage": storage}


def score_candidates(
    goal: str,
    operation: str,
    candidates: list[str],
    args: Any = None,
) -> dict[str, Any]:
    champion = LAB_STATE / "champion" / "model.pt"
    if not champion.exists():
        return {
            "ok": False,
            "available": False,
            "reason": "no_shadow_champion",
            "candidates": [],
            "shadow_only": True,
        }
    scored = []
    for tool in list(candidates or [])[:32]:
        feature = route_feature(goal, operation, tool, args)
        pred = predict_text(LAB_STATE, feature, vocab_size=4096)
        scored.append({
            "tool": tool,
            "success_probability": float(pred["real_probability"]),
            "failure_probability": float(pred["fake_probability"]),
            "confidence": float(pred["confidence"]),
            "label": (
                "uncertain" if pred["label"] == "uncertain"
                else "likely_success" if pred["label"] == "likely_real"
                else "likely_failure"
            ),
        })
    scored.sort(key=lambda row: row["success_probability"], reverse=True)
    return {
        "ok": True,
        "available": True,
        "shadow_only": True,
        "warning": "Advisory only: this ranking is never used to execute or select tools automatically.",
        "candidates": scored,
    }
