from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .data import class_counts, load_records, text_fingerprint


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def audit_state(state_dir: Path) -> dict[str, Any]:
    state_dir = Path(state_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    data_path = state_dir / "data" / "verified.jsonl"
    malformed_lines = 0
    raw_lines = 0
    if data_path.exists():
        with data_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_lines += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not str(row.get("text", "")).strip():
                        malformed_lines += 1
                except Exception:
                    malformed_lines += 1
    records = load_records(data_path)
    checks["dataset_raw_lines"] = raw_lines
    checks["dataset_malformed_lines"] = malformed_lines
    checks["dataset_records"] = len(records)
    if malformed_lines:
        errors.append(f"verified dataset contains {malformed_lines} malformed JSONL rows")
    checks["class_counts"] = class_counts(records)

    labels_by_text: dict[str, set[int]] = {}
    for row in records:
        tid = row.get("text_id") or text_fingerprint(row.get("text", ""))
        labels_by_text.setdefault(tid, set()).add(int(row["label"]))
    conflicts = sorted(tid for tid, labels in labels_by_text.items() if len(labels) > 1)
    duplicate_count = max(0, len(records) - len(labels_by_text))
    checks["duplicate_rows"] = duplicate_count
    checks["conflicting_text_ids"] = conflicts
    if conflicts:
        errors.append(f"dataset contains {len(conflicts)} normalized texts with conflicting labels")
    if duplicate_count:
        warnings.append(f"dataset contains {duplicate_count} duplicate normalized rows")

    canary = _read_json(state_dir / "data" / "canary_ids.json", {}) or {}
    canary_ids = set(canary.get("ids") or [])
    split = _read_json(state_dir / "data" / "split_manifest.json", {}) or {}
    assignments = split.get("assignments") or {}
    split_ids = set(assignments)
    overlap = sorted(canary_ids & split_ids)
    checks["canary_records"] = len(canary_ids)
    checks["split_records"] = len(split_ids)
    checks["canary_split_overlap"] = overlap
    if overlap:
        errors.append(f"{len(overlap)} golden-canary records also appear in train/val/test manifest")

    dataset_ids = set(labels_by_text)
    non_canary = dataset_ids - canary_ids
    missing_assignments = sorted(non_canary - split_ids)
    stale_assignments = sorted(split_ids - non_canary)
    checks["missing_split_assignments"] = len(missing_assignments)
    checks["stale_split_assignments"] = len(stale_assignments)
    if split_ids and missing_assignments:
        warnings.append(f"{len(missing_assignments)} non-canary records are not yet assigned to a persistent split")
    if stale_assignments:
        warnings.append(f"{len(stale_assignments)} split assignments no longer correspond to current non-canary records")

    champion_dir = state_dir / "champion"
    champion_files = {
        name: (champion_dir / name).exists()
        for name in ("genome.json", "model.pt", "metrics.json", "provenance.json")
    }
    checks["champion_files"] = champion_files
    present = sum(champion_files.values())
    if 0 < present < len(champion_files):
        errors.append("champion directory is incomplete")
    provenance = _read_json(champion_dir / "provenance.json", {}) or {}
    genome = _read_json(champion_dir / "genome.json", {}) or {}
    if provenance:
        trained = int(provenance.get("dataset_records", 0) or 0)
        checks["champion_dataset_records"] = trained
        if trained > len(records):
            errors.append("champion provenance references more dataset records than currently exist")

    edge_dir = state_dir / "edge"
    edge_meta = _read_json(edge_dir / "metadata.json", {}) or {}
    edge_model = edge_dir / "model-int8.pt"
    checks["edge_available"] = bool(edge_meta and edge_model.exists())
    if edge_meta and not edge_model.exists():
        errors.append("edge metadata exists without model-int8.pt")
    if edge_model.exists() and not edge_meta:
        errors.append("model-int8.pt exists without edge metadata")
    if edge_meta and genome and edge_meta.get("genome_id") != genome.get("genome_id"):
        errors.append("INT8 edge artifact belongs to a different champion genome")

    lock_path = data_path.with_suffix(data_path.suffix + ".lock")
    if lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = 0.0
        checks["dataset_lock_age_seconds"] = age
        if age > 120:
            warnings.append("verified dataset has a stale write lock")
        else:
            warnings.append("verified dataset is currently write-locked")
    else:
        checks["dataset_lock_age_seconds"] = None

    queue_dir = state_dir / "queue"
    queue_counts: dict[str, int] = {}
    invalid_queue = 0
    if queue_dir.exists():
        for path in queue_dir.glob("*.json"):
            row = _read_json(path, None)
            if not isinstance(row, dict):
                invalid_queue += 1
                continue
            status = str(row.get("status", "unknown"))
            queue_counts[status] = queue_counts.get(status, 0) + 1
    checks["queue_status"] = queue_counts
    checks["invalid_queue_files"] = invalid_queue
    if invalid_queue:
        warnings.append(f"{invalid_queue} queue files are invalid JSON")

    runs_dir = state_dir / "runs"
    heavy_runs = []
    if runs_dir.exists():
        for run in runs_dir.iterdir():
            if not run.is_dir():
                continue
            trial_files = list(run.glob("promotion-trial-*.pt"))
            if trial_files:
                heavy_runs.append({"run": run.name, "loser_trial_files": len(trial_files)})
    checks["runs_with_unpruned_trial_weights"] = heavy_runs
    if heavy_runs:
        warnings.append(f"{len(heavy_runs)} runs still contain redundant promotion-trial weights")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
