from __future__ import annotations

import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .data import append_verified, class_counts, load_records

LIAR_OFFICIAL_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
LIAR_MIRROR_RAW = {
    "train": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv",
    "valid": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv",
    "test": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv",
}
LIAR_LABEL_MAP = {"true": 1, "mostly-true": 1, "false": 0, "pants-fire": 0}
LIAR_AMBIGUOUS = {"half-true", "barely-true"}


def _download(url: str, target: Path, timeout: int = 60, max_bytes: int = 200_000_000) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Airi-PC-NeuroEvolution/1.0"})
    tmp = target.with_suffix(target.suffix + ".tmp")
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("download exceeds configured size limit")
            handle.write(chunk)
    tmp.replace(target)
    return target


def _extract(zip_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"train.tsv": "train", "valid.tsv": "valid", "test.tsv": "test"}
    found: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename).name
            if name not in wanted:
                continue
            target = out_dir / name
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            found[wanted[name]] = target
    if set(found) != {"train", "valid", "test"}:
        raise ValueError("LIAR archive missing expected TSV files")
    return found


def acquire(cache_dir: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    extracted = cache_dir / "liar"
    paths = {name: extracted / f"{name}.tsv" for name in ("train", "valid", "test")}
    if all(path.exists() for path in paths.values()):
        return {"ok": True, "source": "cache", "paths": {k: str(v) for k, v in paths.items()}}
    errors = []
    archive = cache_dir / "liar_dataset.zip"
    try:
        if not archive.exists():
            _download(LIAR_OFFICIAL_URL, archive)
        paths = _extract(archive, extracted)
        return {"ok": True, "source": LIAR_OFFICIAL_URL, "paths": {k: str(v) for k, v in paths.items()}}
    except Exception as exc:
        errors.append({"source": LIAR_OFFICIAL_URL, "error": repr(exc)})
    try:
        extracted.mkdir(parents=True, exist_ok=True)
        for split, url in LIAR_MIRROR_RAW.items():
            _download(url, paths[split], max_bytes=25_000_000)
        return {"ok": True, "source": "github_mirror", "paths": {k: str(v) for k, v in paths.items()}, "errors": errors}
    except Exception as exc:
        errors.append({"source": "github_mirror", "error": repr(exc)})
        return {"ok": False, "errors": errors}


def import_tsv(tsv_path: Path, verified_path: Path, split: str, source_url: str = LIAR_OFFICIAL_URL) -> dict[str, Any]:
    stats = {"split": split, "accepted": 0, "duplicates": 0, "ambiguous": 0, "conflicts": 0, "invalid": 0}
    with Path(tsv_path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3:
                stats["invalid"] += 1
                continue
            item_id, raw_label, statement = row[0].strip(), row[1].strip().lower(), row[2].strip()
            if raw_label in LIAR_AMBIGUOUS:
                stats["ambiguous"] += 1
                continue
            if raw_label not in LIAR_LABEL_MAP or len(statement) < 8:
                stats["invalid"] += 1
                continue
            evidence = {
                "dataset": "LIAR v1.0", "split": split, "item_id": item_id,
                "original_label": raw_label, "research_use_only": True,
                "dataset_source": source_url,
            }
            try:
                added = append_verified(verified_path, {
                    "text": statement, "label": LIAR_LABEL_MAP[raw_label],
                    "source": f"LIAR:{item_id}",
                    "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                })
                stats["duplicates" if added.get("duplicate") else "accepted"] += 1
            except ValueError:
                stats["conflicts"] += 1
    return stats


def bootstrap(state_dir: Path) -> dict[str, Any]:
    state_dir = Path(state_dir)
    acquired = acquire(state_dir / "cache")
    if not acquired.get("ok"):
        return acquired
    verified = state_dir / "data" / "verified.jsonl"
    results = [
        import_tsv(Path(path), verified, split, str(acquired.get("source")))
        for split, path in acquired["paths"].items()
    ]
    rows = load_records(verified)
    return {
        "ok": True,
        "dataset": "LIAR v1.0",
        "license_notice": "Research purposes only; original sources retain copyright.",
        "source": acquired.get("source"),
        "imports": results,
        "verified_records": len(rows),
        "class_counts": class_counts(rows),
        "acquire_errors": acquired.get("errors", []),
    }
