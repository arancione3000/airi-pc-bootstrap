from __future__ import annotations

import hashlib
import os
import json
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[\wÀ-ÿ']+|[^\w\s]", re.UNICODE)


@contextmanager
def _dataset_lock(path: Path, timeout: float = 15.0, stale_after: float = 900.0):
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.time() + max(0.1, float(timeout))
    fd = None
    while fd is None:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode("ascii", errors="ignore"))
        except FileExistsError:
            owner_dead = False
            try:
                raw = lock_path.read_text(encoding="ascii", errors="ignore").strip().split()
                owner_pid = int(raw[0]) if raw and raw[0].isdigit() else None
                if owner_pid and owner_pid != os.getpid():
                    try:
                        os.kill(owner_pid, 0)
                    except PermissionError:
                        pass
                    except OSError:
                        owner_dead = True
                age = time.time() - lock_path.stat().st_mtime
                if owner_dead or age > max(5.0, float(stale_after)):
                    lock_path.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                pass
            if time.time() >= deadline:
                raise TimeoutError(f"timed out waiting for dataset lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            lock_path.unlink(missing_ok=True)


def token_id(token: str, vocab_size: int = 8192) -> int:
    digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
    return 2 + (int.from_bytes(digest, "little") % (vocab_size - 2))


def encode_text(text: str, max_len: int, vocab_size: int = 8192) -> tuple[list[int], list[int]]:
    toks = TOKEN_RE.findall(str(text))[:max_len]
    ids = [token_id(t, vocab_size) for t in toks]
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    if pad > 0:
        ids.extend([0] * pad)
        mask.extend([0] * pad)
    return ids, mask


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "real", "reliable", "vero", "verificato"}:
        return 1
    if s in {"0", "false", "fake", "unreliable", "falso", "bufala"}:
        return 0
    raise ValueError("label must identify verified real=1 or fake=0")


def _canonical_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(_canonical_text(text).encode("utf-8")).hexdigest()[:24]


def _prepare_verified(record: dict) -> dict:
    text = str(record.get("text", "")).strip()
    if len(text) < 8:
        raise ValueError("text is too short")
    label = normalize_label(record.get("label"))
    text_id = text_fingerprint(text)
    row = {
        "text": text,
        "text_id": text_id,
        "label": label,
        "source": str(record.get("source", "")).strip()[:1000],
        "evidence": str(record.get("evidence", "")).strip()[:4000],
        "added_at": float(record.get("added_at") or time.time()),
    }
    row["id"] = hashlib.sha256((text_id + "\0" + str(label)).encode("utf-8")).hexdigest()[:20]
    return row


def append_verified_many(path: Path, records: Iterable[dict]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _dataset_lock(path):
        existing = load_records(path) if path.exists() else []
        known = {
            row.get("text_id") or text_fingerprint(row.get("text", "")): normalize_label(row.get("label"))
            for row in existing
        }
        accepted_rows: list[dict] = []
        results: list[dict] = []
        stats = {"accepted": 0, "duplicates": 0, "conflicts": 0, "invalid": 0}
        for record in records:
            try:
                row = _prepare_verified(record)
            except Exception as exc:
                stats["invalid"] += 1
                results.append({"status": "invalid", "error": str(exc)})
                continue
            previous = known.get(row["text_id"])
            if previous is not None:
                if previous != row["label"]:
                    stats["conflicts"] += 1
                    results.append({"status": "conflict", **row})
                else:
                    stats["duplicates"] += 1
                    results.append({"status": "duplicate", **row})
                continue
            known[row["text_id"]] = row["label"]
            accepted_rows.append(row)
            stats["accepted"] += 1
            results.append({"status": "accepted", **row})
        if accepted_rows:
            with path.open("a", encoding="utf-8") as handle:
                for row in accepted_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {**stats, "rows": results}


def append_verified(path: Path, record: dict) -> dict:
    row = _prepare_verified(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _dataset_lock(path):
        existing = load_records(path) if path.exists() else []
        same_text = [
            item for item in existing
            if (item.get("text_id") or text_fingerprint(item.get("text", ""))) == row["text_id"]
        ]
        if any(normalize_label(item.get("label")) != row["label"] for item in same_text):
            raise ValueError("conflicting verified labels for the same normalized text")
        if same_text:
            duplicate = dict(same_text[0])
            duplicate["duplicate"] = True
            return duplicate
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {**row, "duplicate": False}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row["label"] = normalize_label(row.get("label"))
                text = str(row.get("text", "")).strip()
                if text:
                    row.setdefault("text_id", text_fingerprint(text))
                    out.append(row)
            except Exception:
                continue
    return out


def class_counts(records: Iterable[dict]) -> dict[str, int]:
    counts = {"fake": 0, "real": 0}
    for row in records:
        counts["real" if normalize_label(row["label"]) == 1 else "fake"] += 1
    return counts


def split_records(records: list[dict], seed: int = 1337) -> tuple[list[dict], list[dict], list[dict]]:
    by_label = {0: [], 1: []}
    seen_text: set[str] = set()
    for row in records:
        tid = row.get("text_id") or text_fingerprint(row.get("text", ""))
        if tid in seen_text:
            continue
        seen_text.add(tid)
        by_label[normalize_label(row["label"])].append(row)
    if min(len(by_label[0]), len(by_label[1])) < 4:
        raise ValueError("need at least 4 unique verified samples for each class")
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for label, rows in by_label.items():
        rr = list(rows)
        random.Random(seed + label).shuffle(rr)
        n = len(rr)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
        test.extend(rr[:n_test])
        val.extend(rr[n_test:n_test + n_val])
        train.extend(rr[n_test + n_val:])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(val)
    random.Random(seed + 2).shuffle(test)
    return train, val, test


def source_family(record: dict) -> str:
    source = str(record.get("source", "")).strip().lower()
    if source.startswith("liar:"):
        return "liar"
    if source.startswith("claimreview consensus:"):
        return "claimreview"
    if source.startswith("manual:"):
        return "manual"
    return "other" if source else "unknown"


def source_family_counts(records: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        family = source_family(row)
        counts[family] = counts.get(family, 0) + 1
    return counts


def ensure_canary_partition(
    state_dir: Path,
    records: list[dict],
    *,
    seed: int = 1337,
    fraction: float = 0.05,
    min_per_class: int = 2,
) -> tuple[list[dict], list[dict], dict]:
    state_dir = Path(state_dir)
    canary_path = state_dir / "data" / "canary_ids.json"
    existed_before = canary_path.exists()
    existing: set[str] = set()
    if existed_before:
        try:
            raw = json.loads(canary_path.read_text(encoding="utf-8"))
            ids = raw.get("ids")
            if not isinstance(ids, list):
                raise ValueError("invalid canary manifest")
            existing = {str(x) for x in ids}
        except Exception:
            existed_before = False
            existing = set()

    by_label = {0: [], 1: []}
    for row in records:
        by_label[normalize_label(row["label"])].append(row)

    selected = {row.get("text_id") for row in records if row.get("text_id") in existing}
    if not existed_before:
        for label, rows in by_label.items():
            max_canary = max(0, len(rows) - 4)
            target = min(max_canary, max(int(min_per_class), round(len(rows) * float(fraction))))
            candidates = list(rows)
            candidates.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row.get('text_id') or text_fingerprint(row.get('text',''))}".encode("utf-8")
                ).hexdigest()
            )
            for row in candidates[:target]:
                selected.add(row.get("text_id") or text_fingerprint(row.get("text", "")))

    canary = [row for row in records if (row.get("text_id") or text_fingerprint(row.get("text", ""))) in selected]
    remaining = [row for row in records if (row.get("text_id") or text_fingerprint(row.get("text", ""))) not in selected]
    payload = {
        "version": 1,
        "ids": sorted(selected),
        "records": len(canary),
        "class_counts": class_counts(canary),
        "source_families": source_family_counts(canary),
        "created_now": not existed_before,
    }
    canary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = canary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(canary_path)
    return remaining, canary, payload


def persistent_split_records(
    state_dir: Path,
    records: list[dict],
    seed: int = 1337,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    state_dir = Path(state_dir)
    manifest_path = state_dir / "data" / "split_manifest.json"
    assignments: dict[str, str] = {}
    existed_before = manifest_path.exists()
    if existed_before:
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_assignments = raw.get("assignments")
            if not isinstance(raw_assignments, dict):
                raise ValueError("invalid split manifest")
            assignments = {
                str(k): str(v)
                for k, v in raw_assignments.items()
                if str(v) in {"train", "val", "test"}
            }
            if not assignments:
                raise ValueError("empty split manifest")
        except Exception:
            existed_before = False
            assignments = {}

    if not assignments:
        train0, val0, test0 = split_records(records, seed)
        for name, rows in (("train", train0), ("val", val0), ("test", test0)):
            for row in rows:
                tid = row.get("text_id") or text_fingerprint(row.get("text", ""))
                assignments[tid] = name

    for row in records:
        tid = row.get("text_id") or text_fingerprint(row.get("text", ""))
        if tid in assignments:
            continue
        label = normalize_label(row.get("label"))
        digest = hashlib.sha256(f"split:{seed}:{label}:{tid}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 100
        assignments[tid] = "test" if bucket < 15 else ("val" if bucket < 30 else "train")

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    current_ids = set()
    for row in records:
        tid = row.get("text_id") or text_fingerprint(row.get("text", ""))
        current_ids.add(tid)
        target = assignments.get(tid, "train")
        if target == "test":
            test.append(row)
        elif target == "val":
            val.append(row)
        else:
            train.append(row)

    payload = {
        "version": 1,
        "created_now": not existed_before,
        "seed": seed,
        "assignments": {tid: assignments[tid] for tid in sorted(current_ids)},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts": {
            "train": class_counts(train),
            "val": class_counts(val),
            "test": class_counts(test),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest_path)
    return train, val, test, payload


@dataclass
class DatasetView:
    records: list[dict]
    max_len: int
    vocab_size: int = 8192

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        import torch
        row = self.records[idx]
        ids, mask = encode_text(row["text"], self.max_len, self.vocab_size)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(normalize_label(row["label"]), dtype=torch.long),
        )
