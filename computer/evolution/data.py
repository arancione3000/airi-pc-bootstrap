from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[\wÀ-ÿ']+|[^\w\s]", re.UNICODE)


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


def append_verified(path: Path, record: dict) -> dict:
    text = str(record.get("text", "")).strip()
    if len(text) < 8:
        raise ValueError("text is too short")
    label = normalize_label(record.get("label"))
    text_id = text_fingerprint(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_records(path) if path.exists() else []
    same_text = [r for r in existing if r.get("text_id") == text_id or text_fingerprint(r.get("text", "")) == text_id]
    if any(normalize_label(r.get("label")) != label for r in same_text):
        raise ValueError("conflicting verified labels for the same normalized text")
    if any(normalize_label(r.get("label")) == label for r in same_text):
        row = dict(same_text[0])
        row["duplicate"] = True
        return row
    row = {
        "text": text,
        "text_id": text_id,
        "label": label,
        "source": str(record.get("source", "")).strip()[:1000],
        "evidence": str(record.get("evidence", "")).strip()[:4000],
        "added_at": float(record.get("added_at") or time.time()),
    }
    row["id"] = hashlib.sha256((text_id + "\0" + str(label)).encode("utf-8")).hexdigest()[:20]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    existing: set[str] = set()
    if canary_path.exists():
        try:
            raw = json.loads(canary_path.read_text(encoding="utf-8"))
            existing = {str(x) for x in raw.get("ids", [])}
        except Exception:
            existing = set()

    by_label = {0: [], 1: []}
    for row in records:
        by_label[normalize_label(row["label"])].append(row)

    selected = {row.get("text_id") for row in records if row.get("text_id") in existing}
    for label, rows in by_label.items():
        max_canary = max(0, len(rows) - 4)
        target = min(max_canary, max(int(min_per_class), round(len(rows) * float(fraction))))
        have = [row for row in rows if row.get("text_id") in selected]
        need = max(0, target - len(have))
        if need:
            candidates = [row for row in rows if row.get("text_id") not in selected]
            candidates.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row.get('text_id') or text_fingerprint(row.get('text',''))}".encode("utf-8")
                ).hexdigest()
            )
            for row in candidates[:need]:
                selected.add(row.get("text_id") or text_fingerprint(row.get("text", "")))

    canary = [row for row in records if (row.get("text_id") or text_fingerprint(row.get("text", ""))) in selected]
    remaining = [row for row in records if (row.get("text_id") or text_fingerprint(row.get("text", ""))) not in selected]
    payload = {
        "version": 1,
        "ids": sorted(selected),
        "records": len(canary),
        "class_counts": class_counts(canary),
        "source_families": source_family_counts(canary),
    }
    canary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = canary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(canary_path)
    return remaining, canary, payload


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
