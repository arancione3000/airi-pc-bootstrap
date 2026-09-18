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
    # IDs 0 and 1 are reserved for PAD and UNK/special use.
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


def append_verified(path: Path, record: dict) -> dict:
    text = str(record.get("text", "")).strip()
    if len(text) < 8:
        raise ValueError("text is too short")
    row = {
        "text": text,
        "label": normalize_label(record.get("label")),
        "source": str(record.get("source", "")).strip()[:1000],
        "evidence": str(record.get("evidence", "")).strip()[:4000],
        "added_at": float(record.get("added_at") or time.time()),
    }
    # Stable duplicate protection on the normalized text + label.
    row["id"] = hashlib.sha256((text.strip().lower() + "\0" + str(row["label"])).encode("utf-8")).hexdigest()[:20]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r.get("id") for r in load_records(path)} if path.exists() else set()
    if row["id"] in existing:
        return {**row, "duplicate": True}
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
                if str(row.get("text", "")).strip():
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
    """Deterministic class-stratified 70/15/15-ish split.

    Every class needs at least 4 examples. Tiny datasets are intentionally
    rejected instead of producing misleading fitness numbers.
    """
    by_label = {0: [], 1: []}
    for row in records:
        by_label[normalize_label(row["label"])].append(row)
    if min(len(by_label[0]), len(by_label[1])) < 4:
        raise ValueError("need at least 4 verified samples for each class")
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
