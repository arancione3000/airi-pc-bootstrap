from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .curriculum import ResearchRow, validation_rows


SCHEMA_VERSION = 1
DEFAULT_MAX_ROWS = 1200


def _row_id(row: ResearchRow) -> str:
    raw = json.dumps(
        {"domain": row.domain, "messages": row.messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _prompt_set(rows: list[ResearchRow]) -> set[str]:
    return {
        str(row.messages[0].get("content", ""))
        for row in rows
        if row.messages
    }


def _mechanical_rows(cycle: int, signals: list[str]) -> list[ResearchRow]:
    """Generate exact-label training rows from deterministic rules.

    These examples never depend on model output or web text for their labels.
    Numeric ranges are intentionally far from the fixed validation curriculum.
    """
    c = max(1, int(cycle))
    base = 1000 + c * 17
    word = f"cycleword{c}"
    rows = [
        ResearchRow("language", [
            {"role": "user", "content": f"Reply with exactly {word.upper()} and nothing else."},
            {"role": "assistant", "content": word.upper()},
        ]),
        ResearchRow("coding", [
            {"role": "user", "content": f"Return only Python code defining offset_{c}(x) that returns x + {c + 10}."},
            {"role": "assistant", "content": f"def offset_{c}(x):\n    return x + {c + 10}"},
        ]),
        ResearchRow("data", [
            {"role": "user", "content": f"Mean of {base},{base + 6},{base + 12}? Number only."},
            {"role": "assistant", "content": str(base + 6)},
        ]),
        ResearchRow("reasoning", [
            {"role": "user", "content": f"What is {c + 40}*{c + 7}? Number only."},
            {"role": "assistant", "content": str((c + 40) * (c + 7))},
        ]),
        ResearchRow("tools", [
            {"role": "user", "content": f"Use calculator for {base}+{c + 13}."},
            {"role": "assistant", "content": f'<tool_call>{{"name":"calculator","arguments":{{"expression":"{base}+{c + 13}"}}}}</tool_call>'},
        ]),
        ResearchRow("structured", [
            {"role": "user", "content": f'Return exactly this JSON object: {{"cycle":{c},"ok":true}}'},
            {"role": "assistant", "content": f'{{"cycle":{c},"ok":true}}'},
        ]),
    ]

    signal_set = set(signals)
    if "coding_gap" in signal_set:
        rows.append(ResearchRow("coding", [
            {"role": "user", "content": f"Return only Python code defining scale_{c}(x) that returns x * {c % 7 + 2}."},
            {"role": "assistant", "content": f"def scale_{c}(x):\n    return x * {c % 7 + 2}"},
        ]))
    if "data_gap" in signal_set:
        values = [base + 1, base + 3, base + 5, base + 7]
        rows.append(ResearchRow("data", [
            {"role": "user", "content": f"Sum of {','.join(map(str, values))}? Number only."},
            {"role": "assistant", "content": str(sum(values))},
        ]))
    if "reasoning_gap" in signal_set or "symbolic_reasoning_signal" in signal_set:
        rows.append(ResearchRow("reasoning", [
            {"role": "user", "content": f"Sequence {c},{c + 3},{c + 6},{c + 9}. Next term only."},
            {"role": "assistant", "content": str(c + 12)},
        ]))
    if "tool_gap" in signal_set:
        rows.append(ResearchRow("tools", [
            {"role": "user", "content": f"Use calculator for {base}-{c + 5}."},
            {"role": "assistant", "content": f'<tool_call>{{"name":"calculator","arguments":{{"expression":"{base}-{c + 5}"}}}}</tool_call>'},
        ]))
    return rows


class CurriculumMemory:
    def __init__(self, state_dir: str | Path, *, max_rows: int = DEFAULT_MAX_ROWS):
        self.root = Path(state_dir)
        self.path = self.root / "curriculum-memory.json"
        self.max_rows = max(60, min(int(max_rows), 20_000))

    def _load_raw(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("curriculum memory must be an object")
            if int(value.get("version", 0)) != SCHEMA_VERSION:
                raise ValueError("unsupported curriculum memory version")
            rows = value.get("rows")
            if not isinstance(rows, list):
                raise ValueError("curriculum rows must be a list")
            return value
        except FileNotFoundError:
            return {"version": SCHEMA_VERSION, "last_cycle": 0, "rows": []}

    @staticmethod
    def _decode(row: dict[str, Any]) -> ResearchRow | None:
        try:
            domain = str(row["domain"])
            messages = row["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                return None
            clean = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in messages
                if isinstance(item, dict) and "role" in item and "content" in item
            ]
            if len(clean) != len(messages):
                return None
            return ResearchRow(domain, clean)
        except Exception:
            return None

    def rows(self) -> list[ResearchRow]:
        raw = self._load_raw()
        out: list[ResearchRow] = []
        seen: set[str] = set()
        protected = _prompt_set(validation_rows())
        for item in raw.get("rows", []):
            if not isinstance(item, dict):
                continue
            row = self._decode(item)
            if row is None or not row.messages:
                continue
            if row.messages[0]["content"] in protected:
                continue
            rid = _row_id(row)
            if rid in seen:
                continue
            seen.add(rid)
            out.append(row)
        return out

    def expand(self, cycle: int, *, signals: list[str] | None = None) -> dict[str, Any]:
        raw = self._load_raw()
        current = self.rows()
        existing = {_row_id(row) for row in current}
        protected = _prompt_set(validation_rows())
        added: list[ResearchRow] = []

        for row in _mechanical_rows(cycle, list(signals or [])):
            if row.messages[0]["content"] in protected:
                continue
            rid = _row_id(row)
            if rid in existing:
                continue
            existing.add(rid)
            current.append(row)
            added.append(row)

        if len(current) > self.max_rows:
            # Retain a bounded replay buffer while keeping all domains represented
            # by using the most recent rows after deterministic generation.
            current = current[-self.max_rows :]

        payload = {
            "version": SCHEMA_VERSION,
            "last_cycle": max(int(raw.get("last_cycle", 0) or 0), int(cycle)),
            "rows": [
                {"id": _row_id(row), "domain": row.domain, "messages": row.messages}
                for row in current
            ],
        }
        _atomic_json(self.path, payload)
        counts = Counter(row.domain for row in current)
        return {
            "ok": True,
            "cycle": int(cycle),
            "added": len(added),
            "stored": len(current),
            "domains": dict(sorted(counts.items())),
            "max_rows": self.max_rows,
            "validation_overlap": sorted(_prompt_set(current) & protected),
        }

    def manifest(self) -> dict[str, Any]:
        rows = self.rows()
        counts = Counter(row.domain for row in rows)
        return {
            "version": SCHEMA_VERSION,
            "stored": len(rows),
            "domains": dict(sorted(counts.items())),
            "max_rows": self.max_rows,
            "validation_overlap": sorted(_prompt_set(rows) & _prompt_set(validation_rows())),
        }
