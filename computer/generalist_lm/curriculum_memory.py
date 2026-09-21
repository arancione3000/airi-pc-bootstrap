from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .curriculum import DOMAINS, ResearchRow, validation_rows


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


def canary_rows(cycle: int) -> list[ResearchRow]:
    """Return a rotating held-out canary never added to training replay."""
    c = max(1, int(cycle))
    base = 50_000 + c * 131
    token = f"CANARY{c:04d}"
    return [
        ResearchRow("language", [
            {"role": "user", "content": f"Reply with exactly {token} and nothing else."},
            {"role": "assistant", "content": token},
        ]),
        ResearchRow("coding", [
            {"role": "user", "content": f"Return only Python code defining canary_sub_{c}(x) that returns x - {c % 11 + 3}."},
            {"role": "assistant", "content": f"def canary_sub_{c}(x):\n    return x - {c % 11 + 3}"},
        ]),
        ResearchRow("data", [
            {"role": "user", "content": f"Sum of {base},{base + 2},{base + 5}? Number only."},
            {"role": "assistant", "content": str(base * 3 + 7)},
        ]),
        ResearchRow("reasoning", [
            {"role": "user", "content": f"Sequence {base},{base + 4},{base + 8},{base + 12}. Next term only."},
            {"role": "assistant", "content": str(base + 16)},
        ]),
        ResearchRow("tools", [
            {"role": "user", "content": f"Use calculator for {base}*{c % 5 + 2}."},
            {"role": "assistant", "content": f'<tool_call>{{"name":"calculator","arguments":{{"expression":"{base}*{c % 5 + 2}"}}}}</tool_call>'},
        ]),
        ResearchRow("structured", [
            {"role": "user", "content": f'Return exactly this JSON object: {{"canary":{c},"value":{base}}}'},
            {"role": "assistant", "content": f'{{"canary":{c},"value":{base}}}'},
        ]),
    ]


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
    if "language_gap" in signal_set:
        subject = f"agent{c}"
        object_ = f"report{c}"
        rows.append(ResearchRow("language", [
            {
                "role": "user",
                "content": (
                    "Write one grammatical sentence using exactly these content "
                    f"words: subject={subject}; verb=checks; object={object_}."
                ),
            },
            {"role": "assistant", "content": f"{subject.capitalize()} checks {object_}."},
        ]))
        rows.append(ResearchRow("language", [
            {
                "role": "user",
                "content": (
                    "Write one grammatical sentence using exactly these content "
                    f"words: subject=system{c}; verb=stores; object=result{c}."
                ),
            },
            {"role": "assistant", "content": f"System{c} stores result{c}."},
        ]))
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
        except FileNotFoundError:
            return {"version": SCHEMA_VERSION, "last_cycle": 0, "rows": []}
        if not isinstance(value, dict):
            raise ValueError("curriculum memory must be an object")
        if int(value.get("version", 0)) != SCHEMA_VERSION:
            raise ValueError("unsupported curriculum memory version")
        last_cycle = value.get("last_cycle", 0)
        if not isinstance(last_cycle, int) or last_cycle < 0:
            raise ValueError("curriculum last_cycle must be a non-negative integer")
        rows = value.get("rows")
        if not isinstance(rows, list):
            raise ValueError("curriculum rows must be a list")
        if len(rows) > self.max_rows:
            raise ValueError("curriculum memory exceeds configured replay cap")
        return value

    @staticmethod
    def _decode(item: dict[str, Any]) -> ResearchRow:
        if not isinstance(item, dict):
            raise ValueError("curriculum row must be an object")
        row_id = item.get("id")
        domain = item.get("domain")
        messages = item.get("messages")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("curriculum row requires a digest id")
        if domain not in DOMAINS:
            raise ValueError(f"unsupported curriculum domain: {domain!r}")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("curriculum row requires at least two messages")
        clean: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("curriculum message must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"unsupported curriculum role: {role!r}")
            if not isinstance(content, str):
                raise ValueError("curriculum message content must be a string")
            clean.append({"role": str(role), "content": content})
        row = ResearchRow(str(domain), clean)
        row.sft()  # validates the final assistant supervision target
        expected = _row_id(row)
        if row_id != expected:
            raise ValueError("curriculum row digest mismatch")
        return row

    def rows(self) -> list[ResearchRow]:
        raw = self._load_raw()
        out: list[ResearchRow] = []
        seen: set[str] = set()
        protected = _prompt_set(validation_rows())
        for item in raw.get("rows", []):
            row = self._decode(item)
            rid = _row_id(row)
            if rid in seen:
                raise ValueError("duplicate curriculum replay row")
            seen.add(rid)
            if not row.messages or row.messages[0]["content"] in protected:
                raise ValueError("curriculum replay overlaps protected validation")
            out.append(row)
        return out

    def expand(
        self,
        cycle: int,
        *,
        signals: list[str] | None = None,
        extra_rows: list[ResearchRow] | None = None,
    ) -> dict[str, Any]:
        raw = self._load_raw()
        current = self.rows()
        existing = {_row_id(row) for row in current}
        protected = _prompt_set(validation_rows())
        added: list[ResearchRow] = []

        generated_rows = [
            *_mechanical_rows(cycle, list(signals or [])),
            *list(extra_rows or []),
        ]
        for row in generated_rows:
            if row.domain not in DOMAINS:
                raise ValueError(f"unsupported extra curriculum domain: {row.domain!r}")
            row.sft()
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
            "extra_rows_considered": len(list(extra_rows or [])),
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
