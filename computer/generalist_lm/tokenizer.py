from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PAD = 0
BOS = 1
EOS = 2
SYSTEM = 3
USER = 4
ASSISTANT = 5
TOOL = 6
SEP = 7
BYTE_OFFSET = 8
VOCAB_SIZE = BYTE_OFFSET + 256

ROLE_TOKENS = {
    "system": SYSTEM,
    "user": USER,
    "assistant": ASSISTANT,
    "tool": TOOL,
}


@dataclass(frozen=True)
class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer with explicit chat-role tokens.

    It is intentionally simple and dependency-free. A future learned tokenizer
    may replace it only through the versioned tokenizer interface and benchmark
    gates; stored checkpoints always declare the tokenizer version they use.
    """

    version: str = "byte-v1"
    vocab_size: int = VOCAB_SIZE

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        raw = str(text).encode("utf-8", errors="replace")
        ids = [BYTE_OFFSET + b for b in raw]
        if bos:
            ids.insert(0, BOS)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        data = bytearray()
        for value in ids:
            token = int(value)
            if BYTE_OFFSET <= token < VOCAB_SIZE:
                data.append(token - BYTE_OFFSET)
            elif not skip_special:
                data.extend(f"<{token}>".encode("ascii"))
        return bytes(data).decode("utf-8", errors="replace")

    def serialize_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        bos: bool = True,
    ) -> list[int]:
        out: list[int] = [BOS] if bos else []
        for row in messages:
            role = str(row.get("role", "")).strip().lower()
            if role not in ROLE_TOKENS:
                raise ValueError(f"unsupported chat role: {role!r}")
            out.append(ROLE_TOKENS[role])
            out.extend(self.encode(str(row.get("content", ""))))
            out.append(SEP)
        if add_generation_prompt:
            out.append(ASSISTANT)
        return out
