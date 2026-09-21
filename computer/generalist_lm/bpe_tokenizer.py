from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from .tokenizer import (
    ASSISTANT,
    BOS,
    BYTE_OFFSET,
    EOS,
    ROLE_TOKENS,
    SEP,
    VOCAB_SIZE as BYTE_VOCAB_SIZE,
)

BPE_VERSION = "bpe-v1"
MAX_MERGES = 65_536


def _replace_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    if len(tokens) < 2:
        return tokens
    left, right = pair
    out: list[int] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def _canonical_merges(merges: Sequence[tuple[int, int]]) -> bytes:
    return json.dumps(
        {
            "version": BPE_VERSION,
            "merges": [[int(a), int(b)] for a, b in merges],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def merges_digest(merges: Sequence[tuple[int, int]]) -> str:
    return hashlib.sha256(_canonical_merges(merges)).hexdigest()


@dataclass
class BPETokenizer:
    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.merges) > MAX_MERGES:
            raise ValueError("too many BPE merges")

        token_bytes: dict[int, bytes] = {
            BYTE_OFFSET + value: bytes([value])
            for value in range(256)
        }
        merge_map: dict[tuple[int, int], int] = {}
        next_id = BYTE_VOCAB_SIZE

        for pair in self.merges:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("each BPE merge must be a pair")
            left, right = int(pair[0]), int(pair[1])
            if left < BYTE_OFFSET or right < BYTE_OFFSET:
                raise ValueError("BPE merges cannot reference special tokens")
            if left >= next_id or right >= next_id:
                raise ValueError("BPE merge references a token not yet defined")
            if (left, right) in merge_map:
                raise ValueError("duplicate BPE merge pair")
            token_bytes[next_id] = token_bytes[left] + token_bytes[right]
            merge_map[(left, right)] = next_id
            next_id += 1

        self._token_bytes = token_bytes
        self._merge_map = merge_map
        self.version = BPE_VERSION
        self.vocab_size = BYTE_VOCAB_SIZE + len(self.merges)
        self.digest = merges_digest(self.merges)

    def token_bytes(self, token_id: int) -> bytes:
        token = int(token_id)
        if token not in self._token_bytes:
            raise ValueError(f"token has no byte representation: {token}")
        return bytes(self._token_bytes[token])

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = [BYTE_OFFSET + byte for byte in str(text).encode("utf-8", errors="replace")]
        next_id = BYTE_VOCAB_SIZE
        for pair in self.merges:
            tokens = _replace_pair(tokens, pair, next_id)
            next_id += 1
        if bos:
            tokens.insert(0, BOS)
        if eos:
            tokens.append(EOS)
        return tokens

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        data = bytearray()
        for raw in ids:
            token = int(raw)
            if token in self._token_bytes:
                data.extend(self._token_bytes[token])
            elif token < BYTE_OFFSET:
                if not skip_special:
                    data.extend(f"<{token}>".encode("ascii"))
            else:
                raise ValueError(f"unknown BPE token id: {token}")
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

    def to_dict(self) -> dict:
        return {
            "version": BPE_VERSION,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": self.vocab_size,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "BPETokenizer":
        if not isinstance(raw, dict):
            raise ValueError("BPE tokenizer payload must be an object")
        if raw.get("version") != BPE_VERSION:
            raise ValueError("unsupported BPE tokenizer version")
        merges_raw = raw.get("merges")
        if not isinstance(merges_raw, list):
            raise ValueError("BPE tokenizer merges must be a list")
        merges: list[tuple[int, int]] = []
        for row in merges_raw:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("invalid BPE merge row")
            merges.append((int(row[0]), int(row[1])))
        tokenizer = cls(tuple(merges))
        expected = str(raw.get("digest", ""))
        if expected and expected != tokenizer.digest:
            raise ValueError("BPE tokenizer digest mismatch")
        declared_vocab = raw.get("vocab_size")
        if declared_vocab is not None and int(declared_vocab) != tokenizer.vocab_size:
            raise ValueError("BPE tokenizer vocabulary size mismatch")
        return tokenizer

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = 1024,
    min_frequency: int = 2,
    max_bytes: int = 5_000_000,
) -> BPETokenizer:
    """Train deterministic byte-pair merges on bounded local text.

    This trainer is deliberately simple and reproducible. It starts from the
    byte vocabulary, never creates unknown-token behavior, and stops early when
    no pair reaches the configured frequency.
    """
    target_vocab = max(BYTE_VOCAB_SIZE, min(int(vocab_size), BYTE_VOCAB_SIZE + MAX_MERGES))
    min_frequency = max(2, int(min_frequency))
    max_bytes = max(1, int(max_bytes))

    sequences: list[list[int]] = []
    consumed = 0
    for text in texts:
        raw = str(text).encode("utf-8", errors="replace")
        if not raw:
            continue
        if consumed >= max_bytes:
            break
        raw = raw[: max(0, max_bytes - consumed)]
        consumed += len(raw)
        if raw:
            sequences.append([BYTE_OFFSET + byte for byte in raw])

    merges: list[tuple[int, int]] = []
    next_id = BYTE_VOCAB_SIZE

    while next_id < target_vocab:
        counts: dict[tuple[int, int], int] = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break

        best_pair, best_count = min(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        if best_count < min_frequency:
            break

        merges.append(best_pair)
        sequences = [_replace_pair(seq, best_pair, next_id) for seq in sequences]
        next_id += 1

    return BPETokenizer(tuple(merges))


def extend_bpe(
    source: BPETokenizer,
    texts: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int = 2,
    max_bytes: int = 256_000,
) -> BPETokenizer:
    """Extend an existing BPE vocabulary without renumbering learned tokens.

    Existing merges stay as an immutable prefix. Only additional merge rules
    are learned, so old token ids and their embeddings remain reusable.
    """
    if not isinstance(source, BPETokenizer):
        raise TypeError("source must be a BPETokenizer")
    target_vocab = max(
        int(source.vocab_size),
        min(int(vocab_size), BYTE_VOCAB_SIZE + MAX_MERGES),
    )
    if target_vocab <= int(source.vocab_size):
        return source

    min_frequency = max(2, int(min_frequency))
    max_bytes = max(1, int(max_bytes))
    sequences: list[list[int]] = []
    consumed = 0

    for text in texts:
        raw = str(text).encode("utf-8", errors="replace")
        if not raw:
            continue
        if consumed >= max_bytes:
            break
        raw = raw[: max(0, max_bytes - consumed)]
        consumed += len(raw)
        if not raw:
            continue
        seq = [BYTE_OFFSET + byte for byte in raw]
        next_id = BYTE_VOCAB_SIZE
        for pair in source.merges:
            seq = _replace_pair(seq, pair, next_id)
            next_id += 1
        sequences.append(seq)

    merges = list(source.merges)
    next_id = BYTE_VOCAB_SIZE + len(merges)
    while next_id < target_vocab:
        counts: dict[tuple[int, int], int] = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best_pair, best_count = min(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        if best_count < min_frequency:
            break
        merges.append(best_pair)
        sequences = [_replace_pair(seq, best_pair, next_id) for seq in sequences]
        next_id += 1

    return BPETokenizer(tuple(merges))
