from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.bpe_tokenizer import BPETokenizer, train_bpe
from generalist_lm.tokenizer import ASSISTANT, BOS, ByteTokenizer


def training_texts():
    code = (
        "def add(a, b):\n    return a + b\n\n"
        "def subtract(a, b):\n    return a - b\n\n"
        "class Calculator:\n    def compute(self, value):\n        return value + 1\n"
    )
    prose = (
        "AIRI generalist language model learns code and documentation. "
        "The generalist language model should understand repeated project vocabulary. "
    )
    return [code * 30, prose * 40, "Ciao 🌸 hello world! " * 40]


def test_bpe_roundtrip_preserves_arbitrary_utf8():
    tokenizer = train_bpe(training_texts(), vocab_size=400, min_frequency=2)
    text = "Ciao 🌸 — Ελληνικά — 日本語 — code: return a + b"
    encoded = tokenizer.encode(text, bos=True, eos=True)
    assert encoded[0] == BOS
    assert tokenizer.decode(encoded) == text


def test_bpe_training_is_deterministic():
    first = train_bpe(training_texts(), vocab_size=420, min_frequency=2)
    second = train_bpe(training_texts(), vocab_size=420, min_frequency=2)
    assert first.merges == second.merges
    assert first.digest == second.digest
    assert first.vocab_size == second.vocab_size


def test_bpe_compresses_repeated_code_better_than_bytes():
    text = (
        "def calculate_total(values):\n"
        "    total = sum(values)\n"
        "    return total\n"
    ) * 80
    byte = ByteTokenizer()
    bpe = train_bpe([text], vocab_size=512, min_frequency=2)
    byte_tokens = len(byte.encode(text))
    bpe_tokens = len(bpe.encode(text))
    assert bpe_tokens < byte_tokens
    assert bpe_tokens <= int(byte_tokens * 0.65)


def test_bpe_checkpoint_roundtrip_and_digest_tamper_detection(tmp_path: Path):
    tokenizer = train_bpe(training_texts(), vocab_size=384, min_frequency=2)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.merges == tokenizer.merges
    assert restored.digest == tokenizer.digest
    assert restored.decode(restored.encode("hello 🌸")) == "hello 🌸"

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["digest"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        BPETokenizer.load(path)


def test_bpe_chat_serialization_preserves_role_tokens():
    tokenizer = train_bpe(["hello assistant user system " * 30], vocab_size=320)
    ids = tokenizer.serialize_messages([
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "hello"},
    ])
    assert ids[0] == BOS
    assert ids[-1] == ASSISTANT
    with pytest.raises(ValueError):
        tokenizer.serialize_messages([{"role": "root", "content": "x"}])


def test_bpe_rejects_invalid_future_reference_and_special_merge():
    with pytest.raises(ValueError, match="special tokens"):
        BPETokenizer(((1, 8),))
    with pytest.raises(ValueError, match="not yet defined"):
        BPETokenizer(((264, 8),))
