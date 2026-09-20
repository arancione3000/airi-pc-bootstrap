from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.bpe_tokenizer import BPETokenizer, train_bpe
from generalist_lm.model import GeneralistLMConfig, parameter_count
from generalist_lm.qualification import QUALIFICATION_VERSION, checkpoint_digest, qualification_status
from generalist_lm.runtime import GeneralistRuntime
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



def test_bpe_generalist_checkpoint_roundtrip(tmp_path: Path):
    torch = pytest.importorskip("torch")
    tokenizer = train_bpe(training_texts(), vocab_size=336, min_frequency=2)
    config = GeneralistLMConfig(
        vocab_size=tokenizer.vocab_size,
        tokenizer_version="bpe-v1",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()
    runtime = GeneralistRuntime.fresh(config, tokenizer=tokenizer)
    metadata = runtime.save_checkpoint(tmp_path, metadata={"test": "bpe"})
    assert (tmp_path / "tokenizer.json").exists()
    assert metadata["tokenizer_version"] == "bpe-v1"
    assert metadata["tokenizer_digest"] == tokenizer.digest

    restored = GeneralistRuntime.from_checkpoint(tmp_path)
    assert restored.tokenizer.version == "bpe-v1"
    assert restored.tokenizer.digest == tokenizer.digest
    assert restored.tokenizer.decode(restored.tokenizer.encode("hello 🌸")) == "hello 🌸"
    assert parameter_count(restored.model) == parameter_count(runtime.model)
    assert all(
        torch.equal(left.detach().cpu(), right.detach().cpu())
        for left, right in zip(runtime.model.parameters(), restored.model.parameters())
    )


def test_bpe_checkpoint_requires_tokenizer_artifact(tmp_path: Path):
    pytest.importorskip("torch")
    tokenizer = train_bpe(training_texts(), vocab_size=320)
    config = GeneralistLMConfig(
        vocab_size=tokenizer.vocab_size,
        tokenizer_version="bpe-v1",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
    ).validate()
    GeneralistRuntime.fresh(config, tokenizer=tokenizer).save_checkpoint(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        GeneralistRuntime.from_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        checkpoint_digest(tmp_path)


def test_bpe_tokenizer_is_bound_to_qualification_digest(tmp_path: Path):
    pytest.importorskip("torch")
    tokenizer = train_bpe(training_texts(), vocab_size=320)
    config = GeneralistLMConfig(
        vocab_size=tokenizer.vocab_size,
        tokenizer_version="bpe-v1",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
    ).validate()
    GeneralistRuntime.fresh(config, tokenizer=tokenizer).save_checkpoint(tmp_path)
    digest = checkpoint_digest(tmp_path)
    (tmp_path / "benchmark.json").write_text(json.dumps({
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-qualification-v2",
        "checkpoint_digest": digest,
        "qualified": True,
        "minimum_score": 85.0,
        "report": {"ok": True, "score": 100.0, "critical_failures": []},
    }), encoding="utf-8")
    before = qualification_status(tmp_path)
    assert before["qualified"] is True
    assert before["integrity_ok"] is True

    payload = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    payload["merges"].append(payload["merges"][-1])
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    after = qualification_status(tmp_path)
    assert after["qualified"] is False
    assert after["integrity_ok"] is False


def test_model_config_rejects_unknown_tokenizer_version():
    with pytest.raises(ValueError, match="unsupported tokenizer_version"):
        GeneralistLMConfig(tokenizer_version="download-me-v99").validate()
