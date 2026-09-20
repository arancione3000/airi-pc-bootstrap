from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.distillation import DistillationPrompt, distill_prompts
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.pretraining import (
    load_local_corpus,
    pack_causal_blocks,
    pretrain_causal,
)
from generalist_lm.tokenizer import ByteTokenizer
from generalist_lm.transformers_lora import LoRATrainConfig, train_local_lora


def tiny_config():
    return GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()


def test_local_corpus_loader_bounds_deduplicates_and_blocks_escape(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha beta gamma " * 20, encoding="utf-8")
    (corpus / "b.md").write_text("alpha beta gamma " * 20, encoding="utf-8")
    (corpus / "skip.bin").write_bytes(b"\x00\x01\x02")

    report = load_local_corpus(
        [corpus],
        allowed_roots=[tmp_path],
        max_file_bytes=50_000,
        max_total_bytes=100_000,
    )
    assert len(report.documents) == 1
    assert any(row["reason"] == "duplicate" for row in report.skipped)
    assert any(row["reason"] == "unsupported_suffix" for row in report.skipped)
    assert report.total_bytes > 0

    outside = tmp_path.parent / "outside-corpus.txt"
    with pytest.raises(PermissionError):
        load_local_corpus([outside], allowed_roots=[tmp_path])


def test_local_corpus_loader_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "airi-outside-corpus.txt"
    outside.write_text("secret external corpus text", encoding="utf-8")
    try:
        link = tmp_path / "linked.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        report = load_local_corpus([link], allowed_roots=[tmp_path])
        assert report.documents == []
        assert any(row["reason"] == "symlink_escape" for row in report.skipped)
    finally:
        outside.unlink(missing_ok=True)


def test_causal_pretraining_reduces_loss_on_reviewed_corpus(tmp_path: Path):
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "tiny.txt").write_text(
        ("hello world. orange systems learn from text. " * 30),
        encoding="utf-8",
    )
    report = load_local_corpus([corpus], allowed_roots=[tmp_path])
    tok = ByteTokenizer()
    blocks = pack_causal_blocks(
        report.documents,
        tok,
        context_length=tiny_config().context_length,
    )
    assert blocks

    model = CausalTransformerLM(tiny_config())
    training = pretrain_causal(
        model,
        tok,
        report.documents,
        steps=35,
        batch_size=2,
        learning_rate=8e-3,
        weight_decay=0.0,
        seed=31,
    )
    assert training["ok"] is True
    assert training["final_loss"] < training["initial_loss"]
    assert training["loss_improvement"] > 0
    assert training["corpus_bytes"] == report.total_bytes


class Teacher:
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        del max_new_tokens
        if "add" in prompt.lower():
            return "def add(a, b):\n    return a + b"
        return "ORANGE"


def test_distillation_produces_reviewable_sft_rows_without_trusting_them():
    examples, report = distill_prompts(
        Teacher(),
        [
            DistillationPrompt("language", "Reply ORANGE"),
            DistillationPrompt("coding", "Write add(a,b)"),
            DistillationPrompt("coding", "Write add(a,b)"),
            DistillationPrompt("unknown", "bad"),
        ],
    )
    assert len(examples) == 2
    assert report["accepted"] == 2
    assert "does not bypass held-out qualification" in report["policy"]
    statuses = [row["status"] for row in report["rows"]]
    assert "duplicate" in statuses
    assert "rejected_domain" in statuses
    assert examples[0].messages[-1]["role"] == "assistant"


def test_lora_configuration_is_bounded_and_base_model_must_be_local(tmp_path: Path):
    assert LoRATrainConfig().validate().rank == 8
    with pytest.raises(ValueError):
        LoRATrainConfig(rank=9999).validate()
    with pytest.raises(FileNotFoundError):
        train_local_lora(
            tmp_path / "missing-model",
            [],
            tmp_path / "out",
        )


def test_lora_never_overwrites_base_model_even_before_optional_imports(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    from generalist_lm.training import SFTExample

    examples = [
        SFTExample([
            {"role": "user", "content": "Reply hi"},
            {"role": "assistant", "content": "hi"},
        ])
    ]
    with pytest.raises(ValueError, match="must not overwrite"):
        train_local_lora(base, examples, base / "adapter")
