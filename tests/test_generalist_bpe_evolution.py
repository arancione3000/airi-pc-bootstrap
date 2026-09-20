from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.bpe_tokenizer import BPETokenizer
from generalist_lm.evolution import GeneralistGenome, generate_challengers
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.research_cycle import (
    _research_eligible,
    _tokenizer_for_genome,
    _tokenizer_training_texts,
    _train_genome,
    _transfer_compatible_weights,
)
from generalist_lm.tokenizer import BYTE_OFFSET, VOCAB_SIZE as BYTE_VOCAB_SIZE, ByteTokenizer
from generalist_lm.training import SFTExample, nll_stats_on_examples
from generalist_lm.curriculum import validation_rows


def tiny_config(*, vocab_size: int, tokenizer_version: str):
    return GeneralistLMConfig(
        vocab_size=vocab_size,
        tokenizer_version=tokenizer_version,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()


def test_tokenizer_efficiency_signal_creates_bpe_challenger():
    champion = GeneralistGenome(tokenizer_version="byte-v1").validate()
    candidate = generate_challengers(
        champion,
        signals=["tokenizer_efficiency_gap"],
        count=1,
    )[0]
    assert candidate.tokenizer_version == "bpe-v1"
    assert candidate.parent_id == champion.genome_id
    assert candidate.generation == champion.generation + 1


def test_byte_to_bpe_embedding_migration_is_deterministic():
    torch = pytest.importorskip("torch")

    source_tokenizer = ByteTokenizer()
    pair = (BYTE_OFFSET + ord("a"), BYTE_OFFSET + ord("b"))
    target_tokenizer = BPETokenizer((pair,))

    source_model = CausalTransformerLM(
        tiny_config(vocab_size=source_tokenizer.vocab_size, tokenizer_version="byte-v1")
    )
    target_model = CausalTransformerLM(
        tiny_config(vocab_size=target_tokenizer.vocab_size, tokenizer_version="bpe-v1")
    )

    with torch.no_grad():
        for token_id in range(source_tokenizer.vocab_size):
            source_model.token_embedding.weight[token_id].fill_(float(token_id))

    report = _transfer_compatible_weights(
        source_model,
        target_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    assert report["vocabulary_migrated"] is True
    assert report["shared_token_rows"] == BYTE_VOCAB_SIZE
    assert report["derived_bpe_rows"] == 1

    migrated = target_model.token_embedding.weight.detach()
    assert torch.equal(
        migrated[:BYTE_VOCAB_SIZE],
        source_model.token_embedding.weight.detach()[:BYTE_VOCAB_SIZE],
    )
    expected = (
        float(BYTE_OFFSET + ord("a")) + float(BYTE_OFFSET + ord("b"))
    ) / 2.0
    assert torch.allclose(
        migrated[BYTE_VOCAB_SIZE],
        torch.full_like(migrated[BYTE_VOCAB_SIZE], expected),
    )


def test_nll_per_byte_denominator_is_tokenizer_invariant_for_same_target():
    pytest.importorskip("torch")
    example = SFTExample([
        {"role": "user", "content": "Reply OKOK"},
        {"role": "assistant", "content": "OKOK"},
    ])

    byte = ByteTokenizer()
    bpe = BPETokenizer((
        (BYTE_OFFSET + ord("O"), BYTE_OFFSET + ord("K")),
    ))

    byte_model = CausalTransformerLM(
        tiny_config(vocab_size=byte.vocab_size, tokenizer_version="byte-v1")
    )
    bpe_model = CausalTransformerLM(
        tiny_config(vocab_size=bpe.vocab_size, tokenizer_version="bpe-v1")
    )

    byte_stats = nll_stats_on_examples(byte_model, byte, [example])
    bpe_stats = nll_stats_on_examples(bpe_model, bpe, [example])

    assert byte_stats["target_bytes"] == 4
    assert bpe_stats["target_bytes"] == 4
    assert bpe_stats["target_tokens"] < byte_stats["target_tokens"]
    assert byte_stats["nll_per_byte"] > 0
    assert bpe_stats["nll_per_byte"] > 0


def report(
    *,
    tokenizer: str,
    loss: float,
    nll: float,
    domain_nll: float,
    token_accuracy: float,
):
    return {
        "finite": True,
        "tokenizer_version": tokenizer,
        "loss": loss,
        "nll_per_byte": nll,
        "domain_loss": {"language": loss},
        "domain_nll_per_byte": {"language": domain_nll},
        "target_token_accuracy": token_accuracy,
        "domain_token_accuracy": {"language": token_accuracy},
        "solved_items": [],
        "generation_exact_accuracy": 0.0,
        "domain_generation_accuracy": {"language": 0.0},
        "generated_solved_items": [],
    }


def test_cross_tokenizer_promotion_uses_nll_per_byte_not_token_loss():
    champion = report(
        tokenizer="byte-v1",
        loss=1.0,
        nll=2.0,
        domain_nll=2.0,
        token_accuracy=1.0,
    )
    candidate = report(
        tokenizer="bpe-v1",
        loss=5.0,
        nll=1.0,
        domain_nll=1.0,
        token_accuracy=0.0,
    )
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.1,
        max_domain_regression=0.0,
    )
    assert ok is True, reason
    assert "NLL/byte" in reason


def test_cross_tokenizer_domain_nll_regression_is_blocked():
    champion = report(
        tokenizer="byte-v1",
        loss=3.0,
        nll=2.0,
        domain_nll=1.0,
        token_accuracy=0.2,
    )
    candidate = report(
        tokenizer="bpe-v1",
        loss=1.0,
        nll=1.0,
        domain_nll=1.2,
        token_accuracy=0.9,
    )
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.1,
        max_domain_regression=0.0,
    )
    assert ok is False
    assert "byte-normalized validation domain" in reason



def test_bpe_training_text_builder_excludes_validation_prompts():
    texts = _tokenizer_training_texts([], [])
    joined = "\n".join(texts)
    assert texts
    for row in validation_rows():
        assert str(row.messages[0]["content"]) not in joined


def test_byte_champion_can_train_real_bpe_challenger():
    pytest.importorskip("torch")
    source_genome = GeneralistGenome(
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        learning_rate=3e-3,
        tokenizer_version="byte-v1",
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    source_tokenizer = ByteTokenizer()
    source_model = CausalTransformerLM(
        source_genome.model_config(source_tokenizer.vocab_size)
    )

    candidate = GeneralistGenome(**{
        **source_genome.to_dict(),
        "generation": 1,
        "parent_id": source_genome.genome_id,
        "genome_id": "bpe-e2e-candidate",
        "tokenizer_version": "bpe-v1",
    }).validate()
    tokenizer = _tokenizer_for_genome(
        candidate,
        replay_rows=[],
        pretrain_documents=[],
        bpe_vocab_size=300,
        bpe_max_bytes=32_000,
    )
    assert tokenizer.version == "bpe-v1"
    assert tokenizer.vocab_size > BYTE_VOCAB_SIZE

    runtime, report = _train_genome(
        candidate,
        steps=2,
        seed=123,
        device="cpu",
        replay_rows=[],
        source_model=source_model,
        source_tokenizer=source_tokenizer,
        tokenizer=tokenizer,
        pretrain_documents=[],
        pretrain_steps=0,
    )
    assert runtime.tokenizer.version == "bpe-v1"
    assert report["tokenizer_version"] == "bpe-v1"
    assert report["finite"] is True
    assert report["nll_per_byte"] > 0
    assert report["weight_transfer"]["vocabulary_migrated"] is True
    assert report["weight_transfer"]["derived_bpe_rows"] > 0
