from __future__ import annotations

import gzip
import io
import json

import pytest

from generalist_lm.bootstrap_data import (
    OASST1_REVISION,
    SOURCES,
    _oasst_conversations,
    _parse_oasst,
)
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.phase5_diagnostics import (
    PHASE5_PROBES,
    degeneration_gate,
    evaluate_phase5_language,
    protected_bootstrap_texts,
)
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import ByteTokenizer


def test_phase5_holdout_suite_is_explicit_and_protected():
    assert len(PHASE5_PROBES) == 7
    protected = protected_bootstrap_texts()
    assert "ciao" in protected
    assert "hello" in protected
    assert "write one simple sentence." in protected


def test_bootstrap_sources_are_explicitly_licensed_and_pinned():
    by_id = {row["id"]: row for row in SOURCES}
    assert by_id["tatoeba-en-cc0"]["license"] == "CC0-1.0"
    assert by_id["tatoeba-it-ccby"]["license"] == "CC-BY-2.0-FR"
    assert by_id["oasst1-human"]["license"] == "Apache-2.0"
    assert by_id["oasst1-human"]["revision"] == OASST1_REVISION
    assert len(OASST1_REVISION) == 40
    assert all(row["url"].startswith("https://") for row in SOURCES)
    assert all(row["license_url"].startswith("https://") for row in SOURCES)


def test_oasst_parser_excludes_synthetic_and_builds_human_dialogue():
    rows = [
        {
            "message_id": "u1",
            "parent_id": None,
            "text": "Tell me something simple.",
            "role": "prompter",
            "lang": "en",
            "deleted": False,
            "synthetic": False,
            "review_result": True,
        },
        {
            "message_id": "a1",
            "parent_id": "u1",
            "text": "A cat sleeps.",
            "role": "assistant",
            "lang": "en",
            "deleted": False,
            "synthetic": False,
            "review_result": True,
        },
        {
            "message_id": "a2",
            "parent_id": "u1",
            "text": "Synthetic answer.",
            "role": "assistant",
            "lang": "en",
            "deleted": False,
            "synthetic": True,
            "review_result": True,
        },
    ]
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb") as handle:
        for row in rows:
            handle.write((json.dumps(row) + "\n").encode("utf-8"))
    parsed = _parse_oasst(raw.getvalue())
    assert "a1" in parsed
    assert "a2" not in parsed
    conversations = _oasst_conversations(parsed)
    assert len(conversations) == 1
    assert conversations[0][1][-1]["content"] == "A cat sleeps."


def test_degeneration_gate_rejects_repeated_token_collapse():
    before = {
        "repetition_rate": 0.20,
        "token_entropy": 3.0,
    }
    after = {
        "pathological_repetition": True,
        "repetition_rate": 0.90,
        "token_entropy": 0.2,
        "longest_repeated_token_run": 12,
    }
    ok, reason = degeneration_gate(before, after)
    assert not ok
    assert "repetition" in reason


def test_phase5_diagnostics_run_on_real_local_generalist_model():
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=96,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
    ).validate()
    runtime = GeneralistRuntime(
        CausalTransformerLM(cfg),
        cfg,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )
    report = evaluate_phase5_language(runtime, max_new_tokens=4)
    required = {
        "token_entropy",
        "top1_probability",
        "top5_probability_mass",
        "repetition_rate",
        "longest_repeated_token_run",
        "unique_token_ratio",
        "eos_probability",
        "token_frequency_distribution",
        "generation_length",
        "bpe_token_distribution",
        "language_nll",
        "generation_similarity",
        "exact_accuracy",
        "non_empty_rate",
        "word_output_rate",
    }
    assert required <= set(report)
    assert report["suite_training_excluded"] is True
    assert len(report["traces"]) == len(PHASE5_PROBES)


def test_sampling_controls_preserve_raw_greedy_default():
    import torch

    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
    ).validate()
    model = CausalTransformerLM(cfg)
    tok = ByteTokenizer()
    prompt = torch.tensor([tok.encode("Hello", bos=True)], dtype=torch.long)

    first = model.generate(prompt, max_new_tokens=6, eos_token_id=None)
    second = model.generate(
        prompt,
        max_new_tokens=6,
        eos_token_id=None,
        temperature=0.0,
        top_k=1,
        top_p=0.01,
        repetition_penalty=2.0,
    )
    # Greedy intentionally ignores sampling knobs so canonical evaluation is stable.
    assert torch.equal(first, second)

    torch.manual_seed(123)
    sampled = model.generate(
        prompt,
        max_new_tokens=6,
        eos_token_id=None,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.08,
    )
    assert sampled.shape == first.shape
