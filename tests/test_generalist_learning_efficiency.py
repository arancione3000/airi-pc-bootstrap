from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from generalist_lm.learning_efficiency_audit import dataset_audit, forensic_rollbacks
from generalist_lm.learning_efficiency_benchmark import (
    Trial,
    grad_relation,
    replay_selection,
    slice_hash,
    source_digest,
    trial_matrix,
    validate_output_path,
)
from generalist_lm.bootstrap_training import (
    _filter_protected_replay,
    _segment_language_gate,
)
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.pretraining import CorpusDocument
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.training import (
    SFTExample,
    causal_training_objective,
    reference_kl_loss,
)


def runtime():
    return GeneralistRuntime.fresh(
        GeneralistLMConfig(
            context_length=32, d_model=32, n_layers=1, n_heads=4, d_ff=64
        )
    )


def rows():
    return [
        SFTExample(
            [
                {"role": "user", "content": "Say something nice."},
                {"role": "assistant", "content": s},
            ]
        )
        for s in [
            "Good morning everyone.",
            "Have a pleasant evening.",
            "Please enjoy the sunshine.",
        ]
    ]


def test_replay_token_budget_counts_only_shifted_supervision():
    r = runtime()
    selected, count = replay_selection(
        r,
        rows(),
        rng=random.Random(7),
        examples=2,
        token_fraction=0.25,
        causal_tokens=100,
    )
    assert count >= 34
    assert count < 34 + 32
    assert len(set(selected[:3])) == len(selected[:3])
    assert count / (100 + count) >= 0.25


@pytest.mark.parametrize("fraction", [0, 1, -1, 2])
def test_invalid_actual_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        replay_selection(
            runtime(),
            rows(),
            rng=random.Random(0),
            examples=2,
            token_fraction=fraction,
            causal_tokens=100,
        )


def test_fixed_row_policy_is_reproducible():
    r = runtime()
    first = replay_selection(
        r,
        rows(),
        rng=random.Random(12),
        examples=2,
        token_fraction=None,
        causal_tokens=100,
    )
    second = replay_selection(
        r,
        rows(),
        rng=random.Random(12),
        examples=2,
        token_fraction=None,
        causal_tokens=100,
    )
    assert first == second


def test_replay_loss_weight_is_not_more_diverse_data():
    import torch

    logits = torch.tensor(
        [[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[0, 1, 2]])
    ids = labels.clone()
    one, _ = causal_training_objective(logits, labels, ids)
    doubled, _ = causal_training_objective(
        logits.repeat(2, 1, 1), labels.repeat(2, 1), ids.repeat(2, 1)
    )
    assert torch.allclose(one, doubled)
    assert torch.allclose(
        torch.autograd.grad(one, logits, retain_graph=True)[0],
        torch.autograd.grad(doubled, logits, retain_graph=True)[0],
    )
    assert torch.allclose(
        torch.autograd.grad(one * 4, logits, retain_graph=True)[0],
        4 * torch.autograd.grad(one, logits)[0],
    )


def test_live_reference_kl_has_no_first_step_restoring_pressure():
    import torch

    torch.manual_seed(3)
    logits = torch.randn(2, 5, 16, requires_grad=True)
    labels = torch.randint(0, 16, (2, 5))
    kl, count = reference_kl_loss(logits, logits.detach().clone(), labels)
    grad = torch.autograd.grad(kl, logits)[0]
    assert count == 8
    assert abs(float(kl.detach())) < 1e-7
    assert float(grad.abs().max()) < 1e-7


def test_gradient_relations_measure_conflict():
    import torch

    relation = grad_relation(
        {"a": torch.tensor([3.0, 4.0])}, {"a": torch.tensor([-3.0, -4.0])}
    )
    assert relation == {"left_norm": 5.0, "right_norm": 5.0, "cosine": -1.0}


def test_output_cannot_overwrite_source(tmp_path):
    root = tmp_path / "state"
    for path in [root, root / "metrics.json", tmp_path]:
        with pytest.raises(ValueError):
            validate_output_path(root, path)
    validate_output_path(root, tmp_path / "results.jsonl")


def test_source_digest_detects_checkpoint_or_metadata_mutation(tmp_path):
    (tmp_path / "checkpoint").write_text("before")
    old = source_digest(tmp_path)
    (tmp_path / "checkpoint").write_text("after")
    assert source_digest(tmp_path) != old


def test_slice_hash_changes_with_data_not_global_seed():
    assert slice_hash([[1, 2, 3]], rows()) == slice_hash([[1, 2, 3]], rows())
    assert slice_hash([[1, 2, 3]], rows()) != slice_hash([[1, 2, 4]], rows())


def test_ablation_matrix_keeps_baseline_and_explicit_optimizer_policy():
    matrix = trial_matrix(16256)
    assert matrix[0] == Trial(segment_tokens=16256)
    assert all(t.segment_tokens == 16256 for t in matrix)
    assert {t.optimizer_state for t in matrix} == {"reset", "resume"}
    assert any(t.prefix_acceptance for t in matrix)


def test_forensic_history_does_not_invent_missing_causes():
    report = forensic_rollbacks(
        {
            "segment_guard_rejections": [
                {
                    "reasons": ["anchor"],
                    "attempted_tokens": 4064,
                    "learning_rate_scale": 0.01,
                }
            ]
        },
        {
            "history": [
                {
                    "accepted": False,
                    "updated_at_unix": 100,
                    "accepted_tokens": 0,
                    "elapsed_training_seconds": 1,
                },
                {
                    "accepted": True,
                    "updated_at_unix": 200,
                    "accepted_tokens": 100,
                    "elapsed_training_seconds": 1,
                },
            ]
        },
    )
    assert report["causes_in_retained_rejections"] == {"anchor": 1}
    assert report["recent_window"]["rollback_rate"] == 0.5
    assert (
        "per-objective gradient norms"
        in report["not_recoverable_from_existing_history"]
    )


def test_benchmark_replay_excludes_holdout():
    bad = SFTExample(
        [
            {"role": "user", "content": "Come ti chiami?"},
            {"role": "assistant", "content": "Mi chiamo AIRI."},
        ]
    )
    selected, _ = _filter_protected_replay(rows() + [bad])
    assert bad not in selected


def test_repetition_hard_boundary_is_preserved():
    before = {
        "language_nll": 2.0,
        "repetition_rate": 0.23,
        "generation_similarity": 0.3,
        "multiword_output_rate": 1.0,
        "non_empty_rate": 1.0,
        "token_entropy": 2.0,
        "pathological_repetition": False,
    }
    anchor = {**before, "repetition_rate": 0.1577}
    after = {**before, "repetition_rate": 0.2378}
    ok, gate = _segment_language_gate(before, after, anchor, attempted_tokens=16256)
    assert not ok
    assert (
        "repetition rate exceeds durable anchor by more than 0.08"
        in gate["after_anchor_violations"]
    )


@pytest.fixture
def offline_state(tmp_path):
    import hashlib
    from generalist_lm.bootstrap_data import BootstrapDataBundle, write_bootstrap_replay

    root = tmp_path / "source"
    b = root / "bootstrap-data"
    r = GeneralistRuntime.fresh(
        GeneralistLMConfig(
            context_length=128, d_model=32, n_layers=1, n_heads=4, d_ff=64
        )
    )
    r.save_checkpoint(b / "candidate")
    text = "A person opens a window and watches the street."
    doc = CorpusDocument(
        source="reviewed:en",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        bytes=len(text),
        domain="language",
    )
    bundle = BootstrapDataBundle([doc], [], rows(), [], {})
    write_bootstrap_replay(bundle, r.tokenizer, output_dir=b)
    (b / "progress.json").write_text(
        json.dumps(
            {
                "steps": 0,
                "segment_guard_lr_scale": 0.01,
                "tokens_processed": 100000,
                "target_tokens": 1000000,
            }
        )
    )
    health = {
        "language_nll": 2.0,
        "repetition_rate": 0.2,
        "generation_similarity": 0.3,
        "multiword_output_rate": 1.0,
        "non_empty_rate": 1.0,
        "token_entropy": 2.0,
        "pathological_repetition": False,
    }
    (b / "language-guard.json").write_text(json.dumps({"report": health}))
    return root, health


@pytest.mark.parametrize("accepted", [True, False])
def test_offline_acceptance_and_rejection_never_change_source(
    offline_state, monkeypatch, accepted
):
    import torch
    import generalist_lm.learning_efficiency_benchmark as bench

    torch.set_num_threads(1)
    root, health = offline_state
    initial = source_digest(root)
    monkeypatch.setattr(bench, "evaluate_phase5_language", lambda runtime: dict(health))
    real_gate = bench._segment_language_gate

    def gate(before, after, anchor, **kwargs):
        ok, report = real_gate(before, after, anchor, **kwargs)
        if not accepted:
            report.update(accepted=False, local_reasons=["test rejection"])
        return accepted, report

    monkeypatch.setattr(bench, "_segment_language_gate", gate)
    result = bench.run_trial(root, Trial(segment_tokens=100), seed=3)
    assert result["accepted"] is accepted
    assert result["accepted_equivalent_tokens"] == (
        result["attempted_causal_tokens"] if accepted else 0
    )
    assert result["live_tokens_persisted"] == 0
    assert source_digest(root) == initial
    assert (
        json.loads((root / "bootstrap-data/progress.json").read_text())[
            "tokens_processed"
        ]
        == 100000
    )


def test_interrupted_trial_preserves_source_and_resume_starts_at_same_checkpoint(
    offline_state, monkeypatch
):
    import generalist_lm.learning_efficiency_benchmark as bench

    root, health = offline_state
    initial = source_digest(root)
    calls = 0

    def interruption(runtime):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("simulated interruption after optimizer step")
        return dict(health)

    monkeypatch.setattr(bench, "evaluate_phase5_language", interruption)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        bench.run_trial(root, Trial(segment_tokens=100), seed=3)
    assert source_digest(root) == initial
    assert calls == 2
    monkeypatch.setattr(bench, "evaluate_phase5_language", lambda runtime: dict(health))
    resumed = bench.run_trial(root, Trial(segment_tokens=100), seed=3)
    from generalist_lm.qualification import checkpoint_digest

    assert resumed["checkpoint_hash"] == checkpoint_digest(
        root / "bootstrap-data/candidate"
    )
    assert source_digest(root) == initial
