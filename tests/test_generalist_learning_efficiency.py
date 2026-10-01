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


@pytest.mark.parametrize("policy", ["ffn", "embeddings", "attention_and_norm"])
def test_parameter_localization_freezes_weights_and_excludes_optimizer_moments(policy):
    import torch
    from generalist_lm.learning_efficiency_benchmark import (
        configure_trainable_parameters,
    )
    from generalist_lm.bootstrap_training import _protected_optimizer

    r = runtime()
    report = configure_trainable_parameters(r.model, policy)
    frozen = {
        n: p.detach().clone()
        for n, p in r.model.named_parameters()
        if not p.requires_grad
    }
    optimizer, _ = _protected_optimizer(r.model, learning_rate=0.001)
    selected = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert report["trainable"] + report["frozen"] == sum(
        p.numel() for p in r.model.parameters()
    )
    assert all(p.requires_grad == (id(p) in selected) for p in r.model.parameters())
    logits = r.model(torch.tensor([[3, 12, 15, 20]]))["logits"]
    logits.square().mean().backward()
    optimizer.step()
    for name, parameter in r.model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter.detach(), frozen[name])
            assert parameter not in optimizer.state


def test_unknown_parameter_policy_fails_before_freezing():
    from generalist_lm.learning_efficiency_benchmark import (
        configure_trainable_parameters,
    )

    r = runtime()
    with pytest.raises(ValueError):
        configure_trainable_parameters(r.model, "unknown")
    assert all(p.requires_grad for p in r.model.parameters())


def test_document_splits_and_content_hashes_are_checked(tmp_path):
    import hashlib
    from generalist_lm.learning_efficiency_benchmark import (
        read_split_documents,
        require_disjoint_documents,
    )

    path = tmp_path / "documents.jsonl"
    row = dict(
        source="reviewed:en",
        text="A sunny morning.",
        bytes=16,
        domain="language",
        split="validation",
    )
    row["sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    path.write_text(json.dumps(row) + "\n")
    val = read_split_documents(path, "validation")
    with pytest.raises(ValueError, match="explicit train"):
        read_split_documents(path, "train")
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint_documents(val, val)
    row["text"] += "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_split_documents(path, "validation")


def test_corpus_subset_evaluation_is_read_only_and_restores_mode():
    import hashlib
    import torch
    from generalist_lm.learning_efficiency_benchmark import evaluate_corpus_subset

    r = runtime()
    r.model.train()
    text = "A sunny morning brings light to the room."
    doc = CorpusDocument(
        source="validation:en",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        bytes=len(text),
        domain="language",
    )
    weights = {n: p.detach().clone() for n, p in r.model.named_parameters()}
    report = evaluate_corpus_subset(r, [doc], max_blocks=2)
    assert report["supervised_tokens"] > 0
    assert report["loss"] > 0
    assert r.model.training
    assert all(
        p.grad is None and torch.equal(weights[n], p)
        for n, p in r.model.named_parameters()
    )
    assert (
        report["block_sha256"]
        == evaluate_corpus_subset(r, [doc], max_blocks=2)["block_sha256"]
    )


def test_ffn_down_scaling_changes_only_output_learning_rate_and_keeps_every_parameter():
    from generalist_lm.learning_efficiency_benchmark import scale_ffn_down_groups
    from generalist_lm.bootstrap_training import _protected_optimizer

    r = runtime()
    optimizer, _ = _protected_optimizer(r.model, learning_rate=0.001)
    names = {id(p): n for n, p in r.model.named_parameters()}
    original = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    scale_ffn_down_groups(optimizer, r.model, 1 / 64)
    after = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    assert set(after) == set(original)
    assert len(after) == sum(len(g["params"]) for g in optimizer.param_groups)
    for pid, lr in after.items():
        assert lr == original[pid] * (1 / 64 if ".ff.down." in names[pid] else 1)


@pytest.mark.parametrize("scale", [0, -1, 2, float("nan")])
def test_ffn_down_scaling_rejects_invalid_scale_before_mutating_groups(scale):
    from generalist_lm.learning_efficiency_benchmark import scale_ffn_down_groups
    from generalist_lm.bootstrap_training import _protected_optimizer

    r = runtime()
    optimizer, _ = _protected_optimizer(r.model, learning_rate=0.001)
    before = [id(g) for g in optimizer.param_groups]
    with pytest.raises(ValueError):
        scale_ffn_down_groups(optimizer, r.model, scale)
    assert before == [id(g) for g in optimizer.param_groups]


def test_offline_sequence_reuses_only_accepted_checkpoint_and_never_counts_rejected_tokens(
    offline_state, monkeypatch
):
    import torch
    import generalist_lm.learning_efficiency_benchmark as bench

    torch.set_num_threads(1)
    root, health = offline_state
    original = source_digest(root)
    monkeypatch.setattr(bench, "evaluate_phase5_language", lambda runtime: dict(health))
    real_gate = bench._segment_language_gate
    calls = 0

    def gate(before, after, anchor, **kwargs):
        nonlocal calls
        calls += 1
        ok, report = real_gate(before, after, anchor, **kwargs)
        return calls != 2, report

    monkeypatch.setattr(bench, "_segment_language_gate", gate)
    events = []
    report = bench.run_sequence(
        root, Trial(), [100, 100, 100], seed=3, on_segment=events.append
    )
    first, rejected, last = report["segments"]
    assert [r["accepted"] for r in events] == [True, False, True]
    assert rejected["checkpoint_hash"] == first["exported_checkpoint_hash"]
    assert last["checkpoint_hash"] == first["exported_checkpoint_hash"]
    assert rejected["exported_checkpoint_hash"] is None
    assert (
        last["offline_sequence_prior_accepted_tokens"]
        == first["accepted_equivalent_tokens"]
    )
    assert (
        report["accepted_equivalent_tokens"]
        == first["accepted_equivalent_tokens"] + last["accepted_equivalent_tokens"]
    )
    assert report["live_tokens_persisted"] == 0
    assert source_digest(root) == original


def test_offline_export_refuses_source_or_existing_directory(offline_state, tmp_path):
    import generalist_lm.learning_efficiency_benchmark as bench

    root, _ = offline_state
    with pytest.raises(ValueError):
        bench.run_trial(
            root, Trial(), seed=3, export_checkpoint=root / "bootstrap-data/candidate"
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not overwrite"):
        bench.run_trial(root, Trial(), seed=3, export_checkpoint=existing)


def test_generated_unlikelihood_targets_only_repeated_generated_ngrams():
    import torch
    from generalist_lm.learning_efficiency_benchmark import (
        generated_repetition_objective,
    )

    ids = torch.tensor([[8, 9, 10, 11, 8, 9, 10, 11]])
    logits = torch.zeros(1, 8, 16, requires_grad=True)
    loss, report = generated_repetition_objective(logits, ids, prompt_tokens=4)
    assert report == {"negative_positions": 1, "generated_tokens": 4}
    loss.backward()
    assert logits.grad[0, 6, 11] > 0
    assert logits.grad[0, :6].abs().max() == 0
    assert logits.grad[0, 7].abs().max() == 0


def test_generated_unlikelihood_never_penalizes_structural_or_prompt_tokens():
    import torch
    from generalist_lm.learning_efficiency_benchmark import (
        generated_repetition_objective,
    )

    ids = torch.tensor([[1, 9, 10, 11, 1, 9, 10, 11]])
    logits = torch.zeros(1, 8, 16, requires_grad=True)
    loss, report = generated_repetition_objective(logits, ids, prompt_tokens=4)
    assert report["negative_positions"] == 0
    loss.backward()
    assert logits.grad.abs().max() == 0
    ids = torch.tensor([[8, 9, 10, 11, 8, 9, 10, 11]])
    loss, report = generated_repetition_objective(logits, ids, prompt_tokens=8)
    assert report["negative_positions"] == 0


def test_generated_objective_uses_supplied_training_prefix_and_default_decoding():
    import torch
    from generalist_lm.learning_efficiency_benchmark import (
        training_prefix_generated_objective,
    )

    r = runtime()
    observed = []

    def generation(prompt, **kwargs):
        observed.append((prompt, kwargs))
        return [8, 9, 10, 11]

    r._generate_ids = generation
    r.model.train()
    loss, stats = training_prefix_generated_objective(r, [[8, 9, 10, 11]])
    assert observed == [
        (
            [8, 9, 10, 11],
            {"max_new_tokens": 40, "temperature": 0.0, "repetition_penalty": 1.0},
        )
    ]
    assert stats["generated_tokens"] == 4
    assert stats["negative_positions"] == 1
    loss.backward()
    assert r.model.training
    assert any(p.grad is not None for p in r.model.parameters())


def test_generated_training_trajectory_matching_holdout_is_excluded():
    from generalist_lm.learning_efficiency_benchmark import (
        training_prefix_generated_objective,
    )

    r = runtime()
    r._generate_ids = lambda prompt, **kwargs: r.tokenizer.encode("Come ti chiami?")
    r.model.train()
    loss, stats = training_prefix_generated_objective(
        r, [r.tokenizer.encode("A sunny morning. ")]
    )
    assert stats["holdout_filtered_trajectories"] == 1
    assert stats["negative_positions"] == 0
    assert stats["generated_tokens"] > 0
    assert loss.item() == 0
    loss.backward()
    assert r.model.training


def test_causal_replay_accounting_excludes_prompt_shift_and_padding():
    from generalist_lm.learning_efficiency_benchmark import causal_replay_selection
    from generalist_lm.tokenizer import PAD

    blocks = [[12, 13, 14, PAD, PAD], [20, 21, 22, 23, 24]]
    selected, count = causal_replay_selection(
        blocks, rng=random.Random(1), examples=2, token_fraction=None, causal_tokens=16
    )
    assert set(selected) == {0, 1}
    assert count == 6
    selected, count = causal_replay_selection(
        blocks, rng=random.Random(1), examples=2, token_fraction=0.25, causal_tokens=12
    )
    assert count >= 4
    assert count == sum(sum(t != PAD for t in blocks[i][1:]) for i in selected)


def test_causal_replay_slice_hash_is_separate_from_new_data():
    assert slice_hash([[12, 13]], [], [[20, 21]]) != slice_hash(
        [[12, 13]], [], [[20, 22]]
    )
    assert slice_hash([[12, 13]], []) == slice_hash([[12, 13]], [], [])


def test_offline_causal_protected_replay_records_real_labels_without_changing_new_budget(
    offline_state, monkeypatch
):
    import torch
    import generalist_lm.learning_efficiency_benchmark as bench

    torch.set_num_threads(1)
    root, health = offline_state
    original = source_digest(root)
    monkeypatch.setattr(bench, "evaluate_phase5_language", lambda runtime: dict(health))
    result = bench.run_trial(
        root,
        Trial(
            segment_tokens=100,
            replay_format="causal",
            replay_examples=8,
            replay_weight=0.25,
        ),
        seed=3,
    )
    assert result["replay_tokens"] == sum(s["replay_tokens"] for s in result["steps"])
    assert result["attempted_causal_tokens"] == sum(
        s["causal_tokens"] for s in result["steps"]
    )
    assert result["accepted_equivalent_tokens"] == result["attempted_causal_tokens"]
    assert result["replay_tokens"] > 0
    assert result["actual_replay_token_fraction"] == result["replay_tokens"] / (
        result["replay_tokens"] + result["attempted_causal_tokens"]
    )
    assert source_digest(root) == original


def test_generated_auxiliary_tokens_are_not_accepted_causal_tokens(
    offline_state, monkeypatch
):
    import torch
    import generalist_lm.learning_efficiency_benchmark as bench

    torch.set_num_threads(1)
    root, health = offline_state
    original = source_digest(root)
    monkeypatch.setattr(bench, "evaluate_phase5_language", lambda runtime: dict(health))

    def auxiliary(runtime, prefixes):
        assert len(prefixes) == 8
        loss = next(p for p in runtime.model.parameters() if p.requires_grad).sum() * 0
        return loss, dict(negative_positions=12, generated_tokens=200)

    monkeypatch.setattr(bench, "training_prefix_generated_objective", auxiliary)
    result = bench.run_trial(
        root,
        Trial(
            segment_tokens=100, autoregressive_ul_weight=1.0, autoregressive_prefixes=8
        ),
        seed=3,
    )
    assert result["accepted_equivalent_tokens"] == sum(
        s["causal_tokens"] for s in result["steps"]
    )
    assert all(
        s["autoregressive_unlikelihood"]["new_supervised_causal_tokens_counted"] == 0
        for s in result["steps"]
    )
    assert result["live_tokens_persisted"] == 0
    assert source_digest(root) == original


def test_corpus_validation_is_excluded_from_sft_replay_despite_different_split_hashes():
    import hashlib
    from generalist_lm.learning_efficiency_benchmark import (
        exclude_corpus_validation_replay,
    )

    text = "The flower is pink."
    val = CorpusDocument(
        source="validation:en",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        bytes=len(text),
        domain="language",
    )
    contaminated = SFTExample(
        [
            {"role": "user", "content": "Describe the flower."},
            {"role": "assistant", "content": " THE flower is pink. "},
        ]
    )
    safe = rows()[0]
    selected, count = exclude_corpus_validation_replay([contaminated, safe], [val])
    assert selected == [safe]
    assert count == 1


def test_anchored_decay_preserves_frozen_parameters_and_immutable_anchor():
    import copy
    import torch
    from generalist_lm.learning_efficiency_benchmark import anchored_parameter_decay

    model = torch.nn.Linear(2, 2)
    anchor = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    model.bias.requires_grad_(False)
    bias = model.bias.detach().clone()
    anchor_weights = {n: p.detach().clone() for n, p in anchor.named_parameters()}
    before = model.weight.detach().clone()
    report = anchored_parameter_decay(model, anchor, 0.05)
    assert torch.allclose(model.weight, before * 0.95 + anchor.weight * 0.05)
    assert torch.equal(model.bias, bias)
    assert all(torch.equal(anchor_weights[n], p) for n, p in anchor.named_parameters())
    assert report["actual_parameter_movement_norm"] > 0
    assert report["distance_to_anchor_after"] < report["distance_to_anchor_before"]


def test_anchored_decay_fails_before_mutation_on_topology_mismatch():
    import torch
    from generalist_lm.learning_efficiency_benchmark import anchored_parameter_decay

    model = torch.nn.Linear(2, 2)
    anchor = torch.nn.Linear(3, 2)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="topology"):
        anchored_parameter_decay(model, anchor, 0.01)
    assert torch.equal(model.weight, before)
