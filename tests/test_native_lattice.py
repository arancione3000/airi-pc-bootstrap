from __future__ import annotations

import json
from pathlib import Path

import pytest


def _tiny_config(**overrides):
    from generalist_lm.native_lattice import AiriLatticeConfig

    values = dict(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_cells=1,
        memory_bands=3,
        d_expert=48,
        n_experts=4,
        active_experts=1,
        max_reasoning_steps=3,
        surprise_threshold=0.25,
        dropout=0.0,
    )
    values.update(overrides)
    return AiriLatticeConfig(**values).validate()


def _write_corpus(tmp_path: Path) -> Path:
    rows = []
    for domain in ("general", "reasoning", "code", "math"):
        for index in range(2):
            path = tmp_path / f"{domain}-{index}.txt"
            path.write_text(
                (
                    f"AIRI {domain} training sample {index}. "
                    f"This repeated sequence teaches a stable {domain} pattern. "
                )
                * 12,
                encoding="utf-8",
            )
            rows.append({
                "path": path.name,
                "domain": domain,
                "language": "en",
                "license": "project-owned-test-data",
                "source_type": "owned",
                "approved_for_training": True,
                "weight": 1.0,
            })
    manifest = tmp_path / "native-corpus.json"
    manifest.write_text(
        json.dumps({"version": "native-corpus-v1", "documents": rows}),
        encoding="utf-8",
    )
    return manifest


def test_lattice_forward_is_causal_and_returns_constant_state():
    torch = pytest.importorskip("torch")
    from generalist_lm.native_lattice import AiriLatticeLM

    torch.manual_seed(7)
    config = _tiny_config()
    model = AiriLatticeLM(config).eval()

    left = torch.tensor([[1, 20, 21, 22, 23, 24]], dtype=torch.long)
    right = left.clone()
    right[0, -2:] = torch.tensor([111, 112])

    out_left = model(left, return_state=True)
    out_right = model(right, return_state=True)

    assert out_left["logits"].shape == (1, left.shape[1], config.vocab_size)
    assert len(out_left["lattice_state"]) == config.n_cells
    state = out_left["lattice_state"][0]
    assert state.shape == (
        config.memory_bands,
        1,
        config.d_model,
    )

    # Positions before the modified suffix must be exactly independent of
    # future tokens.
    assert torch.allclose(
        out_left["logits"][:, :-2],
        out_right["logits"][:, :-2],
        atol=1e-6,
        rtol=1e-6,
    )


def test_lattice_sparse_experts_and_adaptive_reasoning_report_real_usage():
    torch = pytest.importorskip("torch")
    from generalist_lm.native_lattice import AiriLatticeLM

    torch.manual_seed(11)
    low_compute = AiriLatticeLM(
        _tiny_config(
            surprise_threshold=0.90,
            max_reasoning_steps=4,
        )
    ).eval()
    torch.manual_seed(11)
    adaptive = AiriLatticeLM(
        _tiny_config(
            surprise_threshold=0.0,
            max_reasoning_steps=4,
        )
    ).eval()

    ids = torch.tensor([[1, 30, 31, 32, 33, 34, 35, 36]], dtype=torch.long)
    low = low_compute(ids)["stats"]
    high = adaptive(ids)["stats"]

    assert len(high["expert_usage"]) == 4
    assert sum(high["expert_usage"]) == pytest.approx(1.0, abs=1e-5)
    assert 1.0 <= low["mean_reasoning_steps"] <= 4.0
    assert 1.0 <= high["mean_reasoning_steps"] <= 4.0
    assert high["mean_reasoning_steps"] >= low["mean_reasoning_steps"]


def test_lattice_generation_reuses_recurrent_state_beyond_prompt():
    torch = pytest.importorskip("torch")
    from generalist_lm.native_lattice import AiriLatticeLM

    torch.manual_seed(5)
    config = _tiny_config()
    model = AiriLatticeLM(config).eval()
    prompt = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)
    generated = model.generate(
        prompt,
        max_new_tokens=5,
        eos_token_id=None,
    )
    assert generated.shape == (1, 9)


def test_lattice_state_memory_is_independent_of_context_length():
    pytest.importorskip("torch")
    from generalist_lm.native_lattice import (
        lattice_state_bytes,
        lattice_active_parameter_estimate,
        lattice_parameter_count,
    )

    short = _tiny_config(context_length=32)
    long = _tiny_config(context_length=8192)
    assert lattice_state_bytes(short) == lattice_state_bytes(long)
    assert lattice_parameter_count(short) == lattice_parameter_count(long)
    assert lattice_active_parameter_estimate(short) < lattice_parameter_count(short)


def test_lattice_lab_runs_scratch_equal_budget_comparison(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.lattice_lab import (
        LatticeLabConfig,
        benchmark_lattice_against_transformer,
    )

    manifest = _write_corpus(tmp_path)
    result = benchmark_lattice_against_transformer(
        str(manifest),
        allowed_roots=[str(tmp_path)],
        lattice_config=_tiny_config(max_reasoning_steps=2),
        lab_config=LatticeLabConfig(
            steps=2,
            batch_size=1,
            learning_rate=3e-3,
            min_learning_rate=5e-4,
            validation_fraction=0.25,
            max_eval_blocks=4,
            seed=123,
            device="cpu",
            minimum_loss_gain=0.0,
            max_domain_regression=10.0,
            max_active_parameter_ratio=2.0,
        ),
    )

    assert result["ok"] is True
    assert result["external_pretrained"] is False
    assert result["research_only"] is True
    assert result["baseline"]["family"] == "airi-native-foundation"
    assert result["candidate"]["family"] == "airi-native-lattice"
    assert result["baseline"]["training"]["steps"] == 2
    assert result["candidate"]["training"]["steps"] == 2
    assert result["candidate"]["active_parameters"] <= result["candidate"]["parameters"]
    assert result["candidate"]["state_bytes_at_context"] < result["baseline"]["state_bytes_at_context"]
    assert isinstance(result["candidate_wins"], bool)
