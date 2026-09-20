from __future__ import annotations

import pytest


def _config(**overrides):
    from generalist_lm.native_lattice import AiriLatticeConfig

    values = dict(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_cells=1,
        memory_bands=4,
        d_expert=64,
        n_experts=4,
        active_experts=1,
        max_reasoning_steps=3,
        surprise_threshold=0.25,
    )
    values.update(overrides)
    return AiriLatticeConfig(**values).validate()


def test_lattice_math_certifies_stability_and_multiscale_memory():
    pytest.importorskip("torch")
    from generalist_lm.lattice_math import (
        memory_half_lives,
        memory_horizon_summary,
        stability_certificate,
    )

    cfg = _config()
    half_lives = memory_half_lives(cfg)
    assert len(half_lives) == cfg.memory_bands
    assert all(right > left for left, right in zip(half_lives, half_lives[1:]))

    summary = memory_horizon_summary(cfg)
    assert summary["slow_half_life"] > summary["fast_half_life"]
    assert summary["dynamic_range"] > 1.0

    certificate = stability_certificate(cfg)
    assert certificate["ok"] is True
    assert certificate["routing_nonexpansive"] is True
    assert certificate["all_memory_decays_contractive"] is True


def test_lattice_pareto_front_keeps_nondominated_architectures():
    from generalist_lm.lattice_math import pareto_front

    rows = [
        {"name": "a", "loss": 2.0, "active_parameters": 100, "state_bytes": 100, "train_seconds": 5},
        {"name": "b", "loss": 1.9, "active_parameters": 110, "state_bytes": 100, "train_seconds": 5},
        {"name": "c", "loss": 2.1, "active_parameters": 120, "state_bytes": 120, "train_seconds": 6},
    ]
    front = pareto_front(rows)
    names = {row["name"] for row in front}
    assert "c" not in names
    assert names == {"a", "b"}


def test_lattice_population_mutates_architecture_inside_stable_bounds():
    pytest.importorskip("torch")
    from generalist_lm.lattice_evolution import (
        generate_lattice_population,
        root_lattice_genome,
    )
    from generalist_lm.lattice_math import stability_certificate

    champion = root_lattice_genome(_config())
    population = generate_lattice_population(
        champion,
        count=8,
        mathesis_signals=[
            "symbolic_reasoning_signal",
            "deep_symbolic_signal",
        ],
        research={
            "tag_counts": {
                "reasoning": 3,
                "efficiency": 2,
                "long-context": 1,
            }
        },
        max_total_parameters=3_000_000,
        max_active_parameter_ratio=2.0,
    )

    assert len(population) >= 4
    families = {mutation.family for mutation, _ in population}
    assert "reasoning" in families or "memory" in families
    for mutation, genome in population:
        assert genome.parent_id == champion.genome_id
        assert genome.generation == champion.generation + 1
        assert stability_certificate(genome.lattice_config())["ok"] is True
        assert mutation.changes


def test_successive_halving_spends_compute_only_on_survivors():
    from generalist_lm.lattice_evolution import successive_halving_plan

    plan = successive_halving_plan(
        8,
        first_stage_steps=2,
        stages=3,
        growth_factor=3,
        survival_fraction=0.5,
    )
    assert plan == [
        {"stage": 0, "candidates": 8, "steps": 2},
        {"stage": 1, "candidates": 4, "steps": 6},
        {"stage": 2, "candidates": 2, "steps": 18},
    ]
