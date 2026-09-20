from __future__ import annotations

import math
from typing import Any, Iterable

from .native_lattice import (
    AiriLatticeConfig,
    lattice_active_parameter_estimate,
    lattice_parameter_count,
    lattice_state_bytes,
)


LATTICE_MATH_VERSION = "airi-lattice-math-v0"


def geometric_memory_decays(config: AiriLatticeConfig) -> tuple[float, ...]:
    config = config.validate()
    lo = math.log(config.memory_decay_min)
    hi = math.log(config.memory_decay_max)
    if config.memory_bands == 1:
        return (math.exp(lo),)
    return tuple(
        math.exp(
            lo + (hi - lo) * index / (config.memory_bands - 1)
        )
        for index in range(config.memory_bands)
    )


def memory_half_lives(config: AiriLatticeConfig) -> tuple[float, ...]:
    """Return the token half-life implied by every recurrent decay band."""
    out = []
    for decay in geometric_memory_decays(config):
        if not (0.0 < decay < 1.0):
            out.append(float("inf"))
        else:
            out.append(math.log(0.5) / math.log(decay))
    return tuple(out)


def memory_horizon_summary(config: AiriLatticeConfig) -> dict[str, Any]:
    half_lives = memory_half_lives(config)
    return {
        "bands": len(half_lives),
        "half_lives_tokens": list(half_lives),
        "fast_half_life": min(half_lives),
        "slow_half_life": max(half_lives),
        "dynamic_range": max(half_lives) / max(1e-9, min(half_lives)),
    }


def stability_certificate(config: AiriLatticeConfig) -> dict[str, Any]:
    """Certify the bounded-state region used by Lattice v0.

    Every routing row is softmax-normalized and therefore a convex combination
    of existing band states. Candidate writes pass through tanh and are bounded
    in [-1, 1]. For 0 < decay < 1 and lattice_mix <= 1, the recurrent update is
    a contraction toward a bounded convex mixture. This is a conservative
    infinity-norm certificate for the memory state, independent of sequence
    length.
    """
    config = config.validate()
    decays = geometric_memory_decays(config)
    routing_nonexpansive = 0.0 <= config.lattice_mix <= 1.0
    contractive = all(0.0 < value < 1.0 for value in decays)
    bounded_candidate = True
    certified = routing_nonexpansive and contractive and bounded_candidate
    return {
        "ok": certified,
        "version": LATTICE_MATH_VERSION,
        "routing_nonexpansive": routing_nonexpansive,
        "all_memory_decays_contractive": contractive,
        "bounded_candidate_write": bounded_candidate,
        "max_decay": max(decays),
        "min_decay": min(decays),
        "lattice_mix": config.lattice_mix,
        "claim": (
            "memory recurrence remains in the conservative bounded-state region"
            if certified
            else "memory recurrence is outside the conservative stability region"
        ),
    }


def expected_reasoning_steps(
    config: AiriLatticeConfig,
    surprise_values: Iterable[float],
) -> float:
    config = config.validate()
    values = [min(1.0, max(0.0, float(value))) for value in surprise_values]
    if not values:
        return 1.0
    total = 0.0
    for surprise in values:
        normalized = max(
            0.0,
            min(
                1.0,
                (surprise - config.surprise_threshold)
                / max(1e-9, 1.0 - config.surprise_threshold),
            ),
        )
        shaped = normalized ** config.surprise_power
        extra = math.floor(
            shaped * (config.max_reasoning_steps - 1) + 1e-6
        )
        total += 1 + extra
    return total / len(values)


def architecture_cost_vector(
    config: AiriLatticeConfig,
    *,
    batch_size: int = 1,
    representative_surprise: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> dict[str, float]:
    config = config.validate()
    total = float(lattice_parameter_count(config))
    active = float(lattice_active_parameter_estimate(config))
    state = float(lattice_state_bytes(config, batch_size=batch_size))
    reasoning = expected_reasoning_steps(config, representative_surprise)
    # This is a relative architecture-search proxy, not a FLOP claim.
    compute_proxy = active * reasoning
    return {
        "total_parameters": total,
        "active_parameters": active,
        "state_bytes": state,
        "expected_reasoning_steps": reasoning,
        "active_parameter_step_proxy": compute_proxy,
    }


def pareto_dominates(
    left: dict[str, float],
    right: dict[str, float],
    *,
    minimize: tuple[str, ...] = (
        "loss",
        "active_parameters",
        "state_bytes",
        "train_seconds",
    ),
) -> bool:
    comparable = [
        key for key in minimize
        if key in left and key in right
        and math.isfinite(float(left[key]))
        and math.isfinite(float(right[key]))
    ]
    if not comparable:
        return False
    no_worse = all(
        float(left[key]) <= float(right[key])
        for key in comparable
    )
    strictly_better = any(
        float(left[key]) < float(right[key])
        for key in comparable
    )
    return no_worse and strictly_better


def pareto_front(
    rows: Iterable[dict[str, Any]],
    *,
    minimize: tuple[str, ...] = (
        "loss",
        "active_parameters",
        "state_bytes",
        "train_seconds",
    ),
) -> list[dict[str, Any]]:
    items = [dict(row) for row in rows]
    front: list[dict[str, Any]] = []
    for index, row in enumerate(items):
        dominated = False
        for other_index, other in enumerate(items):
            if index == other_index:
                continue
            if pareto_dominates(other, row, minimize=minimize):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front
