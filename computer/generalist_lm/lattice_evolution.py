from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable

from .lattice_math import architecture_cost_vector, stability_certificate
from .native_lattice import AiriLatticeConfig


LATTICE_EVOLUTION_VERSION = "airi-lattice-evolution-v0"


def _digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class LatticeGenome:
    generation: int
    genome_id: str
    parent_id: str | None
    config: dict[str, Any]
    learning_rate: float = 2e-3
    min_learning_rate: float = 2e-4
    weight_decay: float = 0.05

    def validate(self) -> "LatticeGenome":
        self.generation = max(0, int(self.generation))
        if not str(self.genome_id).strip():
            raise ValueError("Lattice genome_id is required")
        cfg = AiriLatticeConfig.from_dict(dict(self.config))
        self.config = cfg.to_dict()
        self.learning_rate = float(self.learning_rate)
        self.min_learning_rate = float(self.min_learning_rate)
        self.weight_decay = float(self.weight_decay)
        if not (1e-6 <= self.learning_rate <= 0.05):
            raise ValueError("Lattice genome learning_rate out of bounds")
        if not (0.0 <= self.min_learning_rate <= self.learning_rate):
            raise ValueError("Lattice genome min_learning_rate out of bounds")
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError("Lattice genome weight_decay out of bounds")
        certificate = stability_certificate(cfg)
        if not certificate["ok"]:
            raise ValueError("Lattice genome is outside the certified stability region")
        return self

    def lattice_config(self) -> AiriLatticeConfig:
        return AiriLatticeConfig.from_dict(self.config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatticeMutation:
    name: str
    family: str
    changes: dict[str, Any]
    rationale: str
    mathesis_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "changes": dict(self.changes),
            "rationale": self.rationale,
            "mathesis_tags": list(self.mathesis_tags),
        }


def root_lattice_genome(
    config: AiriLatticeConfig | None = None,
    *,
    generation: int = 0,
) -> LatticeGenome:
    cfg = (config or AiriLatticeConfig()).validate()
    seed = {
        "generation": int(generation),
        "config": cfg.to_dict(),
    }
    return LatticeGenome(
        generation=int(generation),
        genome_id=f"lattice-{int(generation)}-{_digest(seed)[:12]}",
        parent_id=None,
        config=cfg.to_dict(),
    ).validate()


def _apply_config_change(
    raw: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if key not in raw:
        raise ValueError(f"unauthorized Lattice config mutation: {key}")
    raw[key] = value


def apply_lattice_mutation(
    champion: LatticeGenome,
    mutation: LatticeMutation,
) -> LatticeGenome:
    champion.validate()
    cfg = dict(champion.config)
    lr = champion.learning_rate
    min_lr = champion.min_learning_rate
    wd = champion.weight_decay

    for key, value in mutation.changes.items():
        if key == "learning_rate":
            lr = float(value)
        elif key == "min_learning_rate":
            min_lr = float(value)
        elif key == "weight_decay":
            wd = float(value)
        else:
            _apply_config_change(cfg, key, value)

    payload = {
        "parent": champion.genome_id,
        "generation": champion.generation + 1,
        "mutation": mutation.to_dict(),
    }
    return LatticeGenome(
        generation=champion.generation + 1,
        genome_id=f"lattice-{champion.generation + 1}-{_digest(payload)[:12]}",
        parent_id=champion.genome_id,
        config=cfg,
        learning_rate=lr,
        min_learning_rate=min(min_lr, lr),
        weight_decay=wd,
    ).validate()


def _signal_set(
    mathesis_signals: Iterable[str] | None,
    research: dict[str, Any] | None,
) -> set[str]:
    tags = {str(row).strip().lower() for row in (mathesis_signals or ()) if str(row).strip()}
    if isinstance(research, dict):
        counts = research.get("tag_counts")
        if isinstance(counts, dict):
            tags.update(
                str(key).strip().lower()
                for key, value in counts.items()
                if int(value or 0) > 0
            )
    return tags


def mutation_library(
    champion: LatticeGenome,
    *,
    mathesis_signals: Iterable[str] | None = None,
    research: dict[str, Any] | None = None,
) -> list[LatticeMutation]:
    champion.validate()
    cfg = champion.lattice_config()
    tags = _signal_set(mathesis_signals, research)
    mutations: list[LatticeMutation] = []

    reasoning_focus = bool(
        tags
        & {
            "symbolic_reasoning_signal",
            "deep_symbolic_signal",
            "reasoning_gap",
            "reasoning",
            "math",
        }
    )
    efficiency_focus = bool(tags & {"efficiency", "long-context"})
    tokenizer_focus = "tokenizer" in tags
    curriculum_focus = "research_curriculum_signal" in tags or "curriculum" in tags

    # Memory geometry.
    if cfg.memory_bands < 12:
        mutations.append(LatticeMutation(
            "memory-add-band",
            "memory",
            {"memory_bands": cfg.memory_bands + 1},
            "increase temporal resolution of the recurrent memory hierarchy",
            ("reasoning",),
        ))
    if cfg.memory_bands > 2:
        mutations.append(LatticeMutation(
            "memory-remove-band",
            "memory",
            {"memory_bands": cfg.memory_bands - 1},
            "compress the hierarchy when an intermediate timescale is redundant",
            ("efficiency",),
        ))
    mutations.extend([
        LatticeMutation(
            "memory-longer-slow-band",
            "memory",
            {"memory_decay_max": min(0.9999, 1.0 - (1.0 - cfg.memory_decay_max) * 0.5)},
            "extend the slowest mathematical half-life without increasing state size",
            ("long-context",),
        ),
        LatticeMutation(
            "memory-faster-fast-band",
            "memory",
            {"memory_decay_min": max(0.05, cfg.memory_decay_min * 0.8)},
            "make fast memory more reactive while preserving deep memory",
            ("reasoning",),
        ),
        LatticeMutation(
            "memory-deeper-surprise-write",
            "memory",
            {"deep_write_power": min(6.0, cfg.deep_write_power * 1.15)},
            "reserve deep memory more aggressively for high-surprise information",
            ("reasoning",),
        ),
        LatticeMutation(
            "memory-broader-deep-write",
            "memory",
            {"deep_write_power": max(0.5, cfg.deep_write_power * 0.85)},
            "allow more information to reach slow memory when deep writes are too rare",
            ("reasoning",),
        ),
    ])

    # Lattice graph communication.
    mutations.extend([
        LatticeMutation(
            "routing-more-cross-band",
            "routing",
            {"lattice_mix": min(0.8, cfg.lattice_mix + 0.08)},
            "increase information exchange among timescales",
            ("reasoning",),
        ),
        LatticeMutation(
            "routing-more-independent-bands",
            "routing",
            {"lattice_mix": max(0.0, cfg.lattice_mix - 0.06)},
            "reduce cross-band interference and encourage memory specialization",
            ("efficiency",),
        ),
    ])

    # Predictive-coding and local recurrent genes. These are off by default
    # so the original Lattice remains a valid genome; MATHESIS can explicitly
    # test whether the extra structure earns its active-parameter cost.
    if not cfg.predictive_error_memory:
        mutations.append(LatticeMutation(
            "prediction-enable-error-memory",
            "prediction",
            {"predictive_error_memory": True},
            "store learned latent prediction errors instead of only raw novelty",
            ("reasoning", "efficiency"),
        ))
    else:
        mutations.append(LatticeMutation(
            "prediction-disable-error-memory",
            "prediction",
            {"predictive_error_memory": False},
            "remove predictive coding if its extra parameters do not pay for themselves",
            ("efficiency",),
        ))

    if not cfg.local_recurrence:
        mutations.append(LatticeMutation(
            "local-enable-gated-recurrence",
            "local",
            {"local_recurrence": True},
            "add a learned short-horizon recurrent path before multiscale memory",
            ("reasoning", "code"),
        ))
    else:
        mutations.append(LatticeMutation(
            "local-disable-gated-recurrence",
            "local",
            {"local_recurrence": False},
            "remove the local recurrent path if multiscale memory already captures it",
            ("efficiency",),
        ))

    # Adaptive reasoning depth.
    if cfg.max_reasoning_steps < 12:
        mutations.append(LatticeMutation(
            "reasoning-more-recurrence",
            "reasoning",
            {"max_reasoning_steps": cfg.max_reasoning_steps + 1},
            "add one shared reasoning pass without adding another parameter block",
            ("reasoning",),
        ))
    if cfg.max_reasoning_steps > 1:
        mutations.append(LatticeMutation(
            "reasoning-less-recurrence",
            "reasoning",
            {"max_reasoning_steps": cfg.max_reasoning_steps - 1},
            "remove expensive passes if held-out quality does not need them",
            ("efficiency",),
        ))
    mutations.extend([
        LatticeMutation(
            "reasoning-trigger-earlier",
            "reasoning",
            {"surprise_threshold": max(0.02, cfg.surprise_threshold - 0.05)},
            "spend extra recurrent compute on a broader set of novel tokens",
            ("reasoning",),
        ),
        LatticeMutation(
            "reasoning-trigger-later",
            "reasoning",
            {"surprise_threshold": min(0.85, cfg.surprise_threshold + 0.05)},
            "save compute by reserving recurrent reasoning for more novel tokens",
            ("efficiency",),
        ),
        LatticeMutation(
            "reasoning-sharper-budget",
            "reasoning",
            {"surprise_power": min(5.0, cfg.surprise_power * 1.15)},
            "concentrate extra compute on the most surprising inputs",
            ("reasoning",),
        ),
        LatticeMutation(
            "reasoning-smoother-budget",
            "reasoning",
            {"surprise_power": max(0.5, cfg.surprise_power * 0.85)},
            "spread recurrent compute more smoothly over uncertainty",
            ("reasoning",),
        ),
    ])

    # Sparse expert specialization.
    if cfg.n_experts < 32:
        mutations.append(LatticeMutation(
            "experts-more-specialists",
            "experts",
            {"n_experts": min(32, cfg.n_experts * 2)},
            "increase total specialization capacity while keeping top-k execution sparse",
            ("code", "reasoning"),
        ))
    if cfg.n_experts > 2 and cfg.n_experts // 2 >= cfg.active_experts:
        mutations.append(LatticeMutation(
            "experts-fewer-specialists",
            "experts",
            {"n_experts": max(2, cfg.n_experts // 2)},
            "reduce dormant capacity if specialization is not useful",
            ("efficiency",),
        ))
    if cfg.active_experts < min(4, cfg.n_experts):
        mutations.append(LatticeMutation(
            "experts-wider-topk",
            "experts",
            {"active_experts": cfg.active_experts + 1},
            "combine more specialist views for difficult tokens",
            ("reasoning",),
        ))
    if cfg.active_experts > 1:
        mutations.append(LatticeMutation(
            "experts-sparser-topk",
            "experts",
            {"active_experts": cfg.active_experts - 1},
            "reduce active compute while preserving total expert capacity",
            ("efficiency",),
        ))

    for factor, name in ((1.25, "experts-grow-width"), (0.8, "experts-shrink-width")):
        width = max(cfg.d_model, int(round(cfg.d_expert * factor / 16.0)) * 16)
        if width != cfg.d_expert and width <= 4096:
            mutations.append(LatticeMutation(
                name,
                "experts",
                {"d_expert": width},
                "evolve specialist and recurrent-reasoner capacity per active token",
                ("reasoning", "efficiency"),
            ))

    # Physical recurrent depth.
    if cfg.n_cells < 6:
        mutations.append(LatticeMutation(
            "cells-add",
            "depth",
            {"n_cells": cfg.n_cells + 1},
            "add another recurrent transformation/memory stage",
            ("reasoning",),
        ))
    if cfg.n_cells > 1:
        mutations.append(LatticeMutation(
            "cells-remove",
            "depth",
            {"n_cells": cfg.n_cells - 1},
            "test whether shared recurrence can replace physical depth",
            ("efficiency",),
        ))

    # Optimizer surface remains evolvable along with architecture.
    mutations.extend([
        LatticeMutation(
            "optimizer-lr-up",
            "optimizer",
            {
                "learning_rate": min(0.02, champion.learning_rate * 1.20),
                "min_learning_rate": min(0.01, champion.min_learning_rate * 1.10),
            },
            "increase learning speed if the architecture is under-updating",
        ),
        LatticeMutation(
            "optimizer-lr-down",
            "optimizer",
            {
                "learning_rate": max(1e-5, champion.learning_rate * 0.75),
                "min_learning_rate": max(0.0, champion.min_learning_rate * 0.75),
            },
            "favor stability if rapid updates damage held-out loss",
        ),
        LatticeMutation(
            "optimizer-less-decay",
            "optimizer",
            {"weight_decay": max(0.0, champion.weight_decay * 0.5)},
            "relax regularization when tiny scratch models are underfitting",
        ),
    ])

    # Bias ordering toward what MATHESIS/research currently says is weak.
    priority: dict[str, int] = {
        "memory": 2 if reasoning_focus else 1,
        "reasoning": 3 if reasoning_focus else 1,
        "routing": 2 if reasoning_focus else 1,
        "experts": 2 if reasoning_focus or curriculum_focus else 1,
        "prediction": 4 if reasoning_focus else 2,
        "local": 4 if reasoning_focus else 2,
        "optimizer": 1,
        "depth": 1,
    }
    if efficiency_focus:
        priority["memory"] += 1
        priority["routing"] += 1
        priority["experts"] += 1
    if tokenizer_focus:
        # Lattice v0 intentionally keeps tokenization outside the architecture
        # genome; Phase-2 BPE evolution handles it. Reward memory/efficiency
        # searches that can exploit shorter BPE sequences later.
        priority["memory"] += 1

    unique: list[LatticeMutation] = []
    seen: set[str] = set()
    for mutation in mutations:
        signature = _digest(mutation.to_dict())
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mutation)
    unique.sort(
        key=lambda row: (
            -priority.get(row.family, 0),
            row.family,
            row.name,
        )
    )
    return unique


def generate_lattice_population(
    champion: LatticeGenome,
    *,
    count: int = 8,
    exploration_offset: int = 0,
    mathesis_signals: Iterable[str] | None = None,
    research: dict[str, Any] | None = None,
    max_total_parameters: int = 8_000_000,
    max_active_parameter_ratio: float = 1.5,
) -> list[tuple[LatticeMutation, LatticeGenome]]:
    champion.validate()
    base_cost = architecture_cost_vector(champion.lattice_config())
    active_limit = float(base_cost["active_parameters"]) * max(
        1.0,
        float(max_active_parameter_ratio),
    )

    library = mutation_library(
        champion,
        mathesis_signals=mathesis_signals,
        research=research,
    )
    if library:
        # Round-robin mutation families instead of slicing one sorted list.
        # The first real swarm showed that a naive top-N population could be
        # dominated by near-identical memory mutations and never test routing,
        # reasoning or predictive/local structure. Keep one lane per family
        # before returning to a second mutation from the same family.
        buckets: dict[str, list[LatticeMutation]] = {}
        family_order: list[str] = []
        for mutation in library:
            if mutation.family not in buckets:
                buckets[mutation.family] = []
                family_order.append(mutation.family)
            buckets[mutation.family].append(mutation)

        family_shift = max(0, int(exploration_offset)) % len(family_order)
        family_order = family_order[family_shift:] + family_order[:family_shift]
        inner_shift = max(0, int(exploration_offset)) // max(1, len(family_order))
        for family in family_order:
            rows = buckets[family]
            if rows:
                shift = inner_shift % len(rows)
                buckets[family] = rows[shift:] + rows[:shift]

        interleaved: list[LatticeMutation] = []
        cursor = 0
        while True:
            added = False
            for family in family_order:
                rows = buckets[family]
                if cursor < len(rows):
                    interleaved.append(rows[cursor])
                    added = True
            if not added:
                break
            cursor += 1
        library = interleaved

    out: list[tuple[LatticeMutation, LatticeGenome]] = []
    seen: set[str] = set()
    for mutation in library:
        try:
            candidate = apply_lattice_mutation(champion, mutation)
            cfg = candidate.lattice_config()
            certificate = stability_certificate(cfg)
            cost = architecture_cost_vector(cfg)
        except Exception:
            continue
        if not certificate["ok"]:
            continue
        if cost["total_parameters"] > int(max_total_parameters):
            continue
        if cost["active_parameters"] > active_limit:
            continue
        signature = _digest({
            "config": candidate.config,
            "learning_rate": candidate.learning_rate,
            "min_learning_rate": candidate.min_learning_rate,
            "weight_decay": candidate.weight_decay,
        })
        if signature in seen:
            continue
        seen.add(signature)
        out.append((mutation, candidate))
        if len(out) >= max(1, int(count)):
            break
    return out


def successive_halving_plan(
    population_size: int,
    *,
    first_stage_steps: int = 2,
    stages: int = 3,
    growth_factor: int = 3,
    survival_fraction: float = 0.5,
) -> list[dict[str, int]]:
    """Return a compute-efficient tournament schedule.

    Many candidates receive a tiny budget. Only survivors receive more compute.
    This is intentionally simple enough to reproduce in GitHub Actions.
    """
    remaining = max(1, int(population_size))
    steps = max(1, int(first_stage_steps))
    out = []
    for stage in range(max(1, int(stages))):
        out.append({
            "stage": stage,
            "candidates": remaining,
            "steps": steps,
        })
        if remaining <= 1:
            break
        remaining = max(
            1,
            int(math.ceil(remaining * max(0.05, min(1.0, survival_fraction)))),
        )
        steps *= max(2, int(growth_factor))
    return out
