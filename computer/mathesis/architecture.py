from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from .types import ArchitectureGenome


_ALLOWED_EXPERTS = (
    "formalization",
    "algebra",
    "counterexample",
    "program_synthesis",
    "number_theory",
    "research",
    "geometry",
    "optimization",
)


def default_genome() -> ArchitectureGenome:
    return ArchitectureGenome()


def genome_fingerprint(genome: ArchitectureGenome) -> str:
    payload = json.dumps(genome.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def mutate_genome(champion: ArchitectureGenome) -> ArchitectureGenome:
    generation = champion.generation + 1
    experts = list(dict.fromkeys(champion.experts))

    missing = [expert for expert in _ALLOWED_EXPERTS if expert not in experts]
    if missing:
        experts.append(missing[0])

    proof_order = list(champion.proof_order)
    if generation % 3 == 0 and "counterexample" in proof_order:
        proof_order.remove("counterexample")
        proof_order.insert(1, "counterexample")

    candidate = replace(
        champion,
        generation=generation,
        experts=experts,
        proof_order=proof_order,
        counterexample_radius=min(24, champion.counterexample_radius + (1 if generation % 2 else 0)),
        neural_hidden=min(96, champion.neural_hidden + 4),
        parent_id=champion.genome_id,
        genome_id="pending",
    )
    candidate.genome_id = f"omega-{generation}-{genome_fingerprint(candidate)[:8]}"
    return candidate


def architecture_report(genome: ArchitectureGenome) -> dict[str, Any]:
    return {
        "genome": genome.to_dict(),
        "fingerprint": genome_fingerprint(genome),
        "proof_cells": {
            "communication": [
                "goal",
                "assumptions",
                "proof_obligations",
                "candidate_result",
                "verification_status",
            ],
            "max_cells": genome.max_proof_cells,
        },
        "neural_router": {
            "kind": "growing_pure_python_mlp",
            "hidden_units": genome.neural_hidden,
            "growth_is_champion_challenger_gated": True,
        },
        "immutable_kernel": [
            "safe_math.py",
            "verifiers.py",
            "kernel.py",
        ],
    }
