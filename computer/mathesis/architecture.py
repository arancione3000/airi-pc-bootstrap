from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from .types import ArchitectureGenome


_ALLOWED_EXPERTS = (
    "formalization",
    "algebra",
    "polynomials",
    "counterexample",
    "program_synthesis",
    "number_theory",
    "research",
    "calculus",
    "trigonometry",
    "linear_algebra",
    "combinatorics",
    "equations",
    "inequalities",
    "sequences",
    "special_functions",
    "geometry",
    "probability",
    "discrete_math",
    "optimization",
)

_ALLOWED_STRATEGIES = (
    "exact_symbolic",
    "specific_normal_forms",
    "smt_validity",
    "counterexample",
    "lean",
    "conjecture_discovery",
    "finite_difference",
    "exact_sampling",
    "read_only_research",
)


def default_genome() -> ArchitectureGenome:
    return ArchitectureGenome()


def genome_fingerprint(genome: ArchitectureGenome) -> str:
    payload = json.dumps(genome.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _topology_for(experts: list[str]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    if not experts:
        return edges
    for left, right in zip(experts, experts[1:]):
        edges.append((left, right))
    if "counterexample" in experts:
        for expert in experts:
            if expert not in {"formalization", "counterexample"}:
                edge = (expert, "counterexample")
                if edge not in edges:
                    edges.append(edge)
    return edges[:64]


def mutate_genome(
    champion: ArchitectureGenome,
    *,
    variant: int = 0,
    weaknesses: list[str] | None = None,
) -> ArchitectureGenome:
    """Create a bounded architecture challenger.

    Mutation is structural but expressed in a safe architecture DSL. It may
    change experts, graph topology and search budgets; it never rewrites the
    verifier kernel itself.
    """

    generation = champion.generation + 1
    experts = list(dict.fromkeys(champion.experts))
    weaknesses = [str(x) for x in (weaknesses or [])]

    # Prefer an expert corresponding to an observed weak benchmark domain.
    requested_candidates: list[str] = []
    for weakness in weaknesses:
        for expert in _ALLOWED_EXPERTS:
            if expert in weakness and expert not in experts and expert not in requested_candidates:
                requested_candidates.append(expert)

    missing = [expert for expert in _ALLOWED_EXPERTS if expert not in experts]
    if requested_candidates:
        experts.append(requested_candidates[variant % len(requested_candidates)])
    elif missing:
        experts.append(missing[variant % len(missing)])

    portfolio = list(dict.fromkeys(champion.strategy_portfolio))
    strategy_missing = [s for s in _ALLOWED_STRATEGIES if s not in portfolio]
    if strategy_missing and (variant % 2 == 1 or not missing):
        portfolio.append(strategy_missing[0])

    # Three mutation styles are used by the arena:
    #   0 coverage, 1 proof/search depth, 2 efficiency.
    hidden = champion.neural_hidden
    symbolic_depth = champion.symbolic_depth
    discovery_beam = champion.discovery_beam
    research_budget = champion.research_budget
    max_cells = champion.max_proof_cells
    radius = champion.counterexample_radius

    if variant % 3 == 0:
        hidden = min(128, hidden + 4)
        radius = min(40, radius + 1)
    elif variant % 3 == 1:
        symbolic_depth = min(12, symbolic_depth + 1)
        discovery_beam = min(8, discovery_beam + 1)
        max_cells = min(256, max_cells + 16)
    else:
        # Try to become smaller/faster without losing capabilities.
        hidden = max(12, hidden - 2)
        research_budget = min(4, research_budget + (1 if "research" in experts else 0))

    candidate = replace(
        champion,
        generation=generation,
        experts=experts,
        proof_order=list(champion.proof_order),
        counterexample_radius=radius,
        max_proof_cells=max_cells,
        neural_hidden=hidden,
        symbolic_depth=symbolic_depth,
        discovery_beam=discovery_beam,
        research_budget=research_budget,
        strategy_portfolio=portfolio,
        topology=_topology_for(experts),
        parent_id=champion.genome_id,
        genome_id="pending",
    )
    candidate.genome_id = f"omega-{generation}-{genome_fingerprint(candidate)[:8]}"
    return candidate


def generate_challengers(
    champion: ArchitectureGenome,
    *,
    weaknesses: list[str] | None = None,
    count: int = 3,
) -> list[ArchitectureGenome]:
    return [
        mutate_genome(champion, variant=index, weaknesses=weaknesses)
        for index in range(max(1, min(6, int(count))))
    ]


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
        "architecture_graph": {
            "nodes": list(genome.experts),
            "edges": [list(edge) for edge in genome.topology],
            "strategy_portfolio": list(genome.strategy_portfolio),
            "symbolic_depth": genome.symbolic_depth,
            "discovery_beam": genome.discovery_beam,
            "research_budget": genome.research_budget,
        },
        "immutable_kernel": [
            "safe_math.py",
            "verifiers.py",
            "kernel.py",
        ],
    }
