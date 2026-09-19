from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Intent:
    kind: str
    raw: str
    expression: str | None = None
    variable: str | None = None
    target: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProofCell:
    cell_id: str
    goal: str
    assumptions: list[str] = field(default_factory=list)
    strategy: str = "unassigned"
    status: str = "pending"
    result: Any = None
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationCertificate:
    ok: bool
    status: str
    statement: str
    methods: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)
    counterexample: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgramCandidate:
    name: str
    task: str
    code: str
    tests_passed: int
    tests_total: int
    score: float
    verified: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectureGenome:
    generation: int = 0
    experts: list[str] = field(default_factory=lambda: [
        "formalization",
        "algebra",
        "counterexample",
        "program_synthesis",
    ])
    proof_order: list[str] = field(default_factory=lambda: [
        "symbolic",
        "smt",
        "counterexample",
        "lean",
    ])
    counterexample_radius: int = 8
    max_proof_cells: int = 64
    neural_hidden: int = 16
    parent_id: str | None = None
    genome_id: str = "omega-g0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArchitectureGenome":
        return cls(
            generation=int(value.get("generation", 0)),
            experts=[str(x) for x in value.get("experts", [])],
            proof_order=[str(x) for x in value.get("proof_order", [])],
            counterexample_radius=int(value.get("counterexample_radius", 8)),
            max_proof_cells=int(value.get("max_proof_cells", 64)),
            neural_hidden=int(value.get("neural_hidden", 16)),
            parent_id=value.get("parent_id"),
            genome_id=str(value.get("genome_id", "omega-g0")),
        )


@dataclass(frozen=True)
class EvolutionResult:
    promoted: bool
    champion: dict[str, Any]
    candidate: dict[str, Any]
    champion_score: float
    candidate_score: float
    reason: str
    benchmark: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
