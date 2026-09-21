from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable


LATTICE_SCALE_GATE_VERSION = "airi-lattice-scale-gate-v1"


@dataclass(frozen=True)
class LatticeMigrationEvidence:
    genome_id: str
    winning_cycles: int
    seed_wins: int
    seed_count: int
    mean_margin: float
    worst_margin: float
    max_active_parameter_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "winning_cycles": self.winning_cycles,
            "seed_wins": self.seed_wins,
            "seed_count": self.seed_count,
            "mean_margin": self.mean_margin,
            "worst_margin": self.worst_margin,
            "max_active_parameter_ratio": self.max_active_parameter_ratio,
        }


def _history_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _candidate_evidence_for_genome(
    history: Iterable[dict[str, Any]],
    genome_id: str,
) -> LatticeMigrationEvidence:
    cycles: set[int] = set()
    margins: list[float] = []
    seed_wins = 0
    seed_count = 0
    active_ratios: list[float] = []

    for state in history:
        cycle = int(state.get("cycle", 0) or 0)
        reports = state.get("final_reports")
        if not isinstance(reports, list):
            continue
        for row in reports:
            if not isinstance(row, dict):
                continue
            if str(row.get("candidate_id") or "") != str(genome_id):
                continue
            subreports = row.get("reports")
            if not isinstance(subreports, list):
                continue
            valid_here = 0
            wins_here = 0
            for report in subreports:
                if not isinstance(report, dict) or not report.get("ok"):
                    continue
                try:
                    candidate_loss = float(
                        report["candidate"]["training"]["final"]["loss"]
                    )
                    baseline_loss = float(
                        report["baseline"]["training"]["final"]["loss"]
                    )
                    candidate_active = float(
                        report["candidate"]["active_parameters"]
                    )
                    baseline_active = float(
                        report["baseline"]["active_parameters"]
                    )
                except Exception:
                    continue
                if not all(
                    math.isfinite(value)
                    for value in (
                        candidate_loss,
                        baseline_loss,
                        candidate_active,
                        baseline_active,
                    )
                ):
                    continue
                valid_here += 1
                seed_count += 1
                margin = baseline_loss - candidate_loss
                margins.append(margin)
                active_ratios.append(
                    candidate_active / max(1.0, baseline_active)
                )
                if bool(report.get("candidate_wins")):
                    wins_here += 1
                    seed_wins += 1
            if valid_here and wins_here == valid_here:
                cycles.add(cycle)

    return LatticeMigrationEvidence(
        genome_id=str(genome_id),
        winning_cycles=len(cycles),
        seed_wins=seed_wins,
        seed_count=seed_count,
        mean_margin=(
            sum(margins) / len(margins)
            if margins
            else -float("inf")
        ),
        worst_margin=min(margins) if margins else -float("inf"),
        max_active_parameter_ratio=(
            max(active_ratios)
            if active_ratios
            else float("inf")
        ),
    )


def evaluate_lattice_migration_readiness(
    history_path: str | Path,
    champion: dict[str, Any],
    *,
    min_winning_cycles: int = 3,
    min_seed_wins: int = 6,
    min_total_seeds: int = 6,
    min_mean_margin: float = 0.002,
    min_worst_margin: float = 0.0,
    max_active_parameter_ratio: float = 1.20,
) -> dict[str, Any]:
    """Produce a fail-closed readiness attestation for canonical migration.

    This never rewrites the canonical model. It only says whether a specific
    Lattice research genome has accumulated enough independent evidence that a
    separate scale-transfer qualification would be justified.
    """
    genome_id = str(champion.get("genome_id") or "").strip()
    if not genome_id:
        raise ValueError("Lattice migration gate requires champion genome_id")

    evidence = _candidate_evidence_for_genome(
        _history_rows(history_path),
        genome_id,
    )
    checks = {
        "winning_cycles": evidence.winning_cycles >= int(min_winning_cycles),
        "seed_wins": evidence.seed_wins >= int(min_seed_wins),
        "total_seeds": evidence.seed_count >= int(min_total_seeds),
        "mean_margin": evidence.mean_margin >= float(min_mean_margin),
        "worst_margin": evidence.worst_margin >= float(min_worst_margin),
        "active_parameter_ratio": (
            evidence.max_active_parameter_ratio
            <= float(max_active_parameter_ratio)
        ),
    }
    ready = all(checks.values())
    return {
        "ok": True,
        "version": LATTICE_SCALE_GATE_VERSION,
        "migration_ready": ready,
        "genome_id": genome_id,
        "evidence": evidence.to_dict(),
        "requirements": {
            "min_winning_cycles": int(min_winning_cycles),
            "min_seed_wins": int(min_seed_wins),
            "min_total_seeds": int(min_total_seeds),
            "min_mean_margin": float(min_mean_margin),
            "min_worst_margin": float(min_worst_margin),
            "max_active_parameter_ratio": float(max_active_parameter_ratio),
        },
        "checks": checks,
        "canonical_model_changed": False,
        "next_gate": (
            "scale-transfer qualification on larger corpus/budgets"
            if ready
            else "continue architecture research"
        ),
    }
