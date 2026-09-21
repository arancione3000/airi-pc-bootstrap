from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .lattice_evolution import LatticeGenome
from .lattice_math import pareto_front, stability_certificate


LATTICE_ELITE_ARCHIVE_VERSION = "airi-lattice-elite-archive-v1"


def _finite(value: Any, default: float = float("inf")) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def load_elite_archive(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {
            "version": LATTICE_ELITE_ARCHIVE_VERSION,
            "elites": [],
        }
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    rows = raw.get("elites")
    if not isinstance(rows, list):
        rows = []
    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            genome = LatticeGenome(**dict(row["genome"])).validate()
        except Exception:
            continue
        if not stability_certificate(genome.lattice_config()).get("ok"):
            continue
        clean = dict(row)
        clean["genome"] = genome.to_dict()
        valid.append(clean)
    return {
        "version": LATTICE_ELITE_ARCHIVE_VERSION,
        "elites": valid,
    }


def _report_evidence(row: dict[str, Any]) -> dict[str, Any] | None:
    reports = row.get("reports")
    if not isinstance(reports, list) or not reports:
        return None

    candidate_losses: list[float] = []
    baseline_losses: list[float] = []
    wins = 0
    valid_reports = 0
    for report in reports:
        if not isinstance(report, dict) or not report.get("ok"):
            continue
        try:
            candidate_loss = float(
                report["candidate"]["training"]["final"]["loss"]
            )
            baseline_loss = float(
                report["baseline"]["training"]["final"]["loss"]
            )
        except Exception:
            continue
        if not (math.isfinite(candidate_loss) and math.isfinite(baseline_loss)):
            continue
        valid_reports += 1
        candidate_losses.append(candidate_loss)
        baseline_losses.append(baseline_loss)
        if bool(report.get("candidate_wins")):
            wins += 1

    if valid_reports <= 0:
        return None

    try:
        genome = LatticeGenome(**dict(row["genome"])).validate()
    except Exception:
        return None
    if not stability_certificate(genome.lattice_config()).get("ok"):
        return None

    margins = [
        baseline - candidate
        for baseline, candidate in zip(baseline_losses, candidate_losses)
    ]
    mutation = row.get("mutation") if isinstance(row.get("mutation"), dict) else {}
    return {
        "candidate_id": str(row.get("candidate_id") or genome.genome_id),
        "cycle": int(row.get("cycle", 0) or 0),
        "stage": int(row.get("stage", 0) or 0),
        "mutation": dict(mutation),
        "mutation_family": str(mutation.get("family") or "unknown"),
        "genome": genome.to_dict(),
        "seed_wins": wins,
        "seed_count": valid_reports,
        "win_fraction": wins / valid_reports,
        "mean_loss": mean(candidate_losses),
        "mean_baseline_loss": mean(baseline_losses),
        "mean_margin": mean(margins),
        "worst_margin": min(margins),
        "active_parameters": _finite(row.get("active_parameters")),
        "state_bytes": _finite(row.get("state_bytes")),
        "train_seconds": _finite(row.get("train_seconds")),
        "all_seed_wins": wins == valid_reports,
    }


def update_elite_archive(
    archive: dict[str, Any] | None,
    final_reports: Iterable[dict[str, Any]],
    *,
    champion_id: str | None = None,
    max_elites: int = 12,
) -> dict[str, Any]:
    pool: dict[str, dict[str, Any]] = {}
    for row in (archive or {}).get("elites", []):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id") or "")
        if cid:
            pool[cid] = dict(row)

    for row in final_reports:
        if not isinstance(row, dict) or not row.get("ok"):
            continue
        evidence = _report_evidence(row)
        if evidence is None:
            continue
        cid = evidence["candidate_id"]
        if champion_id and cid == champion_id:
            continue
        previous = pool.get(cid)
        if previous is None:
            pool[cid] = evidence
            continue
        # Keep the observation with more independent seeds; break ties by
        # stronger average margin against the matched Transformer.
        old_key = (
            int(previous.get("seed_count", 0) or 0),
            _finite(previous.get("mean_margin"), default=-float("inf")),
        )
        new_key = (
            int(evidence.get("seed_count", 0) or 0),
            _finite(evidence.get("mean_margin"), default=-float("inf")),
        )
        if new_key >= old_key:
            pool[cid] = evidence

    rows = list(pool.values())
    # A near-winner can be valuable even when its raw loss is not the very
    # lowest, so the archive optimizes several axes simultaneously.
    metrics = []
    for row in rows:
        metric = dict(row)
        metric["negative_win_fraction"] = -float(row.get("win_fraction", 0.0))
        metric["negative_margin"] = -float(row.get("mean_margin", -1e9))
        metrics.append(metric)

    front = pareto_front(
        metrics,
        minimize=(
            "negative_win_fraction",
            "negative_margin",
            "mean_loss",
            "active_parameters",
            "state_bytes",
        ),
    )
    front_ids = {str(row.get("candidate_id")) for row in front}

    ordered = sorted(
        rows,
        key=lambda row: (
            0 if str(row.get("candidate_id")) in front_ids else 1,
            -float(row.get("win_fraction", 0.0)),
            -float(row.get("mean_margin", -1e9)),
            float(row.get("mean_loss", float("inf"))),
            float(row.get("active_parameters", float("inf"))),
            str(row.get("candidate_id")),
        ),
    )
    selected = ordered[: max(1, int(max_elites))]
    return {
        "version": LATTICE_ELITE_ARCHIVE_VERSION,
        "elites": selected,
        "summary": {
            "count": len(selected),
            "pareto_count": sum(
                1 for row in selected
                if str(row.get("candidate_id")) in front_ids
            ),
            "near_winners": sum(
                1 for row in selected
                if float(row.get("win_fraction", 0.0)) > 0.0
            ),
        },
    }


def elite_parent_genomes(
    archive: dict[str, Any] | None,
    *,
    champion_id: str,
    max_parents: int = 2,
) -> list[LatticeGenome]:
    rows = list((archive or {}).get("elites") or [])
    rows.sort(
        key=lambda row: (
            -float(row.get("win_fraction", 0.0)),
            -float(row.get("mean_margin", -1e9)),
            float(row.get("mean_loss", float("inf"))),
            str(row.get("candidate_id")),
        )
    )
    parents: list[LatticeGenome] = []
    seen = {str(champion_id)}
    for row in rows:
        try:
            genome = LatticeGenome(**dict(row["genome"])).validate()
        except Exception:
            continue
        if genome.genome_id in seen:
            continue
        seen.add(genome.genome_id)
        parents.append(genome)
        if len(parents) >= max(0, int(max_parents)):
            break
    return parents


def mutation_family_feedback(
    archive: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for row in (archive or {}).get("elites", []):
        if not isinstance(row, dict):
            continue
        family = str(row.get("mutation_family") or "unknown")
        bucket = stats.setdefault(
            family,
            {
                "trials": 0.0,
                "seed_wins": 0.0,
                "seeds": 0.0,
                "margin_sum": 0.0,
            },
        )
        bucket["trials"] += 1.0
        bucket["seed_wins"] += float(row.get("seed_wins", 0) or 0)
        bucket["seeds"] += float(row.get("seed_count", 0) or 0)
        bucket["margin_sum"] += float(row.get("mean_margin", 0.0) or 0.0)

    result: dict[str, dict[str, float]] = {}
    for family, bucket in stats.items():
        trials = max(1.0, bucket["trials"])
        seeds = max(1.0, bucket["seeds"])
        result[family] = {
            "trials": bucket["trials"],
            "win_rate": bucket["seed_wins"] / seeds,
            "mean_margin": bucket["margin_sum"] / trials,
        }
    return result
