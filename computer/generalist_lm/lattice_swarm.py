from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .lattice_evolution import (
    LatticeGenome,
    generate_lattice_population,
    root_lattice_genome,
)
from .lattice_lab import (
    LatticeLabConfig,
    benchmark_lattice_against_transformer,
)
from .lattice_math import (
    architecture_cost_vector,
    pareto_front,
    stability_certificate,
)
from .lattice_meta import (
    elite_parent_genomes,
    load_elite_archive,
    mutation_family_feedback,
    update_elite_archive,
)
from .lattice_scale_gate import evaluate_lattice_migration_readiness
from .native_lattice import AiriLatticeConfig


LATTICE_SWARM_VERSION = "airi-lattice-swarm-v1"
STATE_VERSION = "airi-lattice-state-v1"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_champion(state_dir: Path) -> LatticeGenome:
    path = state_dir / "lattice-champion.json"
    if path.is_file():
        raw = _load_json(path)
        try:
            return LatticeGenome(**raw).validate()
        except Exception:
            pass
    cfg = AiriLatticeConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_cells=1,
        memory_bands=3,
        d_expert=48,
        n_experts=4,
        active_experts=1,
        max_reasoning_steps=2,
        surprise_threshold=0.25,
        tokenizer_version="byte-v1",
    ).validate()
    return root_lattice_genome(cfg)


def prepare_swarm_plan(
    state_dir: str | Path,
    output_path: str | Path,
    *,
    cycle: int,
    mathesis_signals: Iterable[str] | None = None,
    research: dict[str, Any] | None = None,
    population_size: int = 8,
    exploration_offset: int = 0,
    max_total_parameters: int = 5_000_000,
    max_active_parameter_ratio: float = 1.5,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    champion = _load_champion(root)
    elite_archive = load_elite_archive(root / "lattice-elite-archive.json")
    elite_parents = elite_parent_genomes(
        elite_archive,
        champion_id=champion.genome_id,
        max_parents=2,
    )
    parents = [("champion", champion)] + [
        ("elite", genome) for genome in elite_parents
    ]

    total = max(1, int(population_size))
    base_quota = total // len(parents)
    remainder = total % len(parents)
    candidates = []
    seen_genomes: set[str] = set()
    for parent_index, (parent_kind, parent) in enumerate(parents):
        quota = base_quota + (1 if parent_index < remainder else 0)
        if quota <= 0:
            continue
        population = generate_lattice_population(
            parent,
            count=quota,
            exploration_offset=max(0, int(exploration_offset)) + parent_index,
            mathesis_signals=mathesis_signals,
            research=research,
            max_total_parameters=max(1, int(max_total_parameters)),
            max_active_parameter_ratio=max(1.0, float(max_active_parameter_ratio)),
        )
        for mutation, genome in population:
            if genome.genome_id in seen_genomes:
                continue
            seen_genomes.add(genome.genome_id)
            cfg = genome.lattice_config()
            candidates.append({
                "index": len(candidates),
                "candidate_id": genome.genome_id,
                "mutation": mutation.to_dict(),
                "genome": genome.to_dict(),
                "source_parent_id": parent.genome_id,
                "source_parent_generation": parent.generation,
                "source_parent_kind": parent_kind,
                "cost": architecture_cost_vector(cfg),
                "stability": stability_certificate(cfg),
            })
            if len(candidates) >= total:
                break
        if len(candidates) >= total:
            break

    incumbent_cfg = champion.lattice_config()
    incumbent = {
        "index": 9999,
        "candidate_id": champion.genome_id,
        "mutation": {
            "name": "incumbent-revalidation",
            "family": "incumbent",
            "changes": {},
            "rationale": "re-validate the current research champion on fresh independent seeds",
            "mathesis_tags": [],
        },
        "genome": champion.to_dict(),
        "source_parent_id": champion.genome_id,
        "source_parent_generation": champion.generation,
        "source_parent_kind": "incumbent",
        "cost": architecture_cost_vector(incumbent_cfg),
        "stability": stability_certificate(incumbent_cfg),
    }

    plan = {
        "ok": bool(candidates),
        "version": LATTICE_SWARM_VERSION,
        "cycle": int(cycle),
        "champion": champion.to_dict(),
        "signals": list(dict.fromkeys(str(row) for row in (mathesis_signals or ()))),
        "research": {
            "tag_counts": dict((research or {}).get("tag_counts") or {}),
            "evidence_digest": (research or {}).get("evidence_digest"),
        },
        "elite_archive": {
            "summary": dict(elite_archive.get("summary") or {}),
            "parents": [
                {
                    "genome_id": genome.genome_id,
                    "generation": genome.generation,
                }
                for genome in elite_parents
            ],
            "family_feedback": mutation_family_feedback(elite_archive),
        },
        "candidates": candidates,
        "incumbent": incumbent,
        "matrix": {"include": [{"index": row["index"]} for row in candidates]},
        "policy": {
            "parallel_candidates": True,
            "external_pretrained": False,
            "canonical_native_transformer_unchanged": True,
            "candidate_can_self_promote": False,
            "elite_can_self_promote": False,
            "elite_archive_is_research_only": True,
        },
    }
    _atomic_json(Path(output_path), plan)
    if not candidates:
        raise RuntimeError("AIRI Lattice swarm produced no stable candidates")
    return plan


def _candidate_from_plan(plan: dict[str, Any], index: int) -> tuple[dict[str, Any], LatticeGenome]:
    incumbent = plan.get("incumbent")
    if isinstance(incumbent, dict) and int(incumbent.get("index", -1)) == int(index):
        genome = LatticeGenome(**dict(incumbent["genome"])).validate()
        return incumbent, genome

    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("invalid Lattice swarm plan candidates")
    for row in candidates:
        if int(row.get("index", -1)) == int(index):
            genome = LatticeGenome(**dict(row["genome"])).validate()
            return row, genome
    raise IndexError(f"Lattice swarm candidate index {index} not found")


def run_candidate(
    plan_path: str | Path,
    corpus_manifest: str | Path,
    output_path: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    candidate_index: int,
    stage: int,
    steps: int,
    repeat_seeds: int = 1,
    max_eval_blocks: int = 4,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    row, genome = _candidate_from_plan(plan, candidate_index)
    reports = []
    cycle = int(plan.get("cycle", 0))
    for repeat in range(max(1, int(repeat_seeds))):
        seed = (
            9_000_001
            + cycle * 100_003
            + int(stage) * 10_007
            + int(candidate_index) * 101
            + repeat
        )
        try:
            report = benchmark_lattice_against_transformer(
                str(corpus_manifest),
                allowed_roots=[str(Path(item)) for item in allowed_roots],
                lattice_config=genome.lattice_config(),
                lab_config=LatticeLabConfig(
                    steps=max(1, int(steps)),
                    batch_size=1,
                    learning_rate=genome.learning_rate,
                    min_learning_rate=genome.min_learning_rate,
                    weight_decay=genome.weight_decay,
                    validation_fraction=0.20,
                    max_eval_blocks=max(1, int(max_eval_blocks)),
                    seed=seed,
                    device="cpu",
                    minimum_loss_gain=0.001,
                    max_domain_regression=0.08,
                    max_active_parameter_ratio=1.20,
                ),
            )
        except Exception as exc:
            report = {
                "ok": False,
                "candidate_wins": False,
                "decision": f"benchmark_error:{type(exc).__name__}:{exc}",
            }
        reports.append(report)

    valid = [
        report for report in reports
        if report.get("ok")
        and isinstance(report.get("candidate"), dict)
        and isinstance(report["candidate"].get("training"), dict)
    ]
    losses = [
        float(report["candidate"]["training"]["final"]["loss"])
        for report in valid
    ]
    train_seconds = [
        float(report["candidate"]["training"]["train_seconds"])
        for report in valid
    ]
    active_parameters = (
        float(valid[0]["candidate"]["active_parameters"])
        if valid
        else float("inf")
    )
    state_bytes = (
        float(valid[0]["candidate"]["state_bytes_at_context"])
        if valid
        else float("inf")
    )
    result = {
        "ok": len(valid) == len(reports) and bool(valid),
        "version": LATTICE_SWARM_VERSION,
        "cycle": cycle,
        "stage": int(stage),
        "steps": int(steps),
        "candidate_index": int(candidate_index),
        "candidate_id": row["candidate_id"],
        "mutation": row["mutation"],
        "genome": genome.to_dict(),
        "source_parent_kind": row.get("source_parent_kind"),
        "reports": reports,
        "all_seed_wins": bool(reports) and all(
            bool(report.get("ok") and report.get("candidate_wins"))
            for report in reports
        ),
        "loss": mean(losses) if losses else float("inf"),
        "active_parameters": active_parameters,
        "state_bytes": state_bytes,
        "train_seconds": mean(train_seconds) if train_seconds else float("inf"),
        "repeat_seeds": len(reports),
        "external_pretrained": False,
    }
    _atomic_json(Path(output_path), result)
    return result


def _finite_metric(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except Exception:
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def _load_result_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            candidates = sorted(p.rglob("*.json"))
        elif p.is_file():
            candidates = [p]
        else:
            continue
        for candidate in candidates:
            try:
                row = _load_json(candidate)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("version") == LATTICE_SWARM_VERSION:
                rows.append(row)
    unique: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row.get("candidate_index", -1))
        if index < 0:
            continue
        existing = unique.get(index)
        if existing is None or int(row.get("stage", -1)) > int(existing.get("stage", -1)):
            unique[index] = row
    return list(unique.values())


def select_survivors(
    result_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    survivor_count: int,
) -> dict[str, Any]:
    rows = [row for row in _load_result_files(result_paths) if row.get("ok")]
    if not rows:
        raise RuntimeError("no valid Lattice swarm results to reduce")

    metrics = []
    for row in rows:
        metrics.append({
            **row,
            "loss": _finite_metric(row, "loss"),
            "active_parameters": _finite_metric(row, "active_parameters"),
            "state_bytes": _finite_metric(row, "state_bytes"),
            "train_seconds": _finite_metric(row, "train_seconds"),
        })
    front = pareto_front(
        metrics,
        minimize=("loss", "active_parameters", "state_bytes", "train_seconds"),
    )
    ordered = sorted(
        front,
        key=lambda row: (
            row["loss"],
            row["active_parameters"],
            row["state_bytes"],
            row["train_seconds"],
            row["candidate_index"],
        ),
    )
    seen = {int(row["candidate_index"]) for row in ordered}
    if len(ordered) < survivor_count:
        rest = sorted(
            (row for row in metrics if int(row["candidate_index"]) not in seen),
            key=lambda row: (
                row["loss"],
                row["active_parameters"],
                row["state_bytes"],
                row["train_seconds"],
                row["candidate_index"],
            ),
        )
        ordered.extend(rest)
    selected = ordered[: max(1, int(survivor_count))]
    payload = {
        "ok": True,
        "version": LATTICE_SWARM_VERSION,
        "selected": [
            {
                "index": int(row["candidate_index"]),
                "candidate_id": row["candidate_id"],
            }
            for row in selected
        ],
        "matrix": {
            "include": [
                {"index": int(row["candidate_index"])}
                for row in selected
            ]
        },
        "pareto_candidate_ids": [row["candidate_id"] for row in front],
    }
    _atomic_json(Path(output_path), payload)
    return payload


def finalize_swarm(
    state_dir: str | Path,
    plan_path: str | Path,
    result_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = _load_json(plan_path)
    reports = [row for row in _load_result_files(result_paths) if row.get("ok")]
    incumbent_id = str((plan.get("incumbent") or {}).get("candidate_id") or "")
    finalists = [
        row for row in reports
        if row.get("all_seed_wins")
        and str(row.get("candidate_id") or "") != incumbent_id
    ]

    archive_path = root / "lattice-elite-archive.json"
    elite_archive = update_elite_archive(
        load_elite_archive(archive_path),
        reports,
        champion_id=str(plan.get("champion", {}).get("genome_id") or ""),
        max_elites=12,
    )
    _atomic_json(archive_path, elite_archive)

    promoted = False
    winner = None
    reason = "no finalist beat the matched scratch Transformer on every final seed"
    if finalists:
        finalists.sort(
            key=lambda row: (
                _finite_metric(row, "loss"),
                _finite_metric(row, "active_parameters"),
                _finite_metric(row, "state_bytes"),
                _finite_metric(row, "train_seconds"),
                int(row["candidate_index"]),
            )
        )
        best = finalists[0]
        genome = LatticeGenome(**dict(best["genome"])).validate()
        # Re-check the mathematical stability boundary in the external reducer.
        certificate = stability_certificate(genome.lattice_config())
        if not certificate["ok"]:
            raise RuntimeError("final Lattice candidate left the stability region")
        _atomic_json(root / "lattice-champion.json", genome.to_dict())
        promoted = True
        winner = {
            "candidate_id": best["candidate_id"],
            "candidate_index": best["candidate_index"],
            "mutation": best["mutation"],
            "genome": genome.to_dict(),
            "loss": best["loss"],
            "active_parameters": best["active_parameters"],
            "state_bytes": best["state_bytes"],
            "train_seconds": best["train_seconds"],
        }
        reason = "candidate passed the final multi-seed equal-budget Transformer gate"
    elif not (root / "lattice-champion.json").is_file():
        champion = LatticeGenome(**dict(plan["champion"])).validate()
        _atomic_json(root / "lattice-champion.json", champion.to_dict())

    champion = _load_champion(root)
    status = {
        "ok": True,
        "version": STATE_VERSION,
        "swarm_version": LATTICE_SWARM_VERSION,
        "cycle": int(plan.get("cycle", 0)),
        "promoted": promoted,
        "promotion_reason": reason,
        "winner": winner,
        "champion": champion.to_dict(),
        "final_reports": reports,
        "elite_archive": {
            "summary": dict(elite_archive.get("summary") or {}),
            "family_feedback": mutation_family_feedback(elite_archive),
            "top": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "generation": (row.get("genome") or {}).get("generation"),
                    "mutation_family": row.get("mutation_family"),
                    "win_fraction": row.get("win_fraction"),
                    "mean_margin": row.get("mean_margin"),
                }
                for row in list(elite_archive.get("elites") or [])[:5]
            ],
        },
        "policy": {
            "external_pretrained": False,
            "candidate_can_self_promote": False,
            "external_reducer": True,
            "equal_budget_transformer_gate_required": True,
            "final_multi_seed_gate_required": True,
            "canonical_native_transformer_unchanged": True,
            "elite_archive_is_research_only": True,
            "elite_can_self_promote": False,
        },
    }
    status["migration_readiness"] = evaluate_lattice_migration_readiness(
        root / "lattice-history.jsonl",
        champion.to_dict(),
        additional_history=[status],
    )
    _atomic_json(
        root / "lattice-migration-readiness.json",
        status["migration_readiness"],
    )
    _atomic_json(root / "lattice-status.json", status)
    with (root / "lattice-history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_json(Path(output_path), status)
    return status


def _cmd_prepare(args) -> dict[str, Any]:
    research = _load_json(args.research_json) if args.research_json else {}
    signals = []
    if args.signals_json:
        raw = _load_json(args.signals_json)
        if isinstance(raw, dict):
            signals = list(raw.get("signals") or [])
        elif isinstance(raw, list):
            signals = raw
    return prepare_swarm_plan(
        args.state_dir,
        args.output,
        cycle=args.cycle,
        mathesis_signals=signals,
        research=research,
        population_size=args.population_size,
        exploration_offset=args.exploration_offset,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AIRI Lattice parallel architecture swarm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("state_dir")
    prepare.add_argument("output")
    prepare.add_argument("--cycle", type=int, required=True)
    prepare.add_argument("--signals-json")
    prepare.add_argument("--research-json")
    prepare.add_argument("--population-size", type=int, default=8)
    prepare.add_argument("--exploration-offset", type=int, default=0)

    worker = sub.add_parser("worker")
    worker.add_argument("plan")
    worker.add_argument("manifest")
    worker.add_argument("output")
    worker.add_argument("--allowed-root", action="append", required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--stage", type=int, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--repeat-seeds", type=int, default=1)
    worker.add_argument("--max-eval-blocks", type=int, default=4)

    reduce_cmd = sub.add_parser("select")
    reduce_cmd.add_argument("output")
    reduce_cmd.add_argument("results", nargs="+")
    reduce_cmd.add_argument("--survivors", type=int, required=True)

    final = sub.add_parser("finalize")
    final.add_argument("state_dir")
    final.add_argument("plan")
    final.add_argument("output")
    final.add_argument("results", nargs="+")

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        result = _cmd_prepare(args)
    elif args.cmd == "worker":
        result = run_candidate(
            args.plan,
            args.manifest,
            args.output,
            allowed_roots=args.allowed_root,
            candidate_index=args.index,
            stage=args.stage,
            steps=args.steps,
            repeat_seeds=args.repeat_seeds,
            max_eval_blocks=args.max_eval_blocks,
        )
    elif args.cmd == "select":
        result = select_survivors(
            args.results,
            args.output,
            survivor_count=args.survivors,
        )
    elif args.cmd == "finalize":
        result = finalize_swarm(
            args.state_dir,
            args.plan,
            args.results,
            args.output,
        )
    else:
        raise AssertionError(args.cmd)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
