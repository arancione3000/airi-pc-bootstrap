from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
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
from .native_lattice import AiriLatticeConfig


LATTICE_RESEARCH_STATE_VERSION = "airi-lattice-research-state-v1"


def _digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _research_root_config() -> AiriLatticeConfig:
    # Intentionally tiny: architecture search should explore many ideas cheaply.
    # Winning structural ideas can later be scaled and revalidated.
    return AiriLatticeConfig(
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
        surprise_power=1.5,
        deep_write_power=1.6,
        lattice_mix=0.20,
        memory_decay_min=0.25,
        memory_decay_max=0.995,
        tokenizer_version="byte-v1",
    ).validate()


def _load_champion(root: Path) -> LatticeGenome:
    # Architecture champions are versioned by the gate that selected them.
    # A champion from an older research-state version is deliberately reset:
    # old gates may have ignored a dimension (for example throughput) that is
    # now required for a legitimate win.
    status = _load_json(root / "lattice-status.json")
    if (
        isinstance(status, dict)
        and status.get("version") == LATTICE_RESEARCH_STATE_VERSION
    ):
        raw = _load_json(root / "lattice-champion.json")
        if raw:
            try:
                return LatticeGenome(**raw).validate()
            except Exception:
                pass
    return root_lattice_genome(_research_root_config())


def _candidate_math_rows(
    population: list[tuple[Any, LatticeGenome]],
) -> list[dict[str, Any]]:
    rows = []
    for index, (mutation, genome) in enumerate(population):
        config = genome.lattice_config()
        cost = architecture_cost_vector(config)
        certificate = stability_certificate(config)
        rows.append({
            "index": index,
            "mutation": mutation.to_dict(),
            "genome": genome.to_dict(),
            "loss": float("inf"),
            "active_parameters": cost["active_parameters"],
            "state_bytes": cost["state_bytes"],
            "train_seconds": float("inf"),
            "compute_proxy": cost["active_parameter_step_proxy"],
            "expected_reasoning_steps": cost["expected_reasoning_steps"],
            "stability": certificate,
        })
    return rows


def _prefilter_population(
    population: list[tuple[Any, LatticeGenome]],
    *,
    empirical_candidates: int,
) -> list[tuple[Any, LatticeGenome]]:
    """Cheaply choose diverse/stable candidates before any SGD.

    Loss/time are unknown here, so the prefilter uses active compute/state cost
    and keeps architectural family diversity. The empirical benchmark remains
    the authority.
    """
    if not population:
        return []
    rows = _candidate_math_rows(population)
    stable = [row for row in rows if row["stability"]["ok"]]
    stable.sort(
        key=lambda row: (
            row["compute_proxy"],
            row["state_bytes"],
            row["active_parameters"],
            row["index"],
        )
    )

    selected: list[int] = []
    families: set[str] = set()
    for row in stable:
        family = str(row["mutation"]["family"])
        if family in families:
            continue
        selected.append(int(row["index"]))
        families.add(family)
        if len(selected) >= empirical_candidates:
            break
    if len(selected) < empirical_candidates:
        for row in stable:
            index = int(row["index"])
            if index in selected:
                continue
            selected.append(index)
            if len(selected) >= empirical_candidates:
                break
    return [population[index] for index in selected]


def _benchmark_candidate(
    corpus_manifest: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    genome: LatticeGenome,
    steps: int,
    seed: int,
    max_eval_blocks: int,
) -> dict[str, Any]:
    return benchmark_lattice_against_transformer(
        str(corpus_manifest),
        allowed_roots=[str(Path(row)) for row in allowed_roots],
        lattice_config=genome.lattice_config(),
        lab_config=LatticeLabConfig(
            steps=max(1, int(steps)),
            batch_size=1,
            learning_rate=genome.learning_rate,
            min_learning_rate=genome.min_learning_rate,
            weight_decay=genome.weight_decay,
            validation_fraction=0.20,
            max_eval_blocks=max(1, int(max_eval_blocks)),
            seed=int(seed),
            device="cpu",
            minimum_loss_gain=0.001,
            max_domain_regression=0.08,
            max_active_parameter_ratio=1.20,
            min_throughput_ratio=0.50,
        ),
    )


def run_lattice_research_cycle(
    state_dir: str | Path,
    corpus_manifest: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    cycle: int,
    mathesis_signals: Iterable[str] | None = None,
    research: dict[str, Any] | None = None,
    population_size: int = 8,
    empirical_candidates: int = 1,
    benchmark_steps: int = 1,
    max_eval_blocks: int = 4,
    repeat_seeds: int = 1,
) -> dict[str, Any]:
    """Run one cheap structural-intelligence search step.

    Dozens of structural mutations may be generated, but only mathematically
    stable/cost-efficient candidates receive SGD. A configuration promotion
    requires wins on every requested independent scratch seed.
    """
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    champion = _load_champion(root)

    population = generate_lattice_population(
        champion,
        count=max(1, int(population_size)),
        exploration_offset=max(0, int(cycle) - 1),
        mathesis_signals=mathesis_signals,
        research=research,
        max_total_parameters=5_000_000,
        max_active_parameter_ratio=1.5,
    )
    shortlist = _prefilter_population(
        population,
        empirical_candidates=max(1, int(empirical_candidates)),
    )

    trials: list[dict[str, Any]] = []
    winners: list[tuple[float, LatticeGenome, dict[str, Any]]] = []
    for index, (mutation, genome) in enumerate(shortlist):
        reports = []
        all_wins = True
        for repeat in range(max(1, int(repeat_seeds))):
            seed = 700_001 + int(cycle) * 101 + index * 17 + repeat
            try:
                report = _benchmark_candidate(
                    corpus_manifest,
                    allowed_roots=allowed_roots,
                    genome=genome,
                    steps=benchmark_steps,
                    seed=seed,
                    max_eval_blocks=max_eval_blocks,
                )
            except Exception as exc:
                report = {
                    "ok": False,
                    "candidate_wins": False,
                    "decision": f"benchmark_error:{type(exc).__name__}:{exc}",
                }
            reports.append(report)
            all_wins = all_wins and bool(
                report.get("ok") and report.get("candidate_wins")
            )

        valid_losses = [
            float(report["candidate"]["training"]["final"]["loss"])
            for report in reports
            if report.get("ok")
            and isinstance(report.get("candidate"), dict)
        ]
        mean_loss = (
            sum(valid_losses) / len(valid_losses)
            if valid_losses
            else float("inf")
        )
        row = {
            "mutation": mutation.to_dict(),
            "genome": genome.to_dict(),
            "all_seed_wins": all_wins,
            "mean_candidate_loss": mean_loss,
            "reports": reports,
        }
        trials.append(row)
        if all_wins:
            winners.append((mean_loss, genome, row))

    promoted = False
    promotion_reason = "no Lattice candidate passed every independent scratch benchmark"
    winner = None
    if winners:
        winners.sort(key=lambda item: (item[0], item[1].genome_id))
        _loss, genome, row = winners[0]
        champion = genome
        _atomic_json(root / "lattice-champion.json", champion.to_dict())
        promoted = True
        promotion_reason = (
            "candidate beat matched scratch Transformer on every requested seed"
        )
        winner = {
            "genome": champion.to_dict(),
            "mutation": row["mutation"],
            "mean_candidate_loss": row["mean_candidate_loss"],
        }
    elif not (root / "lattice-champion.json").exists():
        _atomic_json(root / "lattice-champion.json", champion.to_dict())

    payload = {
        "ok": True,
        "version": LATTICE_RESEARCH_STATE_VERSION,
        "cycle": int(cycle),
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "winner": winner,
        "champion": champion.to_dict(),
        "generated_candidates": len(population),
        "empirical_candidates": len(shortlist),
        "benchmark_steps": int(benchmark_steps),
        "repeat_seeds": int(repeat_seeds),
        "trials": trials,
        "policy": {
            "external_pretrained": False,
            "research_only": True,
            "math_prefilter_is_not_promotion": True,
            "empirical_equal_budget_gate_required": True,
            "canonical_native_transformer_unchanged": True,
        },
    }
    _atomic_json(root / "lattice-status.json", payload)
    with (root / "lattice-history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return payload
