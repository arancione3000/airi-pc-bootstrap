from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import time
from typing import Any, Iterable, Sequence

from .airi_pc_lab import (
    build_airi_pc_lab_rows,
    load_verified_lab_experiences,
    record_verified_lab_experience,
    run_airi_pc_lab_probe,
    snapshot_airi_pc_lab,
    summarize_lab_learning,
)
from .corpus import repository_corpus
from .curriculum import DOMAINS, ResearchRow, train_rows, validation_rows
from .curriculum_memory import CurriculumMemory, canary_rows
from .evolution import (
    GeneralistGenome,
    compression_candidate,
    generate_challengers,
    progressive_scale_candidate,
    progressive_scale_target,
)
from .distillation import DistillationPrompt, distill_prompts
from .efficiency_engine import (
    active_learning_weights,
    efficiency_bonus,
    efficiency_profile,
    measure_inference_latency,
    island_schedule,
    island_weights,
    self_play_policy,
    sparse_expert_plan,
)
from .generalist_data_growth import grow_generalist_data
from .language_bridge import build_language_bridge_rows
from .bootstrap_data import load_bootstrap_replay
from .mathesis_bridge import mathesis_signals
from .model import estimate_parameter_count, parameter_count
from .pretraining import CorpusDocument, load_local_corpus
from .phase5_diagnostics import evaluate_phase5_language
from .research_cycle import (
    _genome_training_seed,
    _grouped_validation,
    _load_champion,
    _research_budget_reason,
    _research_eligible,
    _research_score,
    _save_champion,
    _tokenizer_for_genome,
    _train_genome,
    _weaknesses,
    adaptive_domain_weights,
    research_seed,
)
from .runtime import GeneralistRuntime
from .verified_self_play import (
    generate_verified_multiagent_rows,
    generate_verified_self_play_rows,
)


GENERALIST_SWARM_VERSION = "airi-generalist-free-speed-v5"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _history_tail(root: Path, limit: int = 8) -> list[dict[str, Any]]:
    path = root / "history.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)):]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _plateau_report(
    root: Path,
    champion_report: dict[str, Any],
    *,
    current_parameters: int,
) -> dict[str, Any]:
    history = _history_tail(root, 8)
    scores = []
    champion_ids = []
    for row in history:
        report = row.get("champion_report")
        champion = row.get("champion")
        if isinstance(report, dict):
            try:
                score = float(report.get("score"))
                if math.isfinite(score):
                    scores.append(score)
            except Exception:
                pass
        if isinstance(champion, dict) and champion.get("genome_id"):
            champion_ids.append(str(champion["genome_id"]))

    same_champion_cycles = 0
    current_id = champion_ids[-1] if champion_ids else None
    for value in reversed(champion_ids):
        if value != current_id:
            break
        same_champion_cycles += 1

    score_span = (
        max(scores) - min(scores)
        if len(scores) >= 2
        else float("inf")
    )
    generation_accuracy = float(
        champion_report.get("generation_exact_accuracy", 0.0) or 0.0
    )

    # The tiny initial model gets one early scale probe even before a long
    # plateau. Later tiers require evidence of saturation.
    under_initial_tier = int(current_parameters) < 250_000
    plateau = (
        same_champion_cycles >= 4
        or (len(scores) >= 4 and score_span < 0.75)
    )
    scale_probe = under_initial_tier or plateau
    return {
        "history_rows": len(history),
        "same_champion_cycles": same_champion_cycles,
        "score_span": None if not math.isfinite(score_span) else score_span,
        "generation_exact_accuracy": generation_accuracy,
        "under_initial_tier": under_initial_tier,
        "plateau": plateau,
        "scale_probe": scale_probe,
    }


def _continual_genome(champion: GeneralistGenome, cycle: int) -> GeneralistGenome:
    raw = champion.to_dict()
    raw.update({
        "generation": champion.generation + 1,
        "parent_id": champion.genome_id,
        "genome_id": f"generalist-{champion.generation + 1}-swarm-continual-{cycle}",
    })
    return GeneralistGenome(**raw).validate()


def _unique_candidates(
    rows: Iterable[tuple[str, GeneralistGenome]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, genome in rows:
        topology = {
            key: value
            for key, value in genome.to_dict().items()
            if key not in {"generation", "parent_id", "genome_id"}
        }
        # Learned checkpoints with identical topology are not duplicates:
        # their weights encode different acquired capabilities. Reserve one
        # independent slot for each fusion lineage while continuing to
        # deduplicate ordinary architecture/topology probes.
        lineage = (
            str(kind)
            if str(kind) in {"language_fusion", "specialist_fusion"}
            else "topology"
        )
        signature = json.dumps(
            {"lineage": lineage, "topology": topology},
            sort_keys=True,
        )
        if signature in seen:
            continue
        seen.add(signature)
        out.append({
            "index": len(out),
            "kind": kind,
            "genome": genome.to_dict(),
            "candidate_id": genome.genome_id,
            "estimated_parameters": estimate_parameter_count(
                genome.model_config()
            ),
        })
        if len(out) >= max(1, int(count)):
            break
    return out


def _phase5_language_fusion_candidate(
    root: Path,
    *,
    max_params: int,
) -> dict[str, Any] | None:
    """Expose a completed Phase-5 language checkpoint as a safe swarm parent.

    The language bootstrap already starts from the current architecture champion.
    This helper closes the loop in the opposite direction: once a language rung
    is fully persisted, its learned weights can enter the normal Generalist
    tournament. It never self-promotes; the usual domain, degeneration and
    external-reducer gates still decide whether the fused lineage becomes the
    champion.
    """
    bootstrap = root / "bootstrap-data"
    progress_path = bootstrap / "progress.json"
    report_path = bootstrap / "report.json"
    checkpoint = bootstrap / "candidate"
    if not progress_path.is_file() or not checkpoint.is_dir():
        return None
    try:
        progress = _load_json(progress_path)
        raw_genome = progress.get("capacity_genome")
        if not isinstance(raw_genome, dict):
            return None
        target_tokens = int(progress.get("target_tokens", 0) or 0)
        tokens_processed = int(progress.get("tokens_processed", 0) or 0)
        completed = {
            int(value)
            for value in (progress.get("completed_rungs") or [])
            if isinstance(value, (int, float))
        }
        sft_completed = {
            int(value)
            for value in (progress.get("sft_completed_rungs") or [])
            if isinstance(value, (int, float))
        }
        if (
            target_tokens <= 0
            or tokens_processed < target_tokens
            or target_tokens not in completed
            or target_tokens not in sft_completed
        ):
            return None

        genome = GeneralistGenome(**raw_genome).validate()
        runtime = GeneralistRuntime.from_checkpoint(checkpoint, device="cpu")
        expected = genome.model_config(runtime.tokenizer.vocab_size).to_dict()
        if runtime.config.to_dict() != expected:
            return None
        parameters = int(parameter_count(runtime.model))
        if parameters > int(max_params):
            return None

        report = _load_json(report_path) if report_path.is_file() else {}
        after = report.get("after") if isinstance(report, dict) else {}
        if not isinstance(after, dict):
            after = {}
        return {
            "genome": genome,
            "checkpoint": "bootstrap-data/candidate",
            "parameters": parameters,
            "target_tokens": target_tokens,
            "tokens_processed": tokens_processed,
            "unique_corpus_target_tokens": int(
                progress.get("unique_corpus_target_tokens", 0) or 0
            ),
            "minimum_success": bool(report.get("minimum_success")) if isinstance(report, dict) else False,
            "language_nll": float(after.get("language_nll", float("inf"))),
            "repetition_rate": float(after.get("repetition_rate", 1.0) or 1.0),
            "multiword_output_rate": float(after.get("multiword_output_rate", 0.0) or 0.0),
            "pathological_repetition": bool(after.get("pathological_repetition", False)),
        }
    except Exception:
        return None


def _internal_distillation_rows(
    root: Path,
    teacher: dict[str, Any] | None,
    *,
    max_rows: int = 12,
) -> tuple[list[ResearchRow], dict[str, Any]]:
    """Distill only from a strong persisted specialist and only on train rows."""
    if not isinstance(teacher, dict):
        return [], {"enabled": False, "reason": "no persisted specialist teacher"}
    island = str(teacher.get("island") or "")
    summary = teacher.get("summary") or {}
    if island not in {"language", "coding", "reasoning"}:
        return [], {
            "enabled": False,
            "reason": "tool/efficiency specialists use verified replay rather than free-form distillation",
        }
    if not bool(summary.get("all_seed_eligible")):
        return [], {"enabled": False, "reason": "specialist has not passed all-seed research gates"}
    if bool(summary.get("any_generation_pathological_repetition", True)):
        return [], {"enabled": False, "reason": "specialist still has pathological repetition"}
    if float(summary.get("mean_generation_similarity", 0.0) or 0.0) < 0.45:
        return [], {"enabled": False, "reason": "specialist generation similarity is below 0.45"}
    if float(summary.get("mean_generation_repetition_rate", 1.0) or 1.0) > 0.45:
        return [], {"enabled": False, "reason": "specialist repetition rate is above 0.45"}

    checkpoint = root / str(teacher.get("checkpoint") or "")
    try:
        runtime = GeneralistRuntime.from_checkpoint(checkpoint, device="cpu")
    except Exception as exc:
        return [], {
            "enabled": False,
            "reason": f"teacher checkpoint unavailable:{type(exc).__name__}:{exc}",
        }

    domain_map = {
        "language": {"language"},
        "coding": {"coding"},
        "reasoning": {"reasoning", "data"},
    }
    prompts: list[DistillationPrompt] = []
    for row in train_rows():
        if row.domain not in domain_map[island]:
            continue
        user = next(
            (
                str(message.get("content", ""))
                for message in row.messages
                if str(message.get("role", "")) == "user"
            ),
            "",
        ).strip()
        if not user:
            continue
        prompts.append(DistillationPrompt(domain=row.domain, prompt=user))
        if len(prompts) >= max(1, min(24, int(max_rows))):
            break

    examples, report = distill_prompts(
        runtime,
        prompts,
        max_new_tokens=128,
        max_output_chars=4000,
    )
    distilled_domain = island if island in {"language", "coding"} else "reasoning"
    rows = [
        ResearchRow(distilled_domain, list(example.messages))
        for example in examples
    ]
    return rows, {
        "enabled": True,
        "teacher_island": island,
        "teacher_candidate_id": summary.get("candidate_id"),
        "accepted": len(rows),
        "requested": len(prompts),
        "research_only": True,
        "promotion_bypass": False,
        "distillation_report": report,
    }


def _persisted_specialist_candidate(
    root: Path,
    *,
    champion_id: str,
    max_params: int,
) -> dict[str, Any] | None:
    specialists = root / "specialists"
    rows: list[dict[str, Any]] = []
    for island in ("language", "coding", "reasoning", "tools", "efficiency"):
        folder = specialists / island
        summary_path = folder / "summary.json"
        if not summary_path.is_file() or not (folder / "model.pt").is_file():
            continue
        try:
            summary = _load_json(summary_path)
            genome = GeneralistGenome(**dict(summary.get("genome") or {})).validate()
        except Exception:
            continue
        if genome.genome_id == champion_id:
            continue
        if int(summary.get("parameters", 0) or 0) > int(max_params):
            continue
        if not bool(summary.get("research_only", True)):
            continue
        rows.append({
            "island": island,
            "checkpoint": str(Path("specialists") / island),
            "summary": summary,
            "genome": genome,
        })
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            -float((row["summary"] or {}).get("mean_fitness_score", 0.0) or 0.0),
            int((row["summary"] or {}).get("parameters", 1 << 60) or (1 << 60)),
        )
    )
    return rows[0]


def prepare_swarm(
    state_dir: str | Path,
    output_path: str | Path,
    *,
    population_size: int = 8,
    max_params: int = 7_000_000,
    max_context: int = 512,
    max_width: int = 384,
    max_layers: int = 10,
    curriculum_max_rows: int = 4000,
    mathesis_state_dir: str | Path | None = None,
    grow_data: bool = True,
    data_max_new_bytes: int = 8_000_000,
    data_max_total_bytes: int = 50_000_000,
    github_token: str | None = None,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    loaded = _load_champion(root, device="cpu")
    if loaded is None:
        genome = research_seed()
        runtime, report = _train_genome(
            genome,
            steps=30,
            seed=_genome_training_seed(genome, namespace="free-speed-bootstrap"),
            device="cpu",
        )
        _save_champion(root, genome, runtime, report)
        loaded = (genome, runtime)

    champion_genome, champion_runtime = loaded
    previous = {}
    try:
        previous = _load_json(root / "status.json")
    except Exception:
        previous = {}
    cycle = int(previous.get("cycle", 0) or 0) + 1

    champion_report = _grouped_validation(
        champion_runtime.model,
        champion_runtime.tokenizer,
        validation_rows(),
        device="cpu",
    )
    champion_report["parameters"] = parameter_count(champion_runtime.model)
    champion_report["score"] = _research_score(
        champion_report,
        champion_report["parameters"],
    )
    rotating = canary_rows(cycle)
    champion_report["canary_cycle"] = cycle
    champion_report["canary"] = _grouped_validation(
        champion_runtime.model,
        champion_runtime.tokenizer,
        rotating,
        device="cpu",
    )
    champion_report["score"] = _research_score(
        champion_report,
        champion_report["parameters"],
    )

    signals = _weaknesses(champion_report)
    mathesis = None
    if mathesis_state_dir and Path(mathesis_state_dir).exists():
        try:
            mathesis = mathesis_signals(mathesis_state_dir)
            signals.extend(mathesis.get("signals") or [])
        except Exception as exc:
            mathesis = {
                "ok": False,
                "signals": [],
                "reason": f"{type(exc).__name__}:{exc}",
            }
    if (
        champion_genome.tokenizer_version == "byte-v1"
        and float(champion_report.get("generation_exact_accuracy", 0.0)) < 1.0
    ):
        signals.append("tokenizer_efficiency_gap")
    signals = list(dict.fromkeys(str(row) for row in signals))

    lab_snapshot = snapshot_airi_pc_lab(Path.cwd())
    lab_rows = build_airi_pc_lab_rows(lab_snapshot, max_rows=18)
    lab_experience_rows = load_verified_lab_experiences(root, max_rows=32)
    lab_training_rows = [*lab_rows, *lab_experience_rows]
    lab_learning = summarize_lab_learning(
        lab_training_rows,
        verified_experience_rows=len(lab_experience_rows),
    )
    specialist_fusion = _persisted_specialist_candidate(
        root,
        champion_id=champion_genome.genome_id,
        max_params=int(max_params),
    )
    distilled_rows, distillation_report = _internal_distillation_rows(
        root,
        specialist_fusion,
        max_rows=12,
    )

    self_play = self_play_policy(
        champion_report,
        verified_tool_experiences=len(lab_experience_rows),
    )
    self_play_solver = champion_runtime
    self_play_solver_source = "champion"
    if bool(self_play.get("enabled")) and specialist_fusion is not None:
        try:
            self_play_solver = GeneralistRuntime.from_checkpoint(
                root / str(specialist_fusion["checkpoint"]),
                device="cpu",
            )
            self_play_solver_source = (
                "specialist:" + str(specialist_fusion.get("island") or "unknown")
            )
        except Exception:
            self_play_solver = champion_runtime
            self_play_solver_source = "champion_fallback"
    self_play_rows, self_play_report = generate_verified_multiagent_rows(
        champion_runtime,
        self_play_solver,
        self_play,
        cycle=cycle,
        max_tasks=12,
    )
    self_play_report["solver_source"] = self_play_solver_source
    memory = CurriculumMemory(root, max_rows=curriculum_max_rows)
    curriculum = memory.expand(
        cycle,
        signals=signals,
        extra_rows=[*lab_training_rows, *self_play_rows, *distilled_rows],
    )
    domain_weights = active_learning_weights(
        champion_report,
        base_weights=adaptive_domain_weights(champion_report),
        verified_tool_experiences=len(lab_experience_rows),
    )

    data_report: dict[str, Any] = {
        "ok": True,
        "skipped": True,
        "reason": "automatic data growth disabled",
    }
    if grow_data:
        try:
            data_report = grow_generalist_data(
                root / "autodata",
                signals=signals,
                github_token=github_token,
                max_new_bytes=int(data_max_new_bytes),
                max_total_bytes=int(data_max_total_bytes),
            )
        except Exception as exc:
            data_report = {
                "ok": False,
                "skipped": False,
                "reason": f"{type(exc).__name__}:{exc}",
            }

    current_params = int(champion_report["parameters"])
    plateau = _plateau_report(
        root,
        champion_report,
        current_parameters=current_params,
    )
    target = progressive_scale_target(
        current_params,
        max_parameters=max_params,
    )
    scale_genomes: list[tuple[str, GeneralistGenome]] = []
    if target is not None and plateau["scale_probe"]:
        for kind, prefer_preserving in (
            ("progressive_scale", True),
            ("progressive_scale_balanced", False),
        ):
            try:
                candidate = progressive_scale_candidate(
                    champion_genome,
                    target_parameters=target,
                    vocab_size=champion_runtime.tokenizer.vocab_size,
                    max_width=max_width,
                    max_layers=max_layers,
                    prefer_function_preserving=prefer_preserving,
                )
            except Exception:
                continue
            if any(
                existing.genome_id == candidate.genome_id
                for _existing_kind, existing in scale_genomes
            ):
                continue
            scale_genomes.append((kind, candidate))

    compressed_genome = compression_candidate(
        champion_genome,
        vocab_size=champion_runtime.tokenizer.vocab_size,
        target_ratio=0.72,
    )
    generated = generate_challengers(
        champion_genome,
        signals=signals,
        count=max(8, population_size),
        exploration_offset=max(0, cycle - 1),
    )
    language_fusion = _phase5_language_fusion_candidate(
        root,
        max_params=int(max_params),
    )
    rows: list[tuple[str, GeneralistGenome]] = []
    if language_fusion is not None:
        # Put the learned-language checkpoint first. If its topology is the
        # same as the champion, topology dedup must retain the better-trained
        # weights rather than replacing them with a fresh continual copy.
        rows.append(("language_fusion", language_fusion["genome"]))
    rows.append(("continual", _continual_genome(champion_genome, cycle)))
    if specialist_fusion is not None:
        rows.append(("specialist_fusion", specialist_fusion["genome"]))
    rows.extend(scale_genomes)
    if compressed_genome is not None:
        rows.append(("compression", compressed_genome))
    rows.extend(("architecture", genome) for genome in generated)

    candidates = _unique_candidates(rows, count=population_size)
    scheduled_islands = island_schedule(len(candidates), signals=signals)
    for idx, candidate in enumerate(candidates):
        kind = str(candidate.get("kind") or "")
        if kind == "language_fusion":
            island = "language"
        elif kind == "specialist_fusion" and specialist_fusion is not None:
            island = str(specialist_fusion["island"])
        elif kind == "compression" or kind.startswith("progressive_scale"):
            island = "efficiency"
        else:
            island = scheduled_islands[idx]
        candidate["island"] = island
        candidate["domain_weights"] = island_weights(domain_weights, island)
        candidate["sparse_expert_role"] = island
    if language_fusion is not None:
        fusion_genome = language_fusion["genome"]
        fusion_evidence = {
            key: value
            for key, value in language_fusion.items()
            if key != "genome"
        }
        for candidate in candidates:
            if candidate.get("candidate_id") != fusion_genome.genome_id:
                continue
            candidate["initial_checkpoint"] = str(language_fusion["checkpoint"])
            candidate["initial_checkpoint_mode"] = "language_fusion"
            candidate["fusion_evidence"] = fusion_evidence
            break
    if specialist_fusion is not None:
        specialist_genome = specialist_fusion["genome"]
        for candidate in candidates:
            if candidate.get("candidate_id") != specialist_genome.genome_id:
                continue
            candidate["initial_checkpoint"] = str(specialist_fusion["checkpoint"])
            candidate["initial_checkpoint_mode"] = "specialist_fusion"
            candidate["specialist_evidence"] = dict(specialist_fusion["summary"])
            candidate["island"] = str(specialist_fusion["island"])
            candidate["domain_weights"] = island_weights(
                domain_weights,
                candidate["island"],
            )
            break
    # If a duplicate collapsed the population, rotate deeper into the standard
    # mutation library until the requested matrix is full.
    offset = cycle + population_size
    while len(candidates) < population_size and offset < cycle + 64:
        extras = generate_challengers(
            champion_genome,
            signals=signals,
            count=population_size,
            exploration_offset=offset,
        )
        candidates = _unique_candidates(
            [
                (row["kind"], GeneralistGenome(**row["genome"]).validate())
                for row in candidates
            ]
            + [("architecture", genome) for genome in extras],
            count=population_size,
        )
        offset += 1

    scheduled_islands = island_schedule(len(candidates), signals=signals)
    for idx, candidate in enumerate(candidates):
        kind = str(candidate.get("kind") or "")
        if kind == "language_fusion":
            island = "language"
        elif kind == "specialist_fusion" and specialist_fusion is not None:
            island = str(specialist_fusion["island"])
        elif kind == "compression" or kind.startswith("progressive_scale"):
            island = "efficiency"
        else:
            island = str(candidate.get("island") or scheduled_islands[idx])
        candidate["island"] = island
        candidate["domain_weights"] = island_weights(domain_weights, island)
        candidate["sparse_expert_role"] = island

    # _unique_candidates() intentionally rebuilds rows during population refill.
    # Re-attach learned checkpoint provenance after that step so a language or
    # specialist fusion candidate never silently falls back to fresh weights.
    if language_fusion is not None:
        fusion_genome = language_fusion["genome"]
        for candidate in candidates:
            if candidate.get("candidate_id") == fusion_genome.genome_id:
                candidate["initial_checkpoint"] = str(language_fusion["checkpoint"])
                candidate["initial_checkpoint_mode"] = "language_fusion"
                candidate["fusion_evidence"] = {
                    key: value
                    for key, value in language_fusion.items()
                    if key != "genome"
                }
                break
    if specialist_fusion is not None:
        specialist_genome = specialist_fusion["genome"]
        for candidate in candidates:
            if candidate.get("candidate_id") == specialist_genome.genome_id:
                candidate["initial_checkpoint"] = str(specialist_fusion["checkpoint"])
                candidate["initial_checkpoint_mode"] = "specialist_fusion"
                candidate["specialist_evidence"] = dict(specialist_fusion["summary"])
                break

    champion_vocab_size = int(getattr(champion_runtime.tokenizer, "vocab_size", 0) or 0)
    bpe_growth_target = champion_vocab_size
    if (
        candidates
        and "language_gap" in set(signals)
        and champion_genome.tokenizer_version == "bpe-v1"
        and champion_vocab_size < 1024
    ):
        bpe_growth_target = min(1024, champion_vocab_size + 128)
        candidates[0]["kind"] = "language_bpe_probe"
        candidates[0]["bpe_vocab_target"] = int(bpe_growth_target)

    plan = {
        "ok": bool(candidates),
        "version": GENERALIST_SWARM_VERSION,
        "cycle": cycle,
        "champion": champion_genome.to_dict(),
        "champion_report": champion_report,
        "signals": signals,
        "mathesis": mathesis,
        "airi_pc_lab": {
            "snapshot": lab_snapshot,
            "learning": lab_learning,
        },
        "curriculum": curriculum,
        "domain_weights": domain_weights,
        "active_learning": {
            "enabled": True,
            "domain_weights": domain_weights,
            "verified_tool_experiences": len(lab_experience_rows),
            "policy": "allocate bounded extra replay to measured weak domains",
        },
        "sparse_experts": sparse_expert_plan(
            available_islands=sorted({str(row.get("island") or "") for row in candidates}),
        ),
        "self_play": {
            **dict(self_play),
            "cycle_report": self_play_report,
        },
        "internal_distillation": distillation_report,
        "compression": {
            "enabled": compressed_genome is not None,
            "candidate_id": (
                compressed_genome.genome_id if compressed_genome is not None else None
            ),
            "policy": "smaller same-width inherited candidate must pass normal gates",
        },
        "data_growth": data_report,
        "plateau": plateau,
        "progressive_tokenizer": {
            "current_vocab_size": champion_vocab_size,
            "target_vocab_size": bpe_growth_target,
            "maximum_vocab_size": 1024,
            "growth_per_probe": 128,
            "probe_generated": any(
                row.get("kind") == "language_bpe_probe"
                for row in candidates
            ),
        },
        "progressive_scaling": {
            "current_parameters": current_params,
            "target_parameters": target,
            "candidate_generated": bool(scale_genomes),
            "candidate_count": len(scale_genomes),
            "strategies": [kind for kind, _genome in scale_genomes],
            "tiers": [
                250_000,
                500_000,
                1_250_000,
                3_000_000,
                7_000_000,
                12_000_000,
                20_000_000,
            ],
            "minimum_scale_budget_multiplier": 1.25,
            "maximum_scale_budget_multiplier": 1.75,
        },
        "converged_champion": {
            "language_fusion_available": language_fusion is not None,
            "specialist_fusion_available": specialist_fusion is not None,
            "specialist_teacher": (
                {
                    "island": specialist_fusion["island"],
                    "checkpoint": specialist_fusion["checkpoint"],
                    "summary": specialist_fusion["summary"],
                }
                if specialist_fusion is not None
                else None
            ),
            "language_teacher": (
                {
                    key: value
                    for key, value in language_fusion.items()
                    if key != "genome"
                }
                if language_fusion is not None
                else None
            ),
            "policy": (
                "completed language checkpoint competes as a normal research candidate; "
                "promotion still requires protected generalist and degeneration gates"
            ),
        },
        "limits": {
            "max_params": int(max_params),
            "max_context": int(max_context),
            "max_width": int(max_width),
            "max_layers": int(max_layers),
        },
        "candidates": candidates,
        "matrix": {
            "include": [{"index": row["index"]} for row in candidates]
        },
        "policy": {
            "parallel_candidates": True,
            "candidate_can_self_promote": False,
            "production_qualification_separate": True,
            "external_pretrained": False,
            "adaptive_curriculum": True,
            "progressive_scaling": True,
            "scale_probe_retention": "reserve one safe scale survivor through reductions",
            "scale_budget_adaptive": True,
            "adaptive_pretraining_budget": True,
            "domain_balanced_pretraining": True,
            "language_gap_routing": True,
            "autoregressive_similarity_ranking": True,
            "corpus_language_bridge": True,
            "progressive_bpe_vocab": 1024,
            "inherited_sft_lr_cap": 0.001,
            "weight_inheritance": True,
            "automatic_best_of_both_fusion": True,
            "language_fusion_self_promotion": False,
            "automatic_data_growth_fail_closed": True,
            "airi_pc_lab_read_only": True,
            "airi_pc_lab_training": True,
            "active_learning": True,
            "evolution_islands": ["language", "coding", "reasoning", "tools", "efficiency"],
            "system_sparse_experts": True,
            "max_active_experts": 1,
            "compression_research": True,
            "verified_self_play": bool(self_play.get("enabled")),
            "internal_distillation": bool(distillation_report.get("enabled")),
            "corpus_language_bridge": True,
            "progressive_bpe_vocab": 1024,
        },
    }
    _atomic_json(Path(output_path), plan)
    return plan


def _candidate(plan: dict[str, Any], index: int) -> dict[str, Any]:
    for row in plan.get("candidates") or []:
        if int(row.get("index", -1)) == int(index):
            return row
    raise IndexError(f"Generalist candidate index {index} not found")


def _corpus_documents(
    repo_root: str | Path,
    state_dir: str | Path,
    *,
    max_repo_bytes: int,
    max_external_bytes: int,
    bootstrap_replay_documents: Sequence[CorpusDocument] | None = None,
) -> tuple[list[CorpusDocument], dict[str, Any]]:
    repo_docs, repo_manifest = repository_corpus(
        repo_root,
        max_files=1200,
        max_bytes=max(64_000, int(max_repo_bytes)),
        max_file_bytes=512_000,
        chunk_chars=6000,
        max_documents=4000,
    )
    documents = list(repo_docs)
    external_report = {
        "documents": 0,
        "total_bytes": 0,
    }
    approved = Path(state_dir) / "autodata" / "approved"
    if approved.is_dir():
        loaded = load_local_corpus(
            [approved],
            allowed_roots=[approved],
            max_file_bytes=512_000,
            max_total_bytes=max(0, int(max_external_bytes)),
            max_documents=10_000,
        )
        manifest_domains: dict[str, str] = {}
        try:
            manifest = _load_json(Path(state_dir) / "autodata" / "manifest.json")
            for item in manifest.get("files", []):
                if not isinstance(item, dict):
                    continue
                digest = str(item.get("sha256") or "")
                domain = str(item.get("domain") or "general")
                if digest:
                    manifest_domains[digest] = domain
        except Exception:
            manifest_domains = {}

        external_documents = [
            CorpusDocument(
                source=document.source,
                text=document.text,
                sha256=document.sha256,
                bytes=document.bytes,
                domain=manifest_domains.get(document.sha256, "general"),
            )
            for document in loaded.documents
        ]
        documents.extend(external_documents)
        external_domain_documents: dict[str, int] = {}
        for document in external_documents:
            external_domain_documents[document.domain] = (
                external_domain_documents.get(document.domain, 0) + 1
            )
        external_report = {
            "documents": len(external_documents),
            "total_bytes": loaded.total_bytes,
            "skipped": len(loaded.skipped),
            "domain_documents": external_domain_documents,
        }

    replay_documents = list(bootstrap_replay_documents or [])
    documents.extend(replay_documents)

    # Cross-source content dedup.
    unique: dict[str, CorpusDocument] = {}
    for row in documents:
        unique.setdefault(row.sha256, row)
    final = list(unique.values())
    replay_domains: dict[str, int] = {}
    for row in replay_documents:
        replay_domains[row.domain] = replay_domains.get(row.domain, 0) + 1
    return final, {
        "repository": repo_manifest.to_dict(),
        "external": external_report,
        "bootstrap_replay": {
            "documents": len(replay_documents),
            "bytes": sum(row.bytes for row in replay_documents),
            "domain_documents": dict(sorted(replay_domains.items())),
        },
        "documents": len(final),
        "bytes": sum(row.bytes for row in final),
    }


def _language_rescue_weights(
    base_task_weights: dict[str, float] | None,
    documents: Sequence[CorpusDocument],
    signals: set[str],
) -> tuple[dict[str, float], dict[str, float] | None, dict[str, Any]]:
    rescue = bool(
        "language_gap" in signals
        or "language_collapse" in signals
        or "autoregressive_collapse" in signals
    )
    task_weights = dict(base_task_weights or {})
    if not rescue:
        return task_weights, None, {
            "enabled": False,
            "reason": "no language collapse signal",
        }

    # SFT remains domain-balanced by default; language gets two extra replay
    # copies, while the other weak domains are capped so they cannot drown it.
    for domain in ("coding", "data", "reasoning", "structured", "tools"):
        if domain in task_weights:
            task_weights[domain] = max(1.0, min(2.0, float(task_weights[domain])))
    task_weights["language"] = max(3.0, float(task_weights.get("language", 1.0)))

    available = sorted({
        str(getattr(document, "domain", "general") or "general")
        for document in documents
    })
    natural = [
        domain for domain in available
        if domain == "general"
        or domain == "dialogue"
        or domain.startswith("language")
    ]
    pretrain_weights: dict[str, float] = {}
    natural_mass = 0.68
    if natural:
        share = natural_mass / len(natural)
        for domain in natural:
            pretrain_weights[domain] = share

    fixed = {
        "code": 0.10,
        "reasoning": 0.14,
        "data": 0.05,
    }
    for domain, weight in fixed.items():
        if domain in available:
            pretrain_weights[domain] = weight

    assigned = sum(pretrain_weights.values())
    others = [domain for domain in available if domain not in pretrain_weights]
    remaining = max(0.01, 1.0 - assigned)
    if others:
        each = remaining / len(others)
        for domain in others:
            pretrain_weights[domain] = each
    elif pretrain_weights:
        scale = 1.0 / sum(pretrain_weights.values())
        pretrain_weights = {
            domain: value * scale
            for domain, value in pretrain_weights.items()
        }

    return task_weights, pretrain_weights, {
        "enabled": True,
        "signals": sorted(signals),
        "sft_language_weight": float(task_weights["language"]),
        "pretraining_domain_weights": dict(sorted(pretrain_weights.items())),
        "natural_domain_mass": float(sum(
            pretrain_weights.get(domain, 0.0)
            for domain in natural
        )),
        "natural_domains": natural,
        "available_domains": available,
    }


def _adaptive_pretrain_steps(
    requested_steps: int,
    *,
    corpus_bytes: int,
    stage: int,
    scale_multiplier: float = 1.0,
) -> int:
    """Increase grounded pretraining only when there is corpus to justify it.

    The budget grows with corpus size and later successive-halving stages, but
    remains capped so autonomous CPU runs stay bounded.
    """
    requested = max(0, int(requested_steps))
    if requested <= 0:
        return 0

    size = max(0, int(corpus_bytes))
    if size >= 8_000_000:
        corpus_multiplier = 4.0
    elif size >= 2_000_000:
        corpus_multiplier = 3.0
    elif size >= 512_000:
        corpus_multiplier = 2.0
    else:
        corpus_multiplier = 1.0

    stage_multiplier = 1.0 + 0.25 * max(0, min(2, int(stage) - 1))
    effective = int(math.ceil(
        requested
        * corpus_multiplier
        * stage_multiplier
        * max(1.0, float(scale_multiplier))
    ))
    return min(48, max(requested, effective))


def _domain_regression(
    champion: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    old = champion.get("domain_nll_per_byte") or champion.get("domain_loss") or {}
    new = candidate.get("domain_nll_per_byte") or candidate.get("domain_loss") or {}
    values = []
    for domain, old_value in old.items():
        if domain in new:
            values.append(float(new[domain]) - float(old_value))
    # Missing comparable held-out domains must remain fail-closed, but the
    # persisted swarm report is strict JSON and must never emit Infinity.
    return max(values) if values else 1_000_000.0


def _language_fusion_retention_gate(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Require a fused candidate to retain the language checkpoint's gains."""
    reasons: list[str] = []
    baseline_nll = float(baseline.get("language_nll", float("inf")))
    current_nll = float(current.get("language_nll", float("inf")))
    if not math.isfinite(current_nll) or current_nll > baseline_nll + 0.15:
        reasons.append("Phase-5 language NLL regressed by more than 0.15")

    baseline_multi = float(baseline.get("multiword_output_rate", 0.0) or 0.0)
    current_multi = float(current.get("multiword_output_rate", 0.0) or 0.0)
    if current_multi < max(0.0, baseline_multi - 0.10):
        reasons.append("multi-word output retention regressed by more than 0.10")

    baseline_rep = float(baseline.get("repetition_rate", 1.0) or 1.0)
    current_rep = float(current.get("repetition_rate", 1.0) or 1.0)
    if current_rep > min(1.0, baseline_rep + 0.10):
        reasons.append("repetition rate regressed by more than 0.10")

    if (
        not bool(baseline.get("pathological_repetition", False))
        and bool(current.get("pathological_repetition", False))
    ):
        reasons.append("fused candidate reintroduced pathological repetition")
    return (not reasons), reasons


def _load_stage_source(
    checkpoint: str | Path,
    *,
    genome: GeneralistGenome,
    cycle: int,
    stage: int,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    root = Path(checkpoint).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("cumulative stage checkpoint requires metadata.json")
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("cumulative stage checkpoint metadata must be an object")
    if str(metadata.get("candidate_id") or "") != genome.genome_id:
        raise ValueError("cumulative stage checkpoint candidate mismatch")
    if int(metadata.get("cycle", -1)) != int(cycle):
        raise ValueError("cumulative stage checkpoint cycle mismatch")
    previous_stage = int(metadata.get("stage", -1))
    if previous_stage != int(stage) - 1:
        raise ValueError("cumulative stage checkpoint must come from immediately previous stage")

    runtime = GeneralistRuntime.from_checkpoint(root, device="cpu")
    expected = genome.model_config(runtime.tokenizer.vocab_size).to_dict()
    if runtime.config.to_dict() != expected:
        raise ValueError("cumulative stage checkpoint architecture mismatch")
    return runtime, metadata


def _load_research_incumbent_source(
    checkpoint: str | Path,
    *,
    genome: GeneralistGenome,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    root = Path(checkpoint).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("research incumbent checkpoint requires metadata.json")
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("research incumbent checkpoint metadata must be an object")
    if str(metadata.get("candidate_id") or "") != genome.genome_id:
        raise ValueError("research incumbent candidate mismatch")
    if bool(metadata.get("production_qualified")):
        raise ValueError("production-qualified checkpoint must not be used as research incumbent")

    runtime = GeneralistRuntime.from_checkpoint(root, device="cpu")
    expected = genome.model_config(runtime.tokenizer.vocab_size).to_dict()
    if runtime.config.to_dict() != expected:
        raise ValueError("research incumbent architecture mismatch")
    return runtime, metadata


def _load_language_fusion_source(
    checkpoint: str | Path,
    *,
    genome: GeneralistGenome,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    root = Path(checkpoint).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("language fusion checkpoint requires metadata.json")
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("language fusion checkpoint metadata must be an object")
    role = str(metadata.get("role") or "")
    if not role.startswith("phase5_language_bootstrap"):
        raise ValueError("language fusion source must come from Phase-5 bootstrap")
    if bool(metadata.get("production_qualified")):
        raise ValueError("production-qualified checkpoint must not be relabeled as fusion source")

    runtime = GeneralistRuntime.from_checkpoint(root, device="cpu")
    expected = genome.model_config(runtime.tokenizer.vocab_size).to_dict()
    if runtime.config.to_dict() != expected:
        raise ValueError("language fusion checkpoint architecture mismatch")
    return runtime, metadata



def _load_live_lineage_source(
    checkpoint: str | Path,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    """Load the newest persisted AIRI weights as a research starting point.

    Unlike a stage/incumbent checkpoint, the live lineage is intentionally
    allowed to have a different topology from the candidate. _train_genome()
    performs the safe architecture-aware transfer into each challenger.
    """
    root = Path(checkpoint).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("live lineage checkpoint requires metadata.json")
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("live lineage checkpoint metadata must be an object")
    role = str(metadata.get("role") or "")
    if not (
        role.startswith("phase5_language_bootstrap")
        or role == "active_airi_lineage"
        or role == "research_champion"
        or role == "production_champion"
    ):
        raise ValueError(f"unsupported live-lineage checkpoint role: {role}")
    runtime = GeneralistRuntime.from_checkpoint(root, device="cpu")
    return runtime, metadata

def _capacity_budget_multiplier(
    kind: str,
    *,
    candidate_parameters: int,
    current_parameters: int,
) -> float:
    candidate = max(1, int(candidate_parameters))
    current = max(1, int(current_parameters))
    normalized_kind = str(kind or "")
    is_capacity_probe = (
        normalized_kind.startswith("progressive_scale")
        or normalized_kind == "architecture_capacity_scale"
        or normalized_kind == "architecture_capacity_plus_structure"
        or normalized_kind == "architecture_incumbent"
        or (
            current >= 10_000
            and candidate >= int(current * 1.35)
        )
    )
    if not is_capacity_probe:
        return 1.0
    ratio = max(1.0, candidate / float(current))
    return float(min(2.25, max(1.25, math.sqrt(ratio))))


def run_candidate(
    plan_path: str | Path,
    state_dir: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    candidate_index: int,
    stage: int,
    steps: int,
    repeat_seeds: int = 1,
    pretrain_steps: int = 1,
    max_repo_bytes: int = 5_000_000,
    max_external_bytes: int = 20_000_000,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    row = _candidate(plan, candidate_index)
    genome = GeneralistGenome(**dict(row["genome"])).validate()
    root = Path(state_dir)
    loaded = _load_champion(root, device="cpu")
    if loaded is None:
        raise RuntimeError("Generalist swarm worker has no champion checkpoint")
    _champion_genome, champion_runtime = loaded
    source_runtime = champion_runtime
    source_metadata: dict[str, Any] = {}
    continued_from_incumbent = False
    continued_from_language_fusion = False
    if source_checkpoint is not None:
        source_runtime, source_metadata = _load_stage_source(
            source_checkpoint,
            genome=genome,
            cycle=int(plan["cycle"]),
            stage=int(stage),
        )
    elif int(stage) == 1 and row.get("initial_checkpoint"):
        initial_checkpoint = root / str(row["initial_checkpoint"])
        initial_mode = str(row.get("initial_checkpoint_mode") or "")
        if initial_mode == "language_fusion":
            source_runtime, source_metadata = _load_language_fusion_source(
                initial_checkpoint,
                genome=genome,
            )
            continued_from_language_fusion = True
        elif initial_mode == "live_lineage":
            source_runtime, source_metadata = _load_live_lineage_source(
                initial_checkpoint,
            )
        else:
            source_runtime, source_metadata = _load_research_incumbent_source(
                initial_checkpoint,
                genome=genome,
            )
            continued_from_incumbent = True

    memory = CurriculumMemory(root, max_rows=20_000)
    replay_rows = memory.rows()
    bootstrap_replay = load_bootstrap_replay(root / "bootstrap-data")
    documents, corpus_report = _corpus_documents(
        repo_root,
        root,
        max_repo_bytes=max_repo_bytes,
        max_external_bytes=max_external_bytes,
        bootstrap_replay_documents=bootstrap_replay.documents,
    )
    signals = set(str(value) for value in (plan.get("signals") or []))
    language_rescue = bool(
        "language_gap" in signals
        or "language_collapse" in signals
        or "autoregressive_collapse" in signals
    )
    language_bridge_rows = build_language_bridge_rows(
        documents,
        max_rows=(256 if language_rescue else 16),
        seed=int(plan["cycle"]),
    )
    bootstrap_sft_rows = [
        ResearchRow("language", list(example.messages))
        for example in bootstrap_replay.sft_train
    ]
    training_replay_rows = (
        replay_rows
        + bootstrap_sft_rows
        + language_bridge_rows
    )

    if genome.tokenizer_version == "bpe-v1":
        if row.get("bpe_vocab_target") is not None:
            bpe_vocab_target = int(row["bpe_vocab_target"])
        elif getattr(source_runtime.tokenizer, "version", "") == "bpe-v1":
            bpe_vocab_target = int(source_runtime.tokenizer.vocab_size)
        else:
            bpe_vocab_target = 512
    else:
        bpe_vocab_target = 512

    tokenizer = _tokenizer_for_genome(
        genome,
        source_runtime=source_runtime,
        replay_rows=training_replay_rows,
        pretrain_documents=documents,
        bpe_vocab_size=bpe_vocab_target,
        bpe_max_bytes=min(256_000, corpus_report["bytes"]),
    )
    limits = plan["limits"]
    budget_reason = _research_budget_reason(
        genome,
        max_params=int(limits["max_params"]),
        max_context=int(limits["max_context"]),
        max_width=int(limits["max_width"]),
        max_layers=int(limits["max_layers"]),
        vocab_size=tokenizer.vocab_size,
    )
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if budget_reason:
        result = {
            "ok": False,
            "version": GENERALIST_SWARM_VERSION,
            "candidate_index": int(candidate_index),
            "candidate_id": genome.genome_id,
            "reason": f"research_resource_budget:{budget_reason}",
        }
        _atomic_json(out_root / "result.json", result)
        return result

    requested_steps = max(1, int(steps))
    requested_pretrain_steps = max(0, int(pretrain_steps))
    effective_steps = requested_steps
    effective_pretrain_steps = requested_pretrain_steps
    scale_budget_multiplier = 1.0
    candidate_kind = str(row.get("kind") or "")
    current_scale_params = int(
        (plan.get("progressive_scaling") or {}).get("current_parameters")
        or (plan.get("champion_report") or {}).get("parameters")
        or 1
    )
    candidate_scale_params = int(
        row.get("estimated_parameters")
        or estimate_parameter_count(genome.model_config(tokenizer.vocab_size))
    )
    scale_budget_multiplier = _capacity_budget_multiplier(
        candidate_kind,
        candidate_parameters=candidate_scale_params,
        current_parameters=current_scale_params,
    )
    if scale_budget_multiplier > 1.0:
        effective_steps = max(
            requested_steps,
            int(math.ceil(requested_steps * scale_budget_multiplier)),
        )
    effective_pretrain_steps = _adaptive_pretrain_steps(
        requested_pretrain_steps,
        corpus_bytes=int(corpus_report.get("bytes", 0) or 0),
        stage=int(stage),
        scale_multiplier=scale_budget_multiplier,
    )

    task_domain_weights, pretraining_domain_weights, rescue_report = (
        _language_rescue_weights(
            dict(row.get("domain_weights") or plan.get("domain_weights") or {}),
            documents,
            signals,
        )
    )
    language_collapse = bool(
        "language_collapse" in signals
        or "autoregressive_collapse" in signals
    )
    anti_collapse_weight = (
        0.08 if language_collapse and int(stage) >= 2
        else 0.05 if language_collapse
        else 0.0
    )
    anti_collapse_eos_weight = 1.75 if language_collapse else 1.0

    reports = []
    runtimes: list[tuple[GeneralistRuntime, dict[str, Any]]] = []
    rotating = canary_rows(int(plan["cycle"]))
    for repeat in range(max(1, int(repeat_seeds))):
        seed = (
            _genome_training_seed(genome, namespace="free-speed")
            + int(stage) * 100_003
            + repeat * 997
        ) % 2_000_000_000
        runtime, report = _train_genome(
            genome,
            steps=effective_steps,
            seed=max(1, int(seed)),
            device="cpu",
            replay_rows=training_replay_rows,
            domain_weights=task_domain_weights,
            source_model=source_runtime.model,
            source_tokenizer=source_runtime.tokenizer,
            tokenizer=tokenizer,
            gradient_accumulation_steps=1,
            precision="fp32",
            pretrain_documents=documents,
            pretrain_steps=effective_pretrain_steps,
            pretraining_domain_weights=pretraining_domain_weights,
            repetition_unlikelihood_weight=anti_collapse_weight,
            eos_loss_weight=anti_collapse_eos_weight,
        )
        report["canary_cycle"] = int(plan["cycle"])
        report["canary"] = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            rotating,
            device="cpu",
        )
        report["score"] = _research_score(
            report,
            int(report["parameters"]),
        )
        report["island"] = str(row.get("island") or "efficiency")
        latency = measure_inference_latency(
            runtime.model,
            runtime.config,
            device="cpu",
            sequence_length=min(32, int(runtime.config.context_length)),
            repeats=3,
        )
        report["latency"] = latency
        report["efficiency"] = efficiency_profile(
            runtime.config,
            report,
            latency=latency,
        )
        report["efficiency_bonus"] = efficiency_bonus(report["efficiency"])
        report["score"] = float(report["score"]) + float(report["efficiency_bonus"])
        if isinstance(report.get("fitness"), dict):
            report["fitness"]["efficiency_bonus"] = float(report["efficiency_bonus"])
            report["fitness"]["score_with_efficiency"] = float(report["score"])
        eligible, reason = _research_eligible(
            plan["champion_report"],
            report,
            minimum_loss_gain=0.01,
            max_domain_regression=0.08,
        )
        reports.append({
            "seed": int(seed),
            "eligible": bool(eligible),
            "reason": reason,
            "report": report,
        })
        runtimes.append((runtime, report))

    valid = [
        item for item in reports
        if isinstance(item.get("report"), dict)
        and item["report"].get("finite")
    ]
    if not valid:
        result = {
            "ok": False,
            "version": GENERALIST_SWARM_VERSION,
            "candidate_index": int(candidate_index),
            "candidate_id": genome.genome_id,
            "reason": "no finite candidate reports",
        }
        _atomic_json(out_root / "result.json", result)
        return result

    best_index = max(
        range(len(reports)),
        key=lambda idx: float(
            reports[idx]["report"].get("score", float("-inf"))
        ),
    )
    best_runtime, best_report = runtimes[best_index]
    checkpoint = out_root / "best-checkpoint"
    shutil.rmtree(checkpoint, ignore_errors=True)
    best_runtime.save_checkpoint(
        checkpoint,
        metadata={
            "role": "free_speed_candidate",
            "candidate_id": genome.genome_id,
            "cycle": int(plan["cycle"]),
            "stage": int(stage),
            "previous_stage": (
                int(source_metadata["stage"])
                if source_checkpoint is not None and "stage" in source_metadata
                else None
            ),
            "incumbent_source_cycle": (
                int(source_metadata.get("cycle", 0) or 0)
                if continued_from_incumbent
                else None
            ),
            "continued_from_incumbent": bool(continued_from_incumbent),
            "continued_from_language_fusion": bool(continued_from_language_fusion),
            "language_fusion_target_tokens": (
                int((row.get("fusion_evidence") or {}).get("target_tokens", 0) or 0)
                if continued_from_language_fusion
                else None
            ),
            "cumulative_steps": (
                int(source_metadata.get("cumulative_steps", 0) or 0)
                + int(effective_steps)
            ),
            "production_qualified": False,
        },
    )
    _atomic_json(checkpoint / "research-metrics.json", best_report)

    nlls = [
        float(
            item["report"].get(
                "nll_per_byte",
                item["report"]["loss"],
            )
        )
        for item in valid
    ]
    fitness_scores = [
        float(item["report"].get("score", float("-inf")))
        for item in valid
    ]
    generations = [
        float(item["report"].get("generation_exact_accuracy", 0.0))
        for item in valid
    ]
    generation_similarities = [
        float(item["report"].get("generation_similarity", 0.0) or 0.0)
        for item in valid
    ]
    generation_nonempty_rates = [
        float(item["report"].get("generation_nonempty_rate", 0.0) or 0.0)
        for item in valid
    ]
    generation_repetition_rates = [
        float(item["report"].get("generation_repetition_rate", 0.0) or 0.0)
        for item in valid
    ]
    generation_longest_runs = [
        int(item["report"].get("generation_longest_repeated_token_run", 0) or 0)
        for item in valid
    ]
    degeneration_failures = [
        bool(item["report"].get("generation_pathological_repetition"))
        for item in valid
    ]
    regressions = [
        _domain_regression(plan["champion_report"], item["report"])
        for item in valid
    ]
    result = {
        "ok": True,
        "version": GENERALIST_SWARM_VERSION,
        "cycle": int(plan["cycle"]),
        "stage": int(stage),
        "steps": int(effective_steps),
        "requested_steps": int(requested_steps),
        "pretrain_steps": int(effective_pretrain_steps),
        "requested_pretrain_steps": int(requested_pretrain_steps),
        "scale_budget_multiplier": float(scale_budget_multiplier),
        "cumulative_steps": (
            int(source_metadata.get("cumulative_steps", 0) or 0)
            + int(effective_steps)
        ),
        "continued_from_stage": (
            int(source_metadata["stage"])
            if source_checkpoint is not None and "stage" in source_metadata
            else None
        ),
        "continued_from_incumbent": bool(continued_from_incumbent),
        "continued_from_language_fusion": bool(continued_from_language_fusion),
        "language_fusion_evidence": (
            dict(row.get("fusion_evidence") or {})
            if continued_from_language_fusion
            else None
        ),
        "incumbent_source_cycle": (
            int(source_metadata.get("cycle", 0) or 0)
            if continued_from_incumbent
            else None
        ),
        "candidate_index": int(candidate_index),
        "candidate_id": genome.genome_id,
        "kind": row.get("kind"),
        "island": str(row.get("island") or "efficiency"),
        "domain_weights": dict(row.get("domain_weights") or {}),
        "genome": genome.to_dict(),
        "reports": reports,
        "all_seed_eligible": all(item["eligible"] for item in reports),
        "any_seed_eligible": any(item["eligible"] for item in reports),
        "mean_nll_per_byte": mean(nlls),
        "best_nll_per_byte": min(nlls),
        "mean_generation_accuracy": mean(generations),
        "mean_generation_similarity": mean(generation_similarities),
        "mean_generation_nonempty_rate": mean(generation_nonempty_rates),
        "mean_generation_repetition_rate": mean(generation_repetition_rates),
        "max_generation_repeated_token_run": max(generation_longest_runs, default=0),
        "any_generation_pathological_repetition": any(degeneration_failures),
        "pretrain_budget_multiplier": (
            float(effective_pretrain_steps / requested_pretrain_steps)
            if requested_pretrain_steps
            else 0.0
        ),
        "worst_domain_regression": max(regressions),
        "parameters": int(best_report["parameters"]),
        "score": float(mean(fitness_scores)),
        "mean_fitness_score": float(mean(fitness_scores)),
        "best_fitness_score": float(max(fitness_scores)),
        "fitness": dict(best_report.get("fitness") or {}),
        "efficiency": dict(best_report.get("efficiency") or {}),
        "efficiency_bonus": float(best_report.get("efficiency_bonus", 0.0) or 0.0),
        "corpus": corpus_report,
        "checkpoint_dir": "best-checkpoint",
        "external_pretrained": False,
        "language_bridge_rows": len(language_bridge_rows),
        "bootstrap_replay_sft_rows": len(bootstrap_sft_rows),
        "bootstrap_replay_manifest": bootstrap_replay.manifest,
        "language_rescue": rescue_report,
        "anti_collapse_objective": {
            "enabled": bool(anti_collapse_weight > 0.0),
            "repetition_unlikelihood_weight": float(anti_collapse_weight),
            "eos_loss_weight": float(anti_collapse_eos_weight),
            "decoding_modified": False,
        },
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_vocab_target": int(bpe_vocab_target),
        "language_bpe_probe": row.get("kind") == "language_bpe_probe",
    }
    _atomic_json(out_root / "result.json", result)
    return result


def _scan_results(paths: Iterable[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for raw in paths:
        path = Path(raw)
        candidates = (
            sorted(path.rglob("result.json"))
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                row = _load_json(candidate)
            except Exception:
                continue
            if (
                isinstance(row, dict)
                and row.get("version") == GENERALIST_SWARM_VERSION
                and "candidate_index" in row
            ):
                found.append((candidate, row))
    by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path, row in found:
        idx = int(row["candidate_index"])
        current = by_index.get(idx)
        if current is None or int(row.get("stage", -1)) > int(current[1].get("stage", -1)):
            by_index[idx] = (path, row)
    return list(by_index.values())


def _persist_specialist_checkpoints(
    root: Path,
    scanned: list[tuple[Path, dict[str, Any]]],
    *,
    cycle: int,
) -> dict[str, Any]:
    """Keep the best safe research checkpoint for each specialist island.

    Specialists are research-only and never bypass normal champion/production
    promotion. Persisting them prevents useful domain discoveries from being
    discarded just because another candidate wins the global tournament.
    """
    allowed = {"language", "coding", "reasoning", "tools", "efficiency"}
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, row in scanned:
        island = str(row.get("island") or "")
        if island not in allowed:
            continue
        if bool(row.get("any_generation_pathological_repetition")):
            continue
        if float(row.get("worst_domain_regression", float("inf"))) > 0.18:
            continue
        checkpoint = path.parent / str(row.get("checkpoint_dir") or "best-checkpoint")
        if not checkpoint.is_dir():
            continue
        grouped.setdefault(island, []).append((checkpoint, row))

    specialists_root = root / "specialists"
    specialists_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for island in sorted(allowed):
        options = grouped.get(island) or []
        if not options:
            existing = _load_json(specialists_root / island / "summary.json") if (
                specialists_root / island / "summary.json"
            ).is_file() else None
            if isinstance(existing, dict):
                summary[island] = existing
            continue

        options.sort(
            key=lambda pair: (
                0 if bool(pair[1].get("all_seed_eligible")) else 1,
                -float(pair[1].get("mean_fitness_score", pair[1].get("score", 0.0))),
                float(pair[1].get("mean_nll_per_byte", float("inf"))),
                int(pair[1].get("parameters", 1 << 60)),
            )
        )
        checkpoint, row = options[0]
        candidate_summary = {
            "island": island,
            "candidate_id": str(row.get("candidate_id") or ""),
            "cycle": int(cycle),
            "stage": int(row.get("stage", 0) or 0),
            "parameters": int(row.get("parameters", 0) or 0),
            "score": float(row.get("score", 0.0) or 0.0),
            "mean_fitness_score": float(
                row.get("mean_fitness_score", row.get("score", 0.0)) or 0.0
            ),
            "mean_nll_per_byte": float(row.get("mean_nll_per_byte", 0.0) or 0.0),
            "mean_generation_similarity": float(
                row.get("mean_generation_similarity", 0.0) or 0.0
            ),
            "mean_generation_repetition_rate": float(
                row.get("mean_generation_repetition_rate", 0.0) or 0.0
            ),
            "all_seed_eligible": bool(row.get("all_seed_eligible")),
            "any_seed_eligible": bool(row.get("any_seed_eligible")),
            "any_generation_pathological_repetition": bool(
                row.get("any_generation_pathological_repetition")
            ),
            "worst_domain_regression": float(
                row.get("worst_domain_regression", 1_000_000.0) or 0.0
            ),
            "research_only": True,
            "production_qualified": False,
            "external_pretrained": False,
            "all_seed_eligible": bool(row.get("all_seed_eligible")),
            "any_generation_pathological_repetition": bool(
                row.get("any_generation_pathological_repetition")
            ),
            "genome": dict(row.get("genome") or {}),
        }
        destination = specialists_root / island
        existing_summary = (
            _load_json(destination / "summary.json")
            if (destination / "summary.json").is_file()
            else {}
        )
        existing_score = float(
            existing_summary.get("mean_fitness_score", float("-inf"))
            if isinstance(existing_summary, dict)
            else float("-inf")
        )
        if candidate_summary["mean_fitness_score"] > existing_score:
            tmp = specialists_root / f".{island}-next"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(checkpoint, tmp)
            _atomic_json(tmp / "summary.json", candidate_summary)
            backup = specialists_root / f".{island}-old"
            shutil.rmtree(backup, ignore_errors=True)
            if destination.exists():
                destination.replace(backup)
            tmp.replace(destination)
            shutil.rmtree(backup, ignore_errors=True)
            summary[island] = candidate_summary
        elif isinstance(existing_summary, dict):
            summary[island] = existing_summary
    return {
        "mode": "system_sparse_experts",
        "max_active_experts": 1,
        "specialists": summary,
        "research_only": True,
    }


def select_survivors(
    result_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    survivors: int,
) -> dict[str, Any]:
    rows = [
        row for _path, row in _scan_results(result_paths)
        if row.get("ok")
    ]
    if not rows:
        raise RuntimeError("Generalist swarm reducer received no valid reports")

    # Fail early on strong regressions, then rank by real generation first,
    # held-out NLL second, and efficiency third. Eligibility is preferred but
    # not mandatory in early stages so promising undertrained models survive.
    safe = [
        row for row in rows
        if float(row.get("worst_domain_regression", float("inf"))) <= 0.18
        and not bool(row.get("any_generation_pathological_repetition"))
    ]
    if not safe:
        safe = rows
    rank_key = lambda row: (
        0 if row.get("any_seed_eligible") else 1,
        -float(row.get("mean_fitness_score", row.get("score", 0.0))),
        -float(row.get("mean_generation_accuracy", 0.0)),
        -float(row.get("mean_generation_similarity", 0.0)),
        -float(row.get("mean_generation_nonempty_rate", 0.0)),
        float(row.get("mean_generation_repetition_rate", 1.0)),
        float(row.get("mean_nll_per_byte", float("inf"))),
        int(row.get("parameters", 1 << 60)),
        int(row["candidate_index"]),
    )
    safe.sort(key=rank_key)
    survivor_count = max(1, int(survivors))
    selected = safe[:survivor_count]
    def is_capacity_probe(row: dict[str, Any]) -> bool:
        kind = str(row.get("kind") or "")
        return (
            kind.startswith("progressive_scale")
            or kind == "architecture_capacity_scale"
            or kind == "architecture_capacity_plus_structure"
            or kind == "architecture_incumbent"
        )

    fusion_rows = [
        row for row in safe
        if str(row.get("kind") or "") == "language_fusion"
    ]
    protected_language_fusion = False
    if (
        survivor_count >= 2
        and fusion_rows
        and not any(str(row.get("kind") or "") == "language_fusion" for row in selected)
    ):
        # The fusion lane receives one research slot only after passing the same
        # early safety filter as every other candidate. It cannot bypass final
        # eligibility or the external reducer.
        fusion_rows.sort(key=rank_key)
        selected[-1] = fusion_rows[0]
        selected.sort(key=rank_key)
        protected_language_fusion = True

    progressive_rows = [
        row for row in safe
        if is_capacity_probe(row)
    ]
    protected_progressive_scale = False
    if (
        survivor_count >= 2
        and not protected_language_fusion
        and progressive_rows
        and not any(is_capacity_probe(row) for row in selected)
    ):
        # Only candidates that already survived the safety filter are eligible
        # for protected exploration. This reserves one slot for the capacity
        # hypothesis without bypassing repetition/domain-regression checks.
        progressive_rows.sort(key=rank_key)
        selected[-1] = progressive_rows[0]
        selected.sort(key=rank_key)
        protected_progressive_scale = True
    payload = {
        "ok": True,
        "version": GENERALIST_SWARM_VERSION,
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
        "protected_progressive_scale": protected_progressive_scale,
        "protected_language_fusion": protected_language_fusion,
    }
    _atomic_json(Path(output_path), payload)
    return payload


def finalize_swarm(
    state_dir: str | Path,
    plan_path: str | Path,
    result_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    allow_direct_promotion: bool = True,
    persist_research_checkpoints: bool = True,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    plan = _load_json(plan_path)
    scanned = [
        (path, row)
        for path, row in _scan_results(result_paths)
        if row.get("ok")
    ]
    finalists = [
        (path, row)
        for path, row in scanned
        if row.get("all_seed_eligible")
    ]
    if persist_research_checkpoints:
        specialist_state = _persist_specialist_checkpoints(
            root,
            scanned,
            cycle=int(plan["cycle"]),
        )
    else:
        shutil.rmtree(root / "specialists", ignore_errors=True)
        specialist_state = {
            "mode": "single_lineage_evidence_only",
            "max_active_experts": 1,
            "specialists": {},
            "research_only": True,
        }
        shutil.rmtree(root / "latest-research", ignore_errors=True)

    # Optionally persist the best Stage-3 research checkpoint independently from the
    # production champion. This is explicitly research-only: the Android lab
    # can inspect cycle-to-cycle progress without bypassing promotion gates.
    latest_research_summary = None
    if scanned:
        research_ranked = sorted(
            scanned,
            key=lambda pair: (
                -float(pair[1].get("mean_fitness_score", pair[1].get("score", 0.0))),
                -float(pair[1].get("mean_generation_accuracy", 0.0)),
                -float(pair[1].get("mean_generation_similarity", 0.0)),
                -float(pair[1].get("mean_generation_nonempty_rate", 0.0)),
                float(pair[1].get("mean_generation_repetition_rate", 1.0)),
                float(pair[1].get("worst_domain_regression", float("inf"))),
                float(pair[1].get("mean_nll_per_byte", float("inf"))),
                int(pair[1].get("parameters", 1 << 60)),
            ),
        )
        research_result_path, research_row = research_ranked[0]
        research_checkpoint = (
            research_result_path.parent
            / str(research_row.get("checkpoint_dir") or "best-checkpoint")
        )
        if research_checkpoint.is_dir():
            latest_root = root / "latest-research"
            if persist_research_checkpoints:
                shutil.rmtree(latest_root, ignore_errors=True)
                shutil.copytree(research_checkpoint, latest_root)
            latest_research_summary = {
                "candidate_id": str(research_row.get("candidate_id") or ""),
                "kind": research_row.get("kind"),
                "cycle": int(plan["cycle"]),
                "stage": int(research_row.get("stage", 0) or 0),
                "parameters": int(research_row.get("parameters", 0) or 0),
                "score": float(research_row.get("score", 0.0) or 0.0),
                "mean_fitness_score": float(
                    research_row.get("mean_fitness_score", research_row.get("score", 0.0)) or 0.0
                ),
                "mean_nll_per_byte": float(
                    research_row.get("mean_nll_per_byte", 0.0) or 0.0
                ),
                "mean_generation_accuracy": float(
                    research_row.get("mean_generation_accuracy", 0.0) or 0.0
                ),
                "mean_generation_similarity": float(
                    research_row.get("mean_generation_similarity", 0.0) or 0.0
                ),
                "mean_generation_nonempty_rate": float(
                    research_row.get("mean_generation_nonempty_rate", 0.0) or 0.0
                ),
                "mean_generation_repetition_rate": float(
                    research_row.get("mean_generation_repetition_rate", 0.0) or 0.0
                ),
                "max_generation_repeated_token_run": int(
                    research_row.get("max_generation_repeated_token_run", 0) or 0
                ),
                "any_generation_pathological_repetition": bool(
                    research_row.get("any_generation_pathological_repetition")
                ),
                "worst_domain_regression": float(
                    research_row.get("worst_domain_regression", 0.0) or 0.0
                ),
                "all_seed_eligible": bool(research_row.get("all_seed_eligible")),
                "any_seed_eligible": bool(research_row.get("any_seed_eligible")),
                "tokenizer_vocab_size": int(
                    research_row.get("tokenizer_vocab_size", 0) or 0
                ),
                "language_bridge_rows": int(
                    research_row.get("language_bridge_rows", 0) or 0
                ),
                "external_pretrained": False,
                "research_only": True,
            }
            if persist_research_checkpoints:
                _atomic_json(
                    latest_root / "research-summary.json",
                    latest_research_summary,
                )

    promoted = False
    winner_summary = None
    promotion_candidate = None
    reason = "no finalist passed every independent research gate"
    if finalists:
        finalists.sort(
            key=lambda pair: (
                -float(pair[1].get("mean_fitness_score", pair[1].get("score", 0.0))),
                -float(pair[1].get("mean_generation_accuracy", 0.0)),
                -float(pair[1].get("mean_generation_similarity", 0.0)),
                -float(pair[1].get("mean_generation_nonempty_rate", 0.0)),
                float(pair[1].get("mean_generation_repetition_rate", 1.0)),
                float(pair[1].get("mean_nll_per_byte", float("inf"))),
                int(pair[1].get("parameters", 1 << 60)),
            )
        )
        result_path, winner = finalists[0]
        checkpoint = result_path.parent / str(winner.get("checkpoint_dir") or "best-checkpoint")
        runtime = GeneralistRuntime.from_checkpoint(checkpoint, device="cpu")
        genome = GeneralistGenome(**dict(winner["genome"])).validate()

        # Recompute validation in the external reducer. Worker-reported metrics
        # are useful for ranking but cannot directly authorize promotion.
        verified = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            validation_rows(),
            device="cpu",
        )
        verified["parameters"] = parameter_count(runtime.model)
        verified["score"] = _research_score(
            verified,
            verified["parameters"],
        )
        verified["canary_cycle"] = int(plan["cycle"])
        verified["canary"] = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            canary_rows(int(plan["cycle"])),
            device="cpu",
        )
        verified["score"] = _research_score(
            verified,
            verified["parameters"],
        )
        eligible, verify_reason = _research_eligible(
            plan["champion_report"],
            verified,
            minimum_loss_gain=0.01,
            max_domain_regression=0.08,
        )
        if eligible and str(winner.get("kind") or "") == "language_fusion":
            plan_candidate = next(
                (
                    row for row in (plan.get("candidates") or [])
                    if str(row.get("candidate_id") or "") == genome.genome_id
                ),
                {},
            )
            baseline_language = dict(plan_candidate.get("fusion_evidence") or {})
            current_language = evaluate_phase5_language(runtime)
            retained, retention_reasons = _language_fusion_retention_gate(
                baseline_language,
                current_language,
            )
            verified["phase5_language_retention"] = {
                "passed": bool(retained),
                "baseline": baseline_language,
                "current": current_language,
                "reasons": retention_reasons,
            }
            if not retained:
                eligible = False
                verify_reason = (
                    "language_fusion_retention_rejected:"
                    + "; ".join(retention_reasons)
                )
        if eligible:
            approved_summary = {
                "candidate_id": genome.genome_id,
                "kind": winner.get("kind"),
                "parameters": verified["parameters"],
                "score": verified["score"],
                "nll_per_byte": verified.get("nll_per_byte"),
                "checkpoint_dir": str(winner.get("checkpoint_dir") or "best-checkpoint"),
                "result_path": str(result_path),
                "genome": genome.to_dict(),
            }
            if allow_direct_promotion:
                _save_champion(root, genome, runtime, verified)
                promoted = True
                reason = verify_reason
                winner_summary = approved_summary
            else:
                # Architecture research may prove that a topology is better,
                # but it must not replace AIRI with a separately trained brain.
                # The caller must migrate the newest live-lineage checkpoint
                # into this topology and re-run retention gates first.
                promotion_candidate = approved_summary
                reason = "external_reducer_approved_for_single_lineage_migration"
        else:
            reason = f"external_reducer_rejected:{verify_reason}"

    loaded = _load_champion(root, device="cpu")
    if loaded is None:
        raise RuntimeError("Generalist swarm lost champion state during finalization")
    champion_genome, champion_runtime = loaded
    champion_report = _grouped_validation(
        champion_runtime.model,
        champion_runtime.tokenizer,
        validation_rows(),
        device="cpu",
    )
    champion_report["parameters"] = parameter_count(champion_runtime.model)
    champion_report["score"] = _research_score(
        champion_report,
        champion_report["parameters"],
    )
    cycle = int(plan["cycle"])
    rotating = canary_rows(cycle)
    champion_report["canary_cycle"] = cycle
    champion_report["canary"] = _grouped_validation(
        champion_runtime.model,
        champion_runtime.tokenizer,
        rotating,
        device="cpu",
    )
    champion_report["score"] = _research_score(
        champion_report,
        champion_report["parameters"],
    )
    replay_rows = CurriculumMemory(root, max_rows=20_000).rows()
    replay_prompts = {
        str(row.messages[0].get("content", ""))
        for row in replay_rows
        if row.messages
    }
    rotating_prompts = {
        str(row.messages[0].get("content", ""))
        for row in rotating
        if row.messages
    }

    lab_plan = plan.get("airi_pc_lab") or {}
    lab_snapshot = lab_plan.get("snapshot") or {
        "version": "airi-pc-lab-v1",
        "mode": "read_only_sandbox",
        "modules": [],
        "capabilities": [],
        "denied_capabilities": [],
    }
    champion_lab_probe = run_airi_pc_lab_probe(
        champion_runtime,
        lab_snapshot,
    )
    champion_experience = record_verified_lab_experience(
        root,
        champion_lab_probe,
        cycle=cycle,
        source="champion",
    )
    airi_pc_lab_report = {
        "version": str(lab_snapshot.get("version") or "airi-pc-lab-v1"),
        "mode": str(lab_snapshot.get("mode") or "read_only_sandbox"),
        "snapshot": lab_snapshot,
        "learning": lab_plan.get("learning") or {},
        "champion": champion_lab_probe,
        "champion_experience": champion_experience,
    }
    latest_runtime_path = root / "latest-research"
    if latest_runtime_path.is_dir():
        try:
            latest_runtime = GeneralistRuntime.from_checkpoint(
                latest_runtime_path,
                device="cpu",
            )
            research_probe = run_airi_pc_lab_probe(
                latest_runtime,
                lab_snapshot,
            )
            airi_pc_lab_report["research"] = research_probe
            airi_pc_lab_report["research_experience"] = (
                record_verified_lab_experience(
                    root,
                    research_probe,
                    cycle=cycle,
                    source="latest_research",
                )
            )
        except Exception as exc:
            airi_pc_lab_report["research"] = {
                "ok": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
    _atomic_json(root / "airi-pc-lab-report.json", airi_pc_lab_report)

    previous_status = {}
    try:
        previous_status = _load_json(root / "status.json")
        if not isinstance(previous_status, dict):
            previous_status = {}
    except Exception:
        previous_status = {}

    status = {
        "ok": True,
        "version": GENERALIST_SWARM_VERSION,
        "cycle": int(plan["cycle"]),
        "promoted": promoted,
        "promotion_reason": reason,
        "winner": winner_summary,
        "promotion_candidate": promotion_candidate,
        "latest_research": latest_research_summary,
        "phase5_bootstrap": previous_status.get("phase5_bootstrap"),
        "lineage": previous_status.get("lineage"),
        "architecture_migration": previous_status.get("architecture_migration"),
        "champion": champion_genome.to_dict(),
        "champion_report": champion_report,
        "signals": plan.get("signals") or [],
        "mathesis": plan.get("mathesis"),
        "airi_pc_lab": airi_pc_lab_report,
        "curriculum_memory": plan.get("curriculum"),
        "adaptive_curriculum": {
            "domain_weights": plan.get("domain_weights") or {},
            "active_learning": plan.get("active_learning") or {},
        },
        "sparse_experts": {
            **dict(plan.get("sparse_experts") or {}),
            **dict(specialist_state or {}),
        },
        "self_play": plan.get("self_play") or {},
        "internal_distillation": plan.get("internal_distillation") or {},
        "compression": plan.get("compression") or {},
        "rotating_canary": {
            "cycle": cycle,
            "domains": [row.domain for row in rotating],
            "training_overlap": sorted(rotating_prompts & replay_prompts),
        },
        "grounded_pretraining": {
            "adaptive_budget": True,
            "max_steps_per_stage": 48,
            "domain_balanced_sampling": True,
            "corpus": {
                "enabled": True,
                "automatic_data_growth": bool(
                    (plan.get("data_growth") or {}).get("ok")
                ),
            },
        },
        "automatic_data_growth": plan.get("data_growth"),
        "converged_champion": plan.get("converged_champion"),
        "progressive_scaling": plan.get("progressive_scaling"),
        "progressive_tokenizer": plan.get("progressive_tokenizer"),
        "plateau": plan.get("plateau"),
        "trials": [row for _path, row in scanned],
        "policy": {
            "research_only": True,
            "latest_research_checkpoint_isolated": True,
            "airi_pc_lab": {
                "mode": "read_only_sandbox",
                "host_actions": False,
                "training_from_verified_snapshot": True,
                "successful_tool_use_recorded": True,
            },
            "parallel_swarm": "8->4->2",
            "candidate_can_self_promote": False,
            "direct_research_checkpoint_promotion": bool(allow_direct_promotion),
            "persistent_research_checkpoints": bool(persist_research_checkpoints),
            "single_lineage_migration_required": not bool(allow_direct_promotion),
            "external_reducer": True,
            "production_qualification_separate": True,
            "external_pretrained": False,
            "adaptive_curriculum": True,
            "progressive_scaling_max_parameters": 20_000_000,
            "automatic_best_of_both_fusion": {
                "enabled": True,
                "language_checkpoint_competes_in_swarm": True,
                "self_promotion": False,
                "protected_reducer_slot_requires_safety_filter": True,
            },
            "scale_probe_retention": "reserve one safe scale survivor through reductions",
            "scale_budget_adaptive": True,
            "adaptive_pretraining_budget": {
                "enabled": True,
                "max_steps_per_stage": 48,
            },
            "domain_balanced_pretraining": True,
            "language_gap_routing": True,
            "autoregressive_similarity_ranking": True,
            "inherited_sft_lr_cap": 0.001,
            "weight_inheritance": "layout-aware Net2Grow + identity residual depth expansion",
            "automatic_data_growth": "permissive SPDX + immutable commit + quarantine + hash + quality filter",
            "continual_learning": {
                "enabled": True,
                "full_replay": True,
                "domain_balanced_base_replay": True,
                "curriculum_max_rows": 4000,
                "gradient_accumulation_steps": 1,
                "precision": "fp32",
            },
            "tokenizer_research": {
                "allowed": ["byte-v1", "bpe-v1"],
                "comparison_metric": "nll_per_byte",
                "protected_eval_rows_excluded_from_tokenizer_training": True,
                "bpe_vocab_max": 1024,
                "growth_per_probe": 128,
                "refresh_max_bytes": 256000,
                "progressive_refresh": "single bounded language probe",
            },
            "rotating_canary": {
                "enabled": True,
                "training_excluded": True,
                "domains": len(DOMAINS),
            },
            "architecture_weight_transfer": {
                "enabled": True,
                "exact_name_and_shape": True,
                "layout_aware_net2grow": True,
                "identity_residual_depth_expansion": True,
                "byte_to_bpe_embedding_migration": True,
            },
        },
        "updated_at": time.time(),
    }
    _atomic_json(root / "status.json", status)
    with (root / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(status, ensure_ascii=False, sort_keys=True) + "\n"
        )
    _atomic_json(Path(output_path), status)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AIRI Generalist Free-Speed parallel research swarm"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("state_dir")
    prepare.add_argument("output")
    prepare.add_argument("--mathesis-state")
    prepare.add_argument("--population-size", type=int, default=8)
    prepare.add_argument("--max-params", type=int, default=7_000_000)
    prepare.add_argument("--max-context", type=int, default=512)
    prepare.add_argument("--max-width", type=int, default=384)
    prepare.add_argument("--max-layers", type=int, default=10)
    prepare.add_argument("--curriculum-max-rows", type=int, default=4000)
    prepare.add_argument("--data-max-new-bytes", type=int, default=8_000_000)
    prepare.add_argument("--data-max-total-bytes", type=int, default=50_000_000)
    prepare.add_argument("--no-data-growth", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("plan")
    worker.add_argument("state_dir")
    worker.add_argument("repo_root")
    worker.add_argument("output_dir")
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--stage", type=int, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--repeat-seeds", type=int, default=1)
    worker.add_argument("--pretrain-steps", type=int, default=1)
    worker.add_argument("--max-repo-bytes", type=int, default=5_000_000)
    worker.add_argument("--max-external-bytes", type=int, default=20_000_000)
    worker.add_argument("--source-checkpoint")

    select = sub.add_parser("select")
    select.add_argument("output")
    select.add_argument("results", nargs="+")
    select.add_argument("--survivors", type=int, required=True)

    final = sub.add_parser("finalize")
    final.add_argument("state_dir")
    final.add_argument("plan")
    final.add_argument("output")
    final.add_argument("results", nargs="+")

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        result = prepare_swarm(
            args.state_dir,
            args.output,
            population_size=args.population_size,
            max_params=args.max_params,
            max_context=args.max_context,
            max_width=args.max_width,
            max_layers=args.max_layers,
            curriculum_max_rows=args.curriculum_max_rows,
            mathesis_state_dir=args.mathesis_state,
            grow_data=not args.no_data_growth,
            data_max_new_bytes=args.data_max_new_bytes,
            data_max_total_bytes=args.data_max_total_bytes,
            github_token=os.environ.get("GITHUB_TOKEN"),
        )
    elif args.cmd == "worker":
        result = run_candidate(
            args.plan,
            args.state_dir,
            args.repo_root,
            args.output_dir,
            candidate_index=args.index,
            stage=args.stage,
            steps=args.steps,
            repeat_seeds=args.repeat_seeds,
            pretrain_steps=args.pretrain_steps,
            max_repo_bytes=args.max_repo_bytes,
            max_external_bytes=args.max_external_bytes,
            source_checkpoint=args.source_checkpoint,
        )
    elif args.cmd == "select":
        result = select_survivors(
            args.results,
            args.output,
            survivors=args.survivors,
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
