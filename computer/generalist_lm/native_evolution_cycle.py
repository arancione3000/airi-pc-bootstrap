from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from .mathesis_bridge import mathesis_signals
from .native_data import load_native_corpus, train_native_bpe
from .native_evaluation import (
    evaluate_native_checkpoint,
    native_continual_decision,
    native_promotion_decision,
)
from .native_evolution import (
    NATIVE_EVOLUTION_VERSION,
    NativeEvolutionGenome,
    NativeMutation,
    apply_mutation,
    current_training_step,
    generate_native_challengers,
    genome_from_checkpoint,
)
from .native_foundation import (
    NATIVE_TOKENIZER_FILENAME,
    create_native_root_checkpoint,
    native_checkpoint_status,
)
from .native_online_research import discover_native_research
from .lattice_research_cycle import run_lattice_research_cycle
from .native_training import train_native_foundation


NATIVE_EVOLUTION_STATE_VERSION = "native-evolution-state-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _copy_checkpoint(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    status = native_checkpoint_status(target)
    if not status.get("ok"):
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(str(status.get("reason") or "copied Native checkpoint failed integrity"))


def _load_genome(path: Path) -> NativeEvolutionGenome | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return NativeEvolutionGenome(**raw).validate()
    except Exception:
        return None


def _save_genome(path: Path, genome: NativeEvolutionGenome) -> None:
    _atomic_json(path, genome.to_dict())


def _seed_for(*parts: Any) -> int:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    return 1 + value % 2_000_000_000


def _promote_checkpoint(
    root: Path,
    source: Path,
    genome: NativeEvolutionGenome,
) -> dict[str, Any]:
    champion = root / "champion"
    staging = root / "champion.next"
    backup = root / "champion.prev"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    _copy_checkpoint(source, staging)

    try:
        if champion.exists():
            champion.rename(backup)
        staging.rename(champion)
        verified = native_checkpoint_status(champion)
        if not verified.get("ok"):
            raise RuntimeError(str(verified.get("reason") or "promoted checkpoint failed integrity"))
        _save_genome(root / "champion-genome.json", genome)
        shutil.rmtree(backup, ignore_errors=True)
        return verified
    except Exception:
        if champion.exists() and backup.exists():
            shutil.rmtree(champion, ignore_errors=True)
            backup.rename(champion)
        elif backup.exists() and not champion.exists():
            backup.rename(champion)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _mathesis_signal_report(state_dir: str | Path | None) -> dict[str, Any]:
    if not state_dir:
        return {
            "ok": False,
            "signals": [],
            "reason": "MATHESIS state not configured",
        }
    path = Path(state_dir).expanduser().resolve()
    if not path.exists():
        return {
            "ok": False,
            "signals": [],
            "reason": "MATHESIS state does not exist",
        }
    try:
        return mathesis_signals(path)
    except Exception as exc:
        return {
            "ok": False,
            "signals": [],
            "reason": f"{type(exc).__name__}:{exc}",
        }


def _candidate_tokenizer(
    candidate: NativeEvolutionGenome,
    mutation: NativeMutation,
    *,
    champion_checkpoint: Path,
    corpus_manifest: str | Path,
    allowed_roots: Sequence[str | Path],
    trial_root: Path,
) -> tuple[NativeEvolutionGenome, str | None, dict[str, Any] | None]:
    if not mutation.requires_retokenization:
        if candidate.tokenizer_version == "bpe-v1":
            tokenizer = champion_checkpoint / NATIVE_TOKENIZER_FILENAME
            if not tokenizer.is_file():
                raise FileNotFoundError("champion BPE tokenizer is missing")
            return candidate, str(tokenizer), None
        return candidate, None, None

    corpus = load_native_corpus(
        corpus_manifest,
        allowed_roots=allowed_roots,
    )
    tokenizer_path = trial_root / "challenger-tokenizer.json"
    result = train_native_bpe(
        corpus,
        tokenizer_path,
        vocab_size=int(candidate.vocab_size),
        min_frequency=1,
        max_bytes=50_000_000,
    )
    adjusted = replace(
        candidate,
        tokenizer_version="bpe-v1",
        vocab_size=int(result["vocab_size"]),
    ).validate()
    return adjusted, str(tokenizer_path), result


def _trial_training_config(
    genome: NativeEvolutionGenome,
    *,
    max_steps: int,
    seed: int,
    device: str,
    precision: str,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    max_eval_blocks: int,
    validation_fraction: float,
):
    return genome.training_config(
        max_steps=max_steps,
        seed=seed,
        device=device,
        precision=precision,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_eval_blocks=max_eval_blocks,
        validation_fraction=validation_fraction,
        save_optimizer_state=True,
    )


def run_native_evolution_cycle(
    state_dir: str | Path,
    seed_checkpoint: str | Path,
    corpus_manifest: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    mathesis_state_dir: str | Path | None = None,
    online_research: bool = True,
    research_override: dict[str, Any] | None = None,
    github_token: str | None = None,
    challenger_count: int = 3,
    steps_per_trial: int = 4,
    reinit_training_steps: int = 0,
    device: str = "cpu",
    precision: str = "fp32",
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    validation_fraction: float = 0.2,
    max_eval_blocks: int = 32,
    verifier_seed: int = 431,
    minimum_gain: float = 0.002,
    max_domain_regression: float = 0.05,
    max_parameter_ratio: float = 1.5,
    max_parameters: int | None = None,
    keep_trial_checkpoints: bool = False,
    lattice_research: bool = False,
    lattice_population_size: int = 8,
    lattice_empirical_candidates: int = 2,
    lattice_benchmark_steps: int = 4,
    lattice_repeat_seeds: int = 2,
) -> dict[str, Any]:
    """Run one proof-gated AIRI Native self-evolution cycle.

    Remote research can select only predefined mutation families. Challengers
    cannot modify this verifier, promote themselves, execute remote code, or
    replace the champion without independently recomputed held-out metrics.
    """
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    champion_path = root / "champion"
    genome_path = root / "champion-genome.json"

    previous: dict[str, Any] = {}
    try:
        previous = json.loads((root / "status.json").read_text(encoding="utf-8"))
    except Exception:
        previous = {}
    cycle = max(0, int(previous.get("cycle", 0) or 0)) + 1

    if not champion_path.exists():
        source = Path(seed_checkpoint).expanduser().resolve()
        _copy_checkpoint(source, champion_path)
    champion_status = native_checkpoint_status(champion_path)
    if not champion_status.get("ok"):
        raise RuntimeError(str(champion_status.get("reason") or "invalid Native champion"))

    champion_genome = _load_genome(genome_path)
    if champion_genome is None:
        champion_genome = genome_from_checkpoint(
            champion_path,
            generation=max(0, cycle - 1),
        )
        _save_genome(genome_path, champion_genome)

    mathesis = _mathesis_signal_report(mathesis_state_dir)
    signals = list(dict.fromkeys(str(row) for row in (mathesis.get("signals") or [])))

    if research_override is not None:
        research = dict(research_override)
    elif online_research:
        research = discover_native_research(
            signals=signals,
            github_token=github_token or os.environ.get("GITHUB_TOKEN"),
        )
    else:
        research = {
            "ok": False,
            "version": "offline",
            "evidence": [],
            "tag_counts": {},
            "errors": [],
            "remote_code_execution": False,
            "remote_content_trusted": False,
        }
    _atomic_json(root / "online-research.json", research)

    current_step = current_training_step(champion_path)
    trial_steps = max(1, int(steps_per_trial))
    target_step = current_step + trial_steps
    trial_root = root / "trials" / f"cycle-{cycle:06d}"
    shutil.rmtree(trial_root, ignore_errors=True)
    trial_root.mkdir(parents=True, exist_ok=True)

    eval_kwargs = {
        "allowed_roots": allowed_roots,
        "validation_fraction": float(validation_fraction),
        "seed": int(verifier_seed),
        "batch_size": max(1, int(micro_batch_size)),
        "max_eval_blocks": max(1, int(max_eval_blocks)),
        "device": device,
    }
    champion_eval = evaluate_native_checkpoint(
        champion_path,
        corpus_manifest,
        **eval_kwargs,
    )

    matched_train_seed = _seed_for(cycle, "matched-train")
    scratch_train_seed = _seed_for(cycle, "scratch-train")
    scratch_root_seed = _seed_for(cycle, "scratch-root")

    control_dir = trial_root / "control"
    control_config = _trial_training_config(
        champion_genome,
        max_steps=target_step,
        seed=matched_train_seed,
        device=device,
        precision=precision,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_eval_blocks=max_eval_blocks,
        validation_fraction=validation_fraction,
    )
    control_training = train_native_foundation(
        champion_path,
        corpus_manifest,
        allowed_roots=allowed_roots,
        output_dir=control_dir,
        config=control_config,
    )
    control_eval = evaluate_native_checkpoint(
        control_dir,
        corpus_manifest,
        **eval_kwargs,
    )

    candidates = generate_native_challengers(
        champion_genome,
        research=research,
        mathesis_signals=signals,
        count=max(1, int(challenger_count)),
        exploration_offset=max(0, cycle - 1),
        max_parameters=max_parameters,
        max_parameter_ratio=max_parameter_ratio,
    )

    trials: list[dict[str, Any]] = []
    eligible: list[tuple[float, Path, NativeEvolutionGenome, dict[str, Any]]] = []

    for index, (mutation, original_candidate) in enumerate(candidates):
        candidate = original_candidate
        row: dict[str, Any] = {
            "index": index,
            "mutation": mutation.to_dict(),
            "genome": candidate.to_dict(),
            "eligible": False,
        }
        candidate_root = trial_root / f"candidate-{index:02d}"
        try:
            if mutation.requires_reinit:
                if int(reinit_training_steps) <= 0:
                    row["reason"] = "reinitialization mutation is proposal-only without a scratch retrain budget"
                    trials.append(row)
                    continue

                candidate, tokenizer_path, tokenizer_report = _candidate_tokenizer(
                    candidate,
                    mutation,
                    champion_checkpoint=champion_path,
                    corpus_manifest=corpus_manifest,
                    allowed_roots=allowed_roots,
                    trial_root=candidate_root,
                )
                row["genome"] = candidate.to_dict()
                if tokenizer_report is not None:
                    row["tokenizer"] = tokenizer_report

                scratch_root = candidate_root / "root"
                trained_dir = candidate_root / "trained"
                create_native_root_checkpoint(
                    scratch_root,
                    candidate.foundation_config(),
                    root_seed=scratch_root_seed,
                    tokenizer_path=tokenizer_path,
                )
                scratch_steps = max(1, int(reinit_training_steps))
                train_config = _trial_training_config(
                    candidate,
                    max_steps=scratch_steps,
                    seed=scratch_train_seed,
                    device=device,
                    precision=precision,
                    micro_batch_size=micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_eval_blocks=max_eval_blocks,
                    validation_fraction=validation_fraction,
                )
                training = train_native_foundation(
                    scratch_root,
                    corpus_manifest,
                    allowed_roots=allowed_roots,
                    output_dir=trained_dir,
                    config=train_config,
                )
                evaluation = evaluate_native_checkpoint(
                    trained_dir,
                    corpus_manifest,
                    **eval_kwargs,
                )
                row["training"] = training
                row["evaluation"] = evaluation
                row["scratch_retrain_steps"] = scratch_steps

                if scratch_steps < target_step:
                    row["reason"] = (
                        "scratch challenger completed a proxy experiment but cannot "
                        "replace a champion trained for a larger cumulative step budget"
                    )
                    row["proxy"] = True
                    trials.append(row)
                    continue
                candidate_dir = trained_dir
            else:
                trained_dir = candidate_root / "trained"
                train_config = _trial_training_config(
                    candidate,
                    max_steps=target_step,
                    seed=matched_train_seed,
                    device=device,
                    precision=precision,
                    micro_batch_size=micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_eval_blocks=max_eval_blocks,
                    validation_fraction=validation_fraction,
                )
                training = train_native_foundation(
                    champion_path,
                    corpus_manifest,
                    allowed_roots=allowed_roots,
                    output_dir=trained_dir,
                    config=train_config,
                )
                evaluation = evaluate_native_checkpoint(
                    trained_dir,
                    corpus_manifest,
                    **eval_kwargs,
                )
                row["training"] = training
                row["evaluation"] = evaluation
                candidate_dir = trained_dir

            accepted, reason = native_promotion_decision(
                champion_eval,
                control_eval,
                row["evaluation"],
                minimum_gain=minimum_gain,
                max_domain_regression=max_domain_regression,
                max_parameter_ratio=max_parameter_ratio,
            )
            row["eligible"] = bool(accepted)
            row["reason"] = reason
            if accepted:
                eligible.append((
                    float(row["evaluation"]["loss"]),
                    candidate_dir,
                    candidate,
                    row,
                ))
        except Exception as exc:
            row["reason"] = f"trial_error:{type(exc).__name__}:{exc}"
        trials.append(row)

    promoted = False
    promotion_kind = "none"
    promotion_reason = "no challenger passed external promotion gates"
    winner_summary: dict[str, Any] | None = None

    if eligible:
        eligible.sort(key=lambda item: (item[0], item[2].parameters, item[2].genome_id))
        _, winner_path, winner_genome, winner_row = eligible[0]
        promoted_status = _promote_checkpoint(root, winner_path, winner_genome)
        promoted = True
        promotion_kind = "challenger"
        promotion_reason = str(winner_row.get("reason"))
        winner_summary = {
            "mutation": winner_row["mutation"],
            "genome": winner_genome.to_dict(),
            "evaluation": winner_row["evaluation"],
            "checkpoint_digest": promoted_status["checkpoint_digest"],
        }
    else:
        continual_ok, continual_reason = native_continual_decision(
            champion_eval,
            control_eval,
            minimum_gain=max(0.0, minimum_gain * 0.5),
            max_domain_regression=max_domain_regression,
        )
        if continual_ok:
            control_genome = apply_mutation(
                champion_genome,
                NativeMutation(
                    name="continual-control",
                    kind="continual",
                    changes={},
                ),
            )
            promoted_status = _promote_checkpoint(root, control_dir, control_genome)
            promoted = True
            promotion_kind = "continual"
            promotion_reason = continual_reason
            winner_summary = {
                "mutation": {
                    "name": "continual-control",
                    "kind": "continual",
                    "changes": {},
                },
                "genome": control_genome.to_dict(),
                "evaluation": control_eval,
                "checkpoint_digest": promoted_status["checkpoint_digest"],
            }
        else:
            promotion_reason = continual_reason

    final_status = native_checkpoint_status(root / "champion")
    final_genome = _load_genome(root / "champion-genome.json")

    lattice_report: dict[str, Any] | None = None
    if lattice_research:
        try:
            lattice_report = run_lattice_research_cycle(
                root / "lattice-research",
                corpus_manifest,
                allowed_roots=allowed_roots,
                cycle=cycle,
                mathesis_signals=signals,
                research=research,
                population_size=max(1, int(lattice_population_size)),
                empirical_candidates=max(1, int(lattice_empirical_candidates)),
                benchmark_steps=max(1, int(lattice_benchmark_steps)),
                max_eval_blocks=min(8, max(2, int(max_eval_blocks))),
                repeat_seeds=max(1, int(lattice_repeat_seeds)),
            )
        except Exception as exc:
            lattice_report = {
                "ok": False,
                "version": "airi-lattice-research-state-v0",
                "cycle": cycle,
                "error": f"{type(exc).__name__}:{exc}",
                "policy": {
                    "canonical_native_transformer_unchanged": True,
                    "failure_isolated_from_native_champion": True,
                },
            }
            _atomic_json(root / "lattice-research-error.json", lattice_report)

    payload = {
        "ok": bool(final_status.get("ok")),
        "version": NATIVE_EVOLUTION_STATE_VERSION,
        "engine_version": NATIVE_EVOLUTION_VERSION,
        "cycle": cycle,
        "promoted": promoted,
        "promotion_kind": promotion_kind,
        "promotion_reason": promotion_reason,
        "winner": winner_summary,
        "champion_before": champion_eval,
        "control": {
            "training": control_training,
            "evaluation": control_eval,
        },
        "champion_after": {
            "checkpoint": final_status,
            "genome": final_genome.to_dict() if final_genome else None,
        },
        "mathesis": mathesis,
        "online_research": {
            "ok": bool(research.get("ok")),
            "version": research.get("version"),
            "evidence_digest": research.get("evidence_digest"),
            "tag_counts": research.get("tag_counts") or {},
            "source_candidates": research.get("source_candidates") or [],
            "errors": research.get("errors") or [],
            "remote_code_execution": False,
            "remote_content_trusted": False,
        },
        "trials": trials,
        "lattice_research": lattice_report,
        "policy": {
            "candidate_can_self_promote": False,
            "verifier_external_to_candidate": True,
            "remote_code_execution": False,
            "external_pretrained": False,
            "equal_budget_control_required": True,
            "scratch_retrain_required_for_architecture_or_tokenizer": True,
        },
    }
    _atomic_json(root / "status.json", payload)
    with (root / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    if not keep_trial_checkpoints:
        shutil.rmtree(trial_root, ignore_errors=True)

    return payload


def main() -> int:
    state = os.environ.get("AIRI_NATIVE_EVOLUTION_STATE", ".ai/native-evolution")
    seed_checkpoint = os.environ.get("AIRI_NATIVE_SEED_CHECKPOINT", "").strip()
    corpus_manifest = os.environ.get("AIRI_NATIVE_CORPUS_MANIFEST", "").strip()
    allowed_root = os.environ.get("AIRI_NATIVE_CORPUS_ROOT", "").strip()
    if not seed_checkpoint or not corpus_manifest or not allowed_root:
        raise SystemExit(
            "AIRI_NATIVE_SEED_CHECKPOINT, AIRI_NATIVE_CORPUS_MANIFEST and "
            "AIRI_NATIVE_CORPUS_ROOT are required"
        )

    result = run_native_evolution_cycle(
        state,
        seed_checkpoint,
        corpus_manifest,
        allowed_roots=[allowed_root],
        mathesis_state_dir=os.environ.get("MATHESIS_STATE_DIR"),
        online_research=os.environ.get("AIRI_NATIVE_ONLINE_RESEARCH", "1") != "0",
        github_token=os.environ.get("GITHUB_TOKEN"),
        challenger_count=max(1, min(int(os.environ.get("AIRI_NATIVE_CHALLENGERS", "3")), 8)),
        steps_per_trial=max(1, min(int(os.environ.get("AIRI_NATIVE_TRIAL_STEPS", "4")), 10000)),
        reinit_training_steps=max(0, min(int(os.environ.get("AIRI_NATIVE_REINIT_STEPS", "0")), 1000000)),
        device=os.environ.get("AIRI_NATIVE_DEVICE", "cpu"),
        precision=os.environ.get("AIRI_NATIVE_PRECISION", "fp32"),
        micro_batch_size=max(1, int(os.environ.get("AIRI_NATIVE_MICRO_BATCH", "1"))),
        gradient_accumulation_steps=max(1, int(os.environ.get("AIRI_NATIVE_GRAD_ACCUM", "1"))),
        validation_fraction=float(os.environ.get("AIRI_NATIVE_VALIDATION_FRACTION", "0.2")),
        max_eval_blocks=max(1, int(os.environ.get("AIRI_NATIVE_MAX_EVAL_BLOCKS", "32"))),
        verifier_seed=int(os.environ.get("AIRI_NATIVE_VERIFIER_SEED", "431")),
        minimum_gain=max(0.0, float(os.environ.get("AIRI_NATIVE_MIN_GAIN", "0.002"))),
        max_domain_regression=max(0.0, float(os.environ.get("AIRI_NATIVE_MAX_DOMAIN_REGRESSION", "0.05"))),
        max_parameter_ratio=max(1.0, float(os.environ.get("AIRI_NATIVE_MAX_PARAMETER_RATIO", "1.5"))),
        max_parameters=(
            int(os.environ["AIRI_NATIVE_MAX_PARAMETERS"])
            if os.environ.get("AIRI_NATIVE_MAX_PARAMETERS")
            else None
        ),
        lattice_research=os.environ.get("AIRI_LATTICE_RESEARCH", "1") != "0",
        lattice_population_size=max(
            1,
            min(int(os.environ.get("AIRI_LATTICE_POPULATION", "8")), 32),
        ),
        lattice_empirical_candidates=max(
            1,
            min(int(os.environ.get("AIRI_LATTICE_EMPIRICAL", "2")), 4),
        ),
        lattice_benchmark_steps=max(
            1,
            min(int(os.environ.get("AIRI_LATTICE_BENCHMARK_STEPS", "4")), 100),
        ),
        lattice_repeat_seeds=max(
            1,
            min(int(os.environ.get("AIRI_LATTICE_REPEAT_SEEDS", "2")), 4),
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
