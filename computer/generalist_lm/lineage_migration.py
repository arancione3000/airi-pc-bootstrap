from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .architecture_ir import ArchitectureSpec, architecture_manifest
from .curriculum import validation_rows
from .curriculum_memory import canary_rows
from .evolution import GeneralistGenome
from .model import parameter_count
from .phase5_diagnostics import degeneration_gate, evaluate_phase5_language
from .research_cycle import (
    _grouped_validation,
    _research_eligible,
    _research_score,
    _save_champion,
    _transfer_compatible_weights,
)
from .runtime import GeneralistRuntime, checkpoint_has_model, checkpoint_model_sha256


LINEAGE_SCHEMA = 1
LINEAGE_VERSION = "airi-single-lineage-v1"


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_matches(genome: GeneralistGenome, runtime: GeneralistRuntime) -> bool:
    try:
        expected = genome.model_config(runtime.tokenizer.vocab_size).to_dict()
    except Exception:
        return False
    return expected == runtime.config.to_dict()


def active_lineage_snapshot(state_dir: str | Path) -> dict[str, Any]:
    """Return the single checkpoint that owns the newest learned AIRI state.

    Once Phase-5 has a valid persisted candidate, that checkpoint is the live
    learning lineage across rung boundaries; the champion is a rollback and
    qualification anchor. If no valid bootstrap candidate exists, fall back to
    the champion. No research challenger is ever returned by this function.
    """
    root = Path(state_dir).expanduser().resolve()
    progress_path = root / "bootstrap-data" / "progress.json"
    candidate_path = root / "bootstrap-data" / "candidate"
    progress = _read_json(progress_path, {})
    if (
        isinstance(progress, dict)
        and checkpoint_has_model(candidate_path)
        and (candidate_path / "config.json").is_file()
    ):
        raw = progress.get("capacity_genome")
        if isinstance(raw, dict):
            genome = GeneralistGenome(**dict(raw)).validate()
            runtime = GeneralistRuntime.from_checkpoint(candidate_path, device="cpu")
            if _config_matches(genome, runtime):
                target = int(progress.get("target_tokens", 0) or 0)
                processed = int(progress.get("tokens_processed", 0) or 0)
                return {
                    "checkpoint": candidate_path,
                    "checkpoint_rel": "bootstrap-data/candidate",
                    "genome": genome,
                    "runtime": runtime,
                    "bootstrap_active": True,
                    "tokens_processed": processed,
                    "target_tokens": target,
                    "parameters": int(parameter_count(runtime.model)),
                    "progress": progress,
                }

    champion_path = root / "champion"
    genome_raw = _read_json(root / "champion-genome.json", {})
    if not isinstance(genome_raw, dict) or not genome_raw:
        raise RuntimeError("single lineage requires champion-genome.json")
    genome = GeneralistGenome(**dict(genome_raw)).validate()
    runtime = GeneralistRuntime.from_checkpoint(champion_path, device="cpu")
    if not _config_matches(genome, runtime):
        raise RuntimeError("champion genome/config mismatch")
    return {
        "checkpoint": champion_path,
        "checkpoint_rel": "champion",
        "genome": genome,
        "runtime": runtime,
        "bootstrap_active": False,
        "tokens_processed": int(progress.get("tokens_processed", 0) or 0)
        if isinstance(progress, dict)
        else 0,
        "target_tokens": int(progress.get("target_tokens", 0) or 0)
        if isinstance(progress, dict)
        else 0,
        "parameters": int(parameter_count(runtime.model)),
        "progress": progress if isinstance(progress, dict) else {},
    }


def adaptive_architecture_parameter_cap(
    state_dir: str | Path,
    requested_cap: int,
    *,
    growth_headroom: float = 1.35,
    absolute_cap: int = 96_000_000,
) -> int:
    """Never let architecture research become smaller than the live AIRI."""
    requested = max(100_000, int(requested_cap))
    try:
        live = active_lineage_snapshot(state_dir)
        live_parameters = max(1, int(live["parameters"]))
    except Exception:
        return min(int(absolute_cap), requested)
    headroom = int(round(live_parameters * max(1.0, float(growth_headroom))))
    return min(int(absolute_cap), max(requested, live_parameters, headroom))


def evaluate_lineage_runtime(runtime: GeneralistRuntime, *, cycle: int) -> dict[str, Any]:
    report = _grouped_validation(
        runtime.model,
        runtime.tokenizer,
        validation_rows(),
        device="cpu",
    )
    report["parameters"] = int(parameter_count(runtime.model))
    report["score"] = _research_score(report, report["parameters"])
    report["canary_cycle"] = max(1, int(cycle))
    report["canary"] = _grouped_validation(
        runtime.model,
        runtime.tokenizer,
        canary_rows(max(1, int(cycle))),
        device="cpu",
    )
    report["score"] = _research_score(report, report["parameters"])
    return report


def _lineage_genome(
    source: GeneralistGenome,
    target: GeneralistGenome,
    cycle: int,
    *,
    target_vocab_size: int,
) -> GeneralistGenome:
    architecture = ArchitectureSpec.from_genome(
        target,
        target_vocab_size=int(target_vocab_size),
    )
    tag = architecture.fingerprint()[:12]
    return replace(
        target,
        generation=max(int(source.generation) + 1, int(target.generation)),
        parent_id=source.genome_id,
        genome_id=f"generalist-lineage-{max(1, int(cycle))}-{tag}",
    ).validate()


def refresh_live_lineage_manifest(
    state_dir: str | Path,
    *,
    reason: str = "training_refresh",
) -> dict[str, Any]:
    """Synchronize lineage metadata with the checkpoint that is actually live.

    This is intentionally cheap in semantics, not authority: it never promotes
    a research checkpoint and never changes weights. It only records the newest
    persisted active checkpoint after training/capacity growth or a completed
    rung so lineage.json cannot drift behind AIRI's real weights.
    """
    root = Path(state_dir).expanduser().resolve()
    live = active_lineage_snapshot(root)
    checkpoint = Path(live["checkpoint"])
    model_sha = checkpoint_model_sha256(checkpoint)
    previous = _read_json(root / "lineage.json", {})
    previous = previous if isinstance(previous, dict) else {}
    lineage_id = str(
        previous.get("lineage_id")
        or (live.get("progress") or {}).get("lineage_id")
        or f"airi-{model_sha[:16]}"
    )
    architecture = ArchitectureSpec.from_genome(
        live["genome"],
        target_vocab_size=int(live["runtime"].tokenizer.vocab_size),
    )
    payload = {
        "schema": LINEAGE_SCHEMA,
        "version": LINEAGE_VERSION,
        "lineage_id": lineage_id,
        "single_active_model": True,
        "active_checkpoint": str(live["checkpoint_rel"]),
        "active_model_sha256": model_sha,
        "active_genome": live["genome"].to_dict(),
        "active_architecture": architecture_manifest(architecture),
        "tokens_processed": int(live.get("tokens_processed", 0) or 0),
        "target_tokens": int(live.get("target_tokens", 0) or 0),
        "parameters": int(live["parameters"]),
        "tokenizer_version": str(live["runtime"].tokenizer.version),
        "tokenizer_vocab_size": int(live["runtime"].tokenizer.vocab_size),
        "bootstrap_active": bool(live["bootstrap_active"]),
        "rollback_snapshots_are_not_competing_models": True,
        "research_challengers_are_temporary": True,
        "migrations": list(previous.get("migrations") or [])[-64:],
        "last_refresh_reason": str(reason),
        "updated_at": time.time(),
    }
    _atomic_json(root / "lineage.json", payload)

    status = _read_json(root / "status.json", {})
    status = dict(status) if isinstance(status, dict) else {}
    status["lineage"] = payload
    _atomic_json(root / "status.json", status)

    progress_path = root / "bootstrap-data" / "progress.json"
    if progress_path.is_file() and bool(live["bootstrap_active"]):
        progress = _read_json(progress_path, {})
        progress["lineage_id"] = lineage_id
        _atomic_json(progress_path, progress)
    return payload



def adopt_verified_descendant(
    state_dir: str | Path,
    *,
    candidate_checkpoint: str | Path,
    candidate_genome: dict[str, Any] | GeneralistGenome,
    cycle: int,
    candidate_id: str,
    research_kind: str | None = None,
    expected_parent_model_sha256: str | None = None,
) -> dict[str, Any]:
    """Adopt a verified trained descendant into the one live AIRI lineage.

    Unlike architecture-only migration, this path keeps the candidate's newly
    trained weights. It is allowed only when the candidate proves it descended
    from the exact live checkpoint that still exists at adoption time. That
    parent-hash lock prevents an independently trained or stale model from
    replacing AIRI while preserving cumulative token/step counters.
    """
    root = Path(state_dir).expanduser().resolve()
    source = active_lineage_snapshot(root)
    source_runtime: GeneralistRuntime = source["runtime"]
    source_genome: GeneralistGenome = source["genome"]
    source_checkpoint = Path(source["checkpoint"])
    source_model_sha = checkpoint_model_sha256(source_checkpoint)

    candidate_path = Path(candidate_checkpoint).expanduser().resolve()
    target_genome = (
        candidate_genome
        if isinstance(candidate_genome, GeneralistGenome)
        else GeneralistGenome(**dict(candidate_genome)).validate()
    )
    candidate_runtime = GeneralistRuntime.from_checkpoint(
        candidate_path,
        device="cpu",
    )
    if not _config_matches(target_genome, candidate_runtime):
        raise RuntimeError("descendant checkpoint does not match candidate genome")

    metadata = _read_json(candidate_path / "metadata.json", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    declared_parent = str(
        metadata.get("lineage_parent_model_sha256")
        or metadata.get("parent_model_sha256")
        or ""
    )
    required_parent = str(expected_parent_model_sha256 or declared_parent or "")
    parent_lock_ok = bool(required_parent and required_parent == source_model_sha)
    if declared_parent and declared_parent != source_model_sha:
        parent_lock_ok = False

    source_eval = evaluate_lineage_runtime(source_runtime, cycle=cycle)
    candidate_eval = evaluate_lineage_runtime(candidate_runtime, cycle=cycle)
    retention_ok, retention_reason = _research_eligible(
        source_eval,
        candidate_eval,
        minimum_loss_gain=0.0,
        max_domain_regression=0.05,
    )
    source_language = evaluate_phase5_language(source_runtime)
    candidate_language = evaluate_phase5_language(candidate_runtime)
    degeneration_ok, degeneration_reason = degeneration_gate(
        source_language,
        candidate_language,
        max_repetition_regression=0.04,
        max_entropy_collapse_fraction=0.70,
    )
    accepted = bool(parent_lock_ok and retention_ok and degeneration_ok)

    lineage_state_path = root / "lineage.json"
    previous_lineage = _read_json(lineage_state_path, {})
    previous_lineage = (
        previous_lineage if isinstance(previous_lineage, dict) else {}
    )
    lineage_id = str(
        previous_lineage.get("lineage_id")
        or (source.get("progress") or {}).get("lineage_id")
        or f"airi-{source_model_sha[:16]}"
    )
    target_architecture = ArchitectureSpec.from_genome(
        target_genome,
        target_vocab_size=candidate_runtime.tokenizer.vocab_size,
    )
    report: dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "version": LINEAGE_VERSION,
        "mode": "verified_trained_descendant",
        "accepted": accepted,
        "lineage_id": lineage_id,
        "cycle": int(cycle),
        "candidate_id": str(candidate_id),
        "research_kind": research_kind,
        "source_checkpoint": str(source["checkpoint_rel"]),
        "source_model_sha256": source_model_sha,
        "declared_parent_model_sha256": declared_parent or None,
        "required_parent_model_sha256": required_parent or None,
        "parent_lock_passed": bool(parent_lock_ok),
        "tokens_processed_before": int(source.get("tokens_processed", 0) or 0),
        "target_tokens": int(source.get("target_tokens", 0) or 0),
        "source_parameters": int(source["parameters"]),
        "target_parameters": int(parameter_count(candidate_runtime.model)),
        "source_genome": source_genome.to_dict(),
        "target_genome": target_genome.to_dict(),
        "target_architecture": architecture_manifest(target_architecture),
        "retention_gate_passed": bool(retention_ok),
        "retention_gate_reason": retention_reason,
        "degeneration_gate_passed": bool(degeneration_ok),
        "degeneration_gate_reason": degeneration_reason,
        "source_validation": source_eval,
        "candidate_validation": candidate_eval,
        "source_language": source_language,
        "candidate_language": candidate_language,
        "candidate_weights_preserved": True,
        "research_checkpoint_becomes_independent_model": False,
        "rollback_available": False,
    }

    evidence_root = root / "lineage-research"
    evidence_root.mkdir(parents=True, exist_ok=True)
    if not accepted:
        _atomic_json(evidence_root / "last-adoption.json", report)
        return report

    lineage_genome = _lineage_genome(
        source_genome,
        target_genome,
        cycle,
        target_vocab_size=candidate_runtime.tokenizer.vocab_size,
    )
    lineage_architecture = ArchitectureSpec.from_genome(
        lineage_genome,
        target_vocab_size=candidate_runtime.tokenizer.vocab_size,
    )

    staging = root / ".lineage-descendant-next"
    shutil.rmtree(staging, ignore_errors=True)
    candidate_runtime.save_checkpoint(
        staging,
        metadata={
            "role": "active_airi_lineage",
            "lineage_id": lineage_id,
            "lineage_version": LINEAGE_VERSION,
            "parent_model_sha256": source_model_sha,
            "descendant_candidate_id": str(candidate_id),
            "descendant_research_kind": research_kind,
            "tokens_processed_inherited": int(
                source.get("tokens_processed", 0) or 0
            ),
            "external_pretrained": False,
            "research_checkpoint_becomes_independent_model": False,
        },
    )
    adopted_model_sha = checkpoint_model_sha256(staging)

    rollback_root = root / "lineage-rollback"
    shutil.rmtree(rollback_root, ignore_errors=True)
    rollback_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_checkpoint, rollback_root / "checkpoint")
    for extra in (
        root / "bootstrap-data" / "progress.json",
        root / "status.json",
        root / "lineage.json",
    ):
        if extra.is_file():
            shutil.copy2(extra, rollback_root / extra.name)
    report["rollback_available"] = True
    report["rollback_checkpoint"] = "lineage-rollback/checkpoint"

    if bool(source["bootstrap_active"]):
        active_path = source_checkpoint
        old_path = root / ".lineage-descendant-previous"
        shutil.rmtree(old_path, ignore_errors=True)
        try:
            active_path.rename(old_path)
            staging.rename(active_path)
        except Exception:
            if active_path.exists():
                shutil.rmtree(active_path, ignore_errors=True)
            if old_path.exists():
                old_path.rename(active_path)
            raise
        shutil.rmtree(old_path, ignore_errors=True)

        progress_path = root / "bootstrap-data" / "progress.json"
        progress = _read_json(progress_path, {})
        inherited_tokens = int(progress.get("tokens_processed", 0) or 0)
        inherited_steps = int(progress.get("steps", 0) or 0)
        progress["lineage_id"] = lineage_id
        progress["capacity_genome"] = lineage_genome.to_dict()
        history = list(progress.get("lineage_descendant_adoptions") or [])
        history.append({
            "cycle": int(cycle),
            "candidate_id": str(candidate_id),
            "research_kind": research_kind,
            "source_model_sha256": source_model_sha,
            "adopted_model_sha256": adopted_model_sha,
            "tokens_processed": inherited_tokens,
            "steps": inherited_steps,
        })
        progress["lineage_descendant_adoptions"] = history[-64:]
        progress["tokens_processed"] = inherited_tokens
        progress["steps"] = inherited_steps
        progress["optimizer_reset_after_lineage_adoption"] = True
        _atomic_json(progress_path, progress)
        (root / "bootstrap-data" / "optimizer.pt").unlink(missing_ok=True)
        shutil.rmtree(root / "bootstrap-data" / "best", ignore_errors=True)
        shutil.rmtree(root / "bootstrap-data" / "pre-sft", ignore_errors=True)
        shutil.rmtree(
            root / "bootstrap-data" / "pre-anticollapse",
            ignore_errors=True,
        )
        active_rel = "bootstrap-data/candidate"
    else:
        shutil.rmtree(staging, ignore_errors=True)
        _save_champion(
            root,
            lineage_genome,
            candidate_runtime,
            candidate_eval,
        )
        active_rel = "champion"

    migrations = list(previous_lineage.get("migrations") or [])
    migrations.append({
        "type": "trained_descendant_adoption",
        "cycle": int(cycle),
        "candidate_id": str(candidate_id),
        "research_kind": research_kind,
        "source_model_sha256": source_model_sha,
        "adopted_model_sha256": adopted_model_sha,
        "tokens_processed": int(source.get("tokens_processed", 0) or 0),
    })
    lineage_payload = {
        "schema": LINEAGE_SCHEMA,
        "version": LINEAGE_VERSION,
        "lineage_id": lineage_id,
        "single_active_model": True,
        "active_checkpoint": active_rel,
        "active_model_sha256": adopted_model_sha,
        "active_genome": lineage_genome.to_dict(),
        "active_architecture": architecture_manifest(lineage_architecture),
        "tokens_processed": int(source.get("tokens_processed", 0) or 0),
        "target_tokens": int(source.get("target_tokens", 0) or 0),
        "parameters": int(parameter_count(candidate_runtime.model)),
        "tokenizer_version": str(candidate_runtime.tokenizer.version),
        "tokenizer_vocab_size": int(candidate_runtime.tokenizer.vocab_size),
        "bootstrap_active": bool(source["bootstrap_active"]),
        "rollback_snapshots_are_not_competing_models": True,
        "research_challengers_are_temporary": True,
        "migrations": migrations[-64:],
        "updated_at": time.time(),
    }
    _atomic_json(lineage_state_path, lineage_payload)

    status = _read_json(root / "status.json", {})
    status = dict(status) if isinstance(status, dict) else {}
    status["lineage"] = lineage_payload
    status["lineage_adoption"] = {
        "accepted": True,
        "candidate_id": str(candidate_id),
        "research_kind": research_kind,
        "source_model_sha256": source_model_sha,
        "adopted_model_sha256": adopted_model_sha,
        "tokens_preserved": int(source.get("tokens_processed", 0) or 0),
    }
    _atomic_json(root / "status.json", status)

    report.update({
        "accepted": True,
        "active_checkpoint": active_rel,
        "adopted_model_sha256": adopted_model_sha,
        "lineage_genome": lineage_genome.to_dict(),
        "tokens_processed_after": int(source.get("tokens_processed", 0) or 0),
        "token_progress_preserved": True,
    })
    _atomic_json(evidence_root / "last-adoption.json", report)
    with (evidence_root / "adoptions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n"
        )

    # Any laboratory model weights are no longer needed after the verdict.
    shutil.rmtree(root / "latest-research", ignore_errors=True)
    shutil.rmtree(root / "specialists", ignore_errors=True)
    return report


def migrate_live_lineage(
    state_dir: str | Path,
    *,
    target_checkpoint: str | Path,
    target_genome: dict[str, Any] | GeneralistGenome,
    cycle: int,
    candidate_id: str,
    research_kind: str | None = None,
    minimum_transfer_fraction: float = 0.35,
) -> dict[str, Any]:
    """Migrate the one live AIRI checkpoint onto a verified architecture.

    The research checkpoint supplies only the tested target topology and values
    for tensors that have no safe mapping. Compatible learned tensors are then
    overlaid from the live checkpoint. The migration is accepted only after
    held-out anti-forgetting and Phase-5 degeneration gates. Token/step progress
    is never reset; optimizer moments are reset because their tensor layout may
    no longer match the architecture.
    """
    root = Path(state_dir).expanduser().resolve()
    source = active_lineage_snapshot(root)
    source_runtime: GeneralistRuntime = source["runtime"]
    source_genome: GeneralistGenome = source["genome"]
    target_genome_obj = (
        target_genome
        if isinstance(target_genome, GeneralistGenome)
        else GeneralistGenome(**dict(target_genome)).validate()
    )

    research_path = Path(target_checkpoint).expanduser().resolve()
    research_runtime = GeneralistRuntime.from_checkpoint(research_path, device="cpu")
    expected_target = target_genome_obj.model_config(
        research_runtime.tokenizer.vocab_size
    ).to_dict()
    if research_runtime.config.to_dict() != expected_target:
        raise RuntimeError("research checkpoint does not match target architecture genome")

    source_model_sha = checkpoint_model_sha256(Path(source["checkpoint"]))
    lineage_state_path = root / "lineage.json"
    lineage_state = _read_json(lineage_state_path, {})
    lineage_id = str(
        (lineage_state or {}).get("lineage_id")
        or (source.get("progress") or {}).get("lineage_id")
        or f"airi-{source_model_sha[:16]}"
    )

    # Seed architecture-specific tensors from the verified laboratory winner,
    # then overwrite every safely transferable learned tensor with the newest
    # live-lineage knowledge.
    target_model = copy.deepcopy(research_runtime.model)
    transfer = _transfer_compatible_weights(
        source_runtime.model,
        target_model,
        source_tokenizer=source_runtime.tokenizer,
        target_tokenizer=research_runtime.tokenizer,
    )
    migrated = GeneralistRuntime(
        target_model,
        research_runtime.config,
        tokenizer=research_runtime.tokenizer,
        device="cpu",
    )

    source_eval = evaluate_lineage_runtime(source_runtime, cycle=cycle)
    migrated_eval = evaluate_lineage_runtime(migrated, cycle=cycle)
    retention_ok, retention_reason = _research_eligible(
        source_eval,
        migrated_eval,
        minimum_loss_gain=-0.015,
        max_domain_regression=0.03,
    )
    source_language = evaluate_phase5_language(source_runtime)
    migrated_language = evaluate_phase5_language(migrated)
    degeneration_ok, degeneration_reason = degeneration_gate(
        source_language,
        migrated_language,
        max_repetition_regression=0.04,
        max_entropy_collapse_fraction=0.70,
    )
    transfer_ok = float(transfer.get("parameter_fraction", 0.0) or 0.0) >= float(
        minimum_transfer_fraction
    )
    accepted = bool(retention_ok and degeneration_ok and transfer_ok)

    architecture = ArchitectureSpec.from_genome(
        target_genome_obj,
        target_vocab_size=migrated.tokenizer.vocab_size,
    )
    report: dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "version": LINEAGE_VERSION,
        "accepted": accepted,
        "lineage_id": lineage_id,
        "cycle": int(cycle),
        "candidate_id": str(candidate_id),
        "research_kind": research_kind,
        "source_checkpoint": str(source["checkpoint_rel"]),
        "source_model_sha256": source_model_sha,
        "tokens_processed_before": int(source.get("tokens_processed", 0) or 0),
        "target_tokens": int(source.get("target_tokens", 0) or 0),
        "source_parameters": int(source["parameters"]),
        "target_parameters": int(parameter_count(migrated.model)),
        "source_genome": source_genome.to_dict(),
        "target_genome": target_genome_obj.to_dict(),
        "target_architecture": architecture_manifest(architecture),
        "weight_transfer": transfer,
        "retention_gate_passed": bool(retention_ok),
        "retention_gate_reason": retention_reason,
        "degeneration_gate_passed": bool(degeneration_ok),
        "degeneration_gate_reason": degeneration_reason,
        "transfer_gate_passed": bool(transfer_ok),
        "minimum_transfer_fraction": float(minimum_transfer_fraction),
        "source_validation": source_eval,
        "migrated_validation": migrated_eval,
        "source_language": source_language,
        "migrated_language": migrated_language,
        "rollback_available": False,
        "optimizer_reset": False,
        "research_checkpoint_becomes_independent_model": False,
    }

    architecture_root = root / "architecture-research"
    architecture_root.mkdir(parents=True, exist_ok=True)
    if not accepted:
        _atomic_json(architecture_root / "lineage-migration.json", report)
        return report

    lineage_genome = _lineage_genome(
        source_genome,
        target_genome_obj,
        cycle,
        target_vocab_size=migrated.tokenizer.vocab_size,
    )
    lineage_architecture = ArchitectureSpec.from_genome(
        lineage_genome,
        target_vocab_size=migrated.tokenizer.vocab_size,
    )
    staging = root / ".lineage-migration-candidate"
    shutil.rmtree(staging, ignore_errors=True)
    migrated.save_checkpoint(
        staging,
        metadata={
            "role": "active_airi_lineage",
            "lineage_id": lineage_id,
            "lineage_version": LINEAGE_VERSION,
            "parent_model_sha256": source_model_sha,
            "architecture_candidate_id": str(candidate_id),
            "architecture_research_kind": research_kind,
            "tokens_processed_inherited": int(source.get("tokens_processed", 0) or 0),
            "external_pretrained": False,
            "research_checkpoint_becomes_independent_model": False,
        },
    )
    migrated_model_sha = checkpoint_model_sha256(staging)

    safe_id = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(candidate_id)
    )[:80]
    # Keep exactly one full previous checkpoint in the working tree. Git commit
    # history preserves older snapshots, while duplicating every large model in
    # the current state branch would grow storage without bound.
    rollback_root = root / "lineage-rollback"
    shutil.rmtree(rollback_root, ignore_errors=True)
    rollback_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(source["checkpoint"]), rollback_root / "checkpoint")
    for extra in (
        root / "bootstrap-data" / "progress.json",
        root / "status.json",
        root / "lineage.json",
    ):
        if extra.is_file():
            shutil.copy2(extra, rollback_root / extra.name)

    history_root = root / "lineage-history"
    history_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        history_root / f"cycle-{max(1, int(cycle)):06d}-{safe_id or 'architecture'}.json",
        {
            "cycle": int(cycle),
            "candidate_id": str(candidate_id),
            "source_model_sha256": source_model_sha,
            "tokens_processed": int(source.get("tokens_processed", 0) or 0),
            "source_checkpoint": str(source["checkpoint_rel"]),
            "rollback_checkpoint": "lineage-rollback/checkpoint",
        },
    )
    report["rollback_available"] = True
    report["rollback_checkpoint"] = "lineage-rollback/checkpoint"

    if bool(source["bootstrap_active"]):
        active_path = Path(source["checkpoint"])
        old_path = root / ".lineage-migration-previous"
        shutil.rmtree(old_path, ignore_errors=True)
        try:
            active_path.rename(old_path)
            staging.rename(active_path)
        except Exception:
            if active_path.exists():
                shutil.rmtree(active_path, ignore_errors=True)
            if old_path.exists():
                old_path.rename(active_path)
            raise
        shutil.rmtree(old_path, ignore_errors=True)

        progress_path = root / "bootstrap-data" / "progress.json"
        progress = _read_json(progress_path, {})
        inherited_tokens = int(progress.get("tokens_processed", 0) or 0)
        inherited_steps = int(progress.get("steps", 0) or 0)
        progress["lineage_id"] = lineage_id
        progress["capacity_genome"] = lineage_genome.to_dict()
        progress["architecture_fingerprint"] = architecture.fingerprint()
        history = list(progress.get("architecture_migrations") or [])
        history.append({
            "cycle": int(cycle),
            "candidate_id": str(candidate_id),
            "source_model_sha256": source_model_sha,
            "migrated_model_sha256": migrated_model_sha,
            "tokens_processed": inherited_tokens,
            "steps": inherited_steps,
            "architecture_fingerprint": architecture.fingerprint(),
            "weight_transfer_fraction": float(
                transfer.get("parameter_fraction", 0.0) or 0.0
            ),
        })
        progress["architecture_migrations"] = history[-64:]
        progress["optimizer_reset_after_architecture_migration"] = True
        # Explicit invariants: architecture changes never purchase progress by
        # resetting counters or marking unfinished rungs complete.
        progress["tokens_processed"] = inherited_tokens
        progress["steps"] = inherited_steps
        _atomic_json(progress_path, progress)
        (root / "bootstrap-data" / "optimizer.pt").unlink(missing_ok=True)
        shutil.rmtree(root / "bootstrap-data" / "best", ignore_errors=True)
        shutil.rmtree(root / "bootstrap-data" / "pre-sft", ignore_errors=True)
        shutil.rmtree(root / "bootstrap-data" / "pre-anticollapse", ignore_errors=True)
        report["optimizer_reset"] = True
        active_rel = "bootstrap-data/candidate"
    else:
        shutil.rmtree(staging, ignore_errors=True)
        _save_champion(
            root,
            lineage_genome,
            migrated,
            migrated_eval,
        )
        active_rel = "champion"

    status = _read_json(root / "status.json", {})
    status = dict(status) if isinstance(status, dict) else {}
    lineage_history = list((lineage_state or {}).get("migrations") or [])
    lineage_history.append({
        "cycle": int(cycle),
        "candidate_id": str(candidate_id),
        "source_model_sha256": source_model_sha,
        "migrated_model_sha256": migrated_model_sha,
        "architecture_fingerprint": architecture.fingerprint(),
        "tokens_processed": int(source.get("tokens_processed", 0) or 0),
    })
    lineage_payload = {
        "schema": LINEAGE_SCHEMA,
        "version": LINEAGE_VERSION,
        "lineage_id": lineage_id,
        "single_active_model": True,
        "active_checkpoint": active_rel,
        "active_model_sha256": migrated_model_sha,
        "active_genome": lineage_genome.to_dict(),
        "active_architecture": architecture_manifest(lineage_architecture),
        "tokens_processed": int(source.get("tokens_processed", 0) or 0),
        "target_tokens": int(source.get("target_tokens", 0) or 0),
        "optimizer_state_preserved": not bool(source["bootstrap_active"]),
        "rollback_snapshots_are_not_competing_models": True,
        "research_challengers_are_temporary": True,
        "migrations": lineage_history[-64:],
        "updated_at": time.time(),
    }
    _atomic_json(lineage_state_path, lineage_payload)
    status["lineage"] = lineage_payload
    status["architecture_migration"] = {
        "accepted": True,
        "candidate_id": str(candidate_id),
        "architecture_fingerprint": architecture.fingerprint(),
        "source_model_sha256": source_model_sha,
        "migrated_model_sha256": migrated_model_sha,
        "tokens_preserved": int(source.get("tokens_processed", 0) or 0),
    }
    _atomic_json(root / "status.json", status)

    report.update({
        "accepted": True,
        "active_checkpoint": active_rel,
        "migrated_model_sha256": migrated_model_sha,
        "lineage_genome": lineage_genome.to_dict(),
        "tokens_processed_after": int(source.get("tokens_processed", 0) or 0),
        "token_progress_preserved": True,
    })
    _atomic_json(architecture_root / "lineage-migration.json", report)
    with (architecture_root / "lineage-migrations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    return report
