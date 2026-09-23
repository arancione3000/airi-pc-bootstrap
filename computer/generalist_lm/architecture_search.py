from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from dataclasses import replace

from .architecture_ir import ArchitectureSpec, architecture_id, architecture_manifest
from .architecture_mutations import proposal_set
from .architecture_verifier import verify_architecture
from .evolution import GeneralistGenome
from .generalist_swarm import (
    finalize_swarm,
    prepare_swarm,
    run_candidate,
    select_survivors,
)
from .mathesis_bridge import mathesis_architecture_hypotheses
from .meta_controller import decide_next_action
from .model import estimate_parameter_count
from .runtime import checkpoint_has_model, checkpoint_model_sha256
from .lineage_migration import (
    active_lineage_snapshot,
    adaptive_architecture_parameter_cap,
    adopt_verified_descendant,
    evaluate_lineage_runtime,
)


ARCHITECTURE_SEARCH_VERSION = "airi-architecture-search-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _read(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _research_quality_key(row: dict[str, Any]) -> tuple:
    pathological = bool(
        row.get("any_generation_pathological_repetition")
        if "any_generation_pathological_repetition" in row
        else row.get("pathological_repetition")
    )
    return (
        0 if bool(row.get("all_seed_eligible")) else 1,
        pathological,
        -float(row.get("mean_generation_similarity", 0.0) or 0.0),
        float(row.get("mean_generation_repetition_rate", 1.0) or 1.0),
        float(row.get("mean_nll_per_byte", float("inf"))),
        int(row.get("parameters", 1 << 60) or (1 << 60)),
    )


def _load_incumbent_candidate(
    state_dir: Path,
    *,
    parent_fingerprint: str,
    current_vocab: int,
    parameter_cap: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    root = state_dir / "architecture-research" / "incumbent"
    summary = _read(root / "summary.json", {})
    checkpoint = root / "checkpoint"
    report = {
        "available": False,
        "used": False,
        "reason": "missing",
    }
    if not isinstance(summary, dict) or not summary:
        return None, report
    if str(summary.get("parent_fingerprint") or "") != str(parent_fingerprint):
        report.update({
            "available": True,
            "reason": "parent_changed",
            "candidate_id": summary.get("candidate_id"),
        })
        return None, report
    if bool(summary.get("production_qualified")):
        report.update({"available": True, "reason": "already_production_qualified"})
        return None, report
    if bool(summary.get("external_pretrained")):
        report.update({"available": True, "reason": "external_pretrained_forbidden"})
        return None, report
    required = [
        checkpoint / "config.json",
        checkpoint / "tokenizer.json",
        checkpoint / "metadata.json",
    ]
    if (
        not checkpoint_has_model(checkpoint)
        or not all(path.is_file() and path.stat().st_size > 0 for path in required)
    ):
        report.update({"available": True, "reason": "checkpoint_incomplete"})
        return None, report

    genome_raw = summary.get("genome")
    if not isinstance(genome_raw, dict):
        report.update({"available": True, "reason": "missing_genome"})
        return None, report
    try:
        genome = GeneralistGenome(**dict(genome_raw)).validate()
        architecture_raw = summary.get("architecture") or {}
        vocab_size = int(
            architecture_raw.get("target_vocab_size")
            or summary.get("tokenizer_vocab_size")
            or current_vocab
        )
        spec = ArchitectureSpec.from_genome(
            genome,
            target_vocab_size=vocab_size,
        )
    except Exception as exc:
        report.update({
            "available": True,
            "reason": f"invalid_incumbent:{type(exc).__name__}:{exc}",
        })
        return None, report

    fingerprint = spec.fingerprint()
    expected_fingerprint = str(summary.get("fingerprint") or "")
    if expected_fingerprint and expected_fingerprint != fingerprint:
        report.update({"available": True, "reason": "fingerprint_mismatch"})
        return None, report
    parameters = spec.parameter_estimate(vocab_size=vocab_size)
    if parameters > int(parameter_cap):
        report.update({"available": True, "reason": "parameter_cap_exceeded"})
        return None, report

    candidate = {
        "kind": "architecture_incumbent",
        "genome": genome.to_dict(),
        "candidate_id": genome.genome_id,
        "estimated_parameters": int(parameters),
        "architecture": spec.to_dict(),
        "architecture_fingerprint": fingerprint,
        "hypothesis": (
            "continue training the best research-only architecture checkpoint "
            "from a previous cycle instead of rediscovering it from scratch"
        ),
        "initial_checkpoint": "architecture-research/incumbent/checkpoint",
        "incumbent_cycle": int(summary.get("cycle", 0) or 0),
        "incumbent_cumulative_steps": int(
            summary.get("cumulative_steps", 0) or 0
        ),
        "research_only": True,
    }
    report.update({
        "available": True,
        "used": True,
        "reason": "same_parent_research_incumbent",
        "candidate_id": genome.genome_id,
        "parameters": int(parameters),
        "fingerprint": fingerprint,
    })
    return candidate, report


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase5_signals(state_dir: Path) -> list[str]:
    status = _read(state_dir / "status.json", {})
    phase5 = status.get("phase5_bootstrap") or {}
    after = phase5.get("after") or {}
    out: list[str] = []
    if after.get("pathological_repetition"):
        out.extend(["language_collapse", "autoregressive_collapse"])
    if float(after.get("repetition_rate", 0.0) or 0.0) >= 0.65:
        out.append("language_collapse")
    if phase5 and not bool(phase5.get("minimum_success")):
        out.append("language_gap")
    return list(dict.fromkeys(out))


def _negative_architecture_memory(
    state_dir: Path,
    *,
    parent_fingerprint: str,
) -> tuple[set[str], dict[str, Any]]:
    """Remember failed topologies only for the same parent architecture.

    An exact topology can become useful after the champion changes, so failures
    are not globally blacklisted forever. Under the same parent, however, the
    autonomous loop must not continuously rediscover the same failed brain.
    """
    root = state_dir / "architecture-research"
    fingerprints: set[str] = set()
    sources: list[str] = []

    last_plan = _read(root / "last-plan.json", {})
    previous_parent = (
        (last_plan.get("architecture_parent") or {}).get("fingerprint")
        if isinstance(last_plan, dict)
        else None
    )
    if previous_parent == parent_fingerprint:
        leaderboard = _read(root / "leaderboard.json", {})
        for row in leaderboard.get("entries") or []:
            if bool(row.get("all_seed_eligible")):
                continue
            fingerprint = str(row.get("fingerprint") or "").strip()
            if fingerprint:
                fingerprints.add(fingerprint)
        for row in last_plan.get("static_rejections") or []:
            fingerprint = str(row.get("fingerprint") or "").strip()
            if fingerprint:
                fingerprints.add(fingerprint)
        if fingerprints:
            sources.append("previous_same_parent_cycle")

    rejected_root = root / "rejected"
    if rejected_root.is_dir():
        for path in sorted(rejected_root.glob("*.json")):
            row = _read(path, {})
            if str(row.get("parent_fingerprint") or "") != parent_fingerprint:
                continue
            fingerprint = str(row.get("fingerprint") or "").strip()
            if fingerprint:
                fingerprints.add(fingerprint)
                sources.append(path.name)

    return fingerprints, {
        "parent_fingerprint": parent_fingerprint,
        "rejected_fingerprints": sorted(fingerprints),
        "count": len(fingerprints),
        "sources": sorted(set(sources)),
        "scope": "same-parent topology failures only",
    }


def prioritize_architecture_proposals(
    proposals: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reserve scarce population slots for capacity experiments.

    The meta-controller may decide that the current champion is under-capacity.
    In that situation Architecture Search must not fill every slot with cheap
    same-size mutations and silently omit the requested scale experiment.
    Sorting is stable within each proposal class so the existing weakness-
    directed structural order is preserved.
    """
    priority = {
        "capacity_plus_structure": 0,
        "capacity_scale": 1,
        "structural_mutation": 2,
    }
    return sorted(
        [dict(row) for row in proposals],
        key=lambda row: (
            priority.get(str(row.get("kind") or ""), 9),
            int(row.get("parameter_estimate", 1 << 60) or (1 << 60)),
        ),
    )


def _proposal_architecture(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("architecture")
    return dict(raw) if isinstance(raw, dict) else {}


def _is_gqa_proposal(row: dict[str, Any]) -> bool:
    return str(_proposal_architecture(row).get("attention_type") or "") == "gqa"


def _is_local_attention_proposal(row: dict[str, Any]) -> bool:
    return int(
        _proposal_architecture(row).get("local_attention_window") or 0
    ) > 0


def select_diverse_architecture_proposals(
    proposals: Iterable[dict[str, Any]],
    *,
    slots: int,
    attention_slots: int = 2,
) -> list[dict[str, Any]]:
    """Fill a bounded population without letting scale crowd out topology.

    Capacity probes remain important, but when verified GQA/local-attention
    proposals exist we reserve up to two slots for them.  This makes
    Architecture Search compare "bigger" against "different", rather than
    accidentally reducing meta-evolution to parameter scaling.
    """
    available = [dict(row) for row in proposals]
    limit = max(0, int(slots))
    if limit == 0:
        return []

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(predicate) -> None:
        for row in available:
            fingerprint = str(row.get("fingerprint") or "")
            if not fingerprint or fingerprint in seen:
                continue
            if predicate(row):
                selected.append(row)
                seen.add(fingerprint)
                return

    structural_budget = min(max(0, int(attention_slots)), limit)
    if structural_budget >= 1:
        take(_is_gqa_proposal)
    if structural_budget >= 2 and len(selected) < limit:
        take(_is_local_attention_proposal)

    for row in prioritize_architecture_proposals(available):
        if len(selected) >= limit:
            break
        fingerprint = str(row.get("fingerprint") or "")
        if not fingerprint or fingerprint in seen:
            continue
        selected.append(row)
        seen.add(fingerprint)
    return selected


def link_mathesis_hypotheses(
    proposal: dict[str, Any],
    hypotheses: Iterable[dict[str, Any]],
) -> list[str]:
    """Return MATHESIS hypothesis ids that materially match a proposal."""
    arch = _proposal_architecture(proposal)
    out: list[str] = []
    for hypothesis in hypotheses:
        mutation = hypothesis.get("mutation")
        if not isinstance(mutation, dict):
            continue
        matched = False
        if mutation.get("attention_type") == "gqa":
            matched = str(arch.get("attention_type") or "") == "gqa"
        elif "local_attention_window_policy" in mutation:
            matched = int(arch.get("local_attention_window") or 0) > 0
        elif mutation.get("norm_type") == "rmsnorm":
            matched = (
                str(arch.get("norm_type") or "") == "rmsnorm"
                and str(arch.get("position_encoding") or "") == "rope"
            )
        if matched:
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
            if hypothesis_id:
                out.append(hypothesis_id)
    return list(dict.fromkeys(out))


def prepare_architecture_search(
    state_dir: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path,
    mathesis_state_dir: str | Path | None = None,
    population_size: int = 8,
    parameter_cap: int = 7_000_000,
    max_context: int = 512,
    max_width: int = 384,
    max_layers: int = 10,
    force_search: bool = False,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    requested_parameter_cap = int(parameter_cap)
    parameter_cap = adaptive_architecture_parameter_cap(
        root,
        requested_parameter_cap,
    )
    live = active_lineage_snapshot(root)
    live_config = live["runtime"].config
    # The laboratory must never become structurally smaller than AIRI itself.
    # Keep bounded headroom so it can test the next architecture rather than
    # merely re-validating today's dimensions forever.
    max_context = min(
        8192,
        max(int(max_context), int(live_config.context_length) * 2),
    )
    max_width = min(
        4096,
        max(int(max_width), int(live_config.d_model) + 128),
    )
    max_layers = min(
        96,
        max(int(max_layers), int(live_config.n_layers) + 2),
    )
    decision = decide_next_action(root, parameter_cap=int(parameter_cap))

    base_path = output.with_suffix(".base.json")
    base = prepare_swarm(
        root,
        base_path,
        population_size=max(8, int(population_size)),
        max_params=int(parameter_cap),
        max_context=int(max_context),
        max_width=int(max_width),
        max_layers=int(max_layers),
        curriculum_max_rows=5000,
        mathesis_state_dir=mathesis_state_dir,
        grow_data=False,
    )

    signals = list(base.get("signals") or [])
    signals.extend(_phase5_signals(root))
    signals = list(dict.fromkeys(str(row) for row in signals))
    base["signals"] = signals

    # Architecture research is always relative to the newest learned AIRI,
    # including an unfinished Phase-5 candidate. The production champion can
    # remain a rollback anchor, but it is not allowed to make the laboratory
    # compare against stale weights.
    champion = live["genome"]
    current_vocab = int(live["runtime"].tokenizer.vocab_size)
    base["champion"] = champion.to_dict()
    base["champion_report"] = evaluate_lineage_runtime(
        live["runtime"],
        cycle=int(base.get("cycle", 1) or 1),
    )
    inherited_live_source = dict(base.get("live_lineage_source") or {})
    live_model_sha = str(
        inherited_live_source.get("model_sha256")
        or checkpoint_model_sha256(Path(live["checkpoint"]))
    )
    lineage_id = str(
        inherited_live_source.get("lineage_id")
        or f"airi-{live_model_sha[:16]}"
    )
    base["live_lineage_source"] = {
        "checkpoint": str(live["checkpoint_rel"]),
        "lineage_id": lineage_id,
        "model_sha256": live_model_sha,
        "bootstrap_active": bool(live["bootstrap_active"]),
        "tokens_processed": int(live.get("tokens_processed", 0) or 0),
        "target_tokens": int(live.get("target_tokens", 0) or 0),
        "parameters": int(live["parameters"]),
    }
    scaling = dict(base.get("progressive_scaling") or {})
    scaling["current_parameters"] = int(live["parameters"])
    base["progressive_scaling"] = scaling
    parent = ArchitectureSpec.from_genome(
        champion,
        target_vocab_size=current_vocab,
    )

    mathesis_hypotheses: list[dict[str, Any]] = []
    if mathesis_state_dir and Path(mathesis_state_dir).exists():
        try:
            mathesis_hypotheses = mathesis_architecture_hypotheses(
                mathesis_state_dir,
                observed_signals=signals,
            )
        except Exception as exc:
            mathesis_hypotheses = [{
                "ok": False,
                "reason": f"{type(exc).__name__}:{exc}",
                "promotion_authority": False,
            }]

    negative_fingerprints, negative_memory = _negative_architecture_memory(
        root,
        parent_fingerprint=parent.fingerprint(),
    )
    raw_proposals = proposal_set(
        parent,
        signals=signals,
        parameter_cap=int(parameter_cap),
        max_candidates=max(8, int(population_size) * 3),
        max_width=int(max_width),
        max_layers=int(max_layers),
    )
    proposals = [
        proposal
        for proposal in raw_proposals
        if str(proposal.get("fingerprint") or "") not in negative_fingerprints
    ]

    verified_proposals: list[dict[str, Any]] = []
    rejected_static: list[dict[str, Any]] = []
    for proposal in proposals:
        spec = ArchitectureSpec.from_dict(dict(proposal["architecture"]))
        verification = verify_architecture(
            spec,
            vocab_size=current_vocab,
            batch_size=1,
            sequence_length=min(20, spec.context_length - 1),
            max_parameters=int(parameter_cap),
        )
        enriched = {
            **proposal,
            "static_verification": verification.report,
            "mathesis_support": link_mathesis_hypotheses(
                proposal,
                mathesis_hypotheses,
            ),
        }
        if verification.ok:
            verified_proposals.append(enriched)
        else:
            rejected_static.append(enriched)

    candidates: list[dict[str, Any]] = []

    # Same-topology control: gains must beat simply continuing the current
    # architecture under the exact same training budget, but the experiment
    # receives its own lineage identity/checkpoint metadata.
    control_generation = int(parent.generation) + 1
    control_spec = replace(
        parent,
        generation=control_generation,
        parent_id=parent.architecture_id,
        architecture_id=architecture_id(
            parent.architecture_id,
            control_generation,
            parent.canonical_payload(),
        ),
    ).validate()
    control = control_spec.to_genome()
    candidates.append({
        "index": 0,
        "kind": "architecture_control",
        "genome": control.to_dict(),
        "candidate_id": control.genome_id,
        "estimated_parameters": estimate_parameter_count(
            control.model_config(current_vocab)
        ),
        "architecture": control_spec.to_dict(),
        "architecture_fingerprint": control_spec.fingerprint(),
        "hypothesis": "same topology, equal training budget control",
        "initial_checkpoint": str(live["checkpoint_rel"]),
        "initial_checkpoint_mode": "live_lineage",
    })

    seen = {parent.fingerprint()}
    incumbent_candidate = None
    incumbent_report = {
        "available": False,
        "reason": "single_lineage_policy_discards_persistent_research_weights",
    }
    remaining_slots = max(
        0,
        int(population_size) - len(candidates),
    )
    ordered_proposals = select_diverse_architecture_proposals(
        verified_proposals,
        slots=remaining_slots,
        attention_slots=2,
    )
    for proposal in ordered_proposals:
        spec = ArchitectureSpec.from_dict(dict(proposal["architecture"]))
        if spec.fingerprint() in seen:
            continue
        seen.add(spec.fingerprint())
        genome = spec.to_genome()
        candidates.append({
            "index": len(candidates),
            "kind": f"architecture_{proposal['kind']}",
            "genome": genome.to_dict(),
            "candidate_id": genome.genome_id,
            "estimated_parameters": spec.parameter_estimate(
                vocab_size=current_vocab
            ),
            "architecture": spec.to_dict(),
            "architecture_fingerprint": spec.fingerprint(),
            "hypothesis": proposal["hypothesis"],
            "mathesis_support": proposal.get("mathesis_support") or [],
            "falsification": proposal["falsification"],
            "initial_checkpoint": str(live["checkpoint_rel"]),
            "initial_checkpoint_mode": "live_lineage",
        })
        if len(candidates) >= max(2, int(population_size)):
            break

    for candidate in candidates:
        candidate["initial_checkpoint"] = str(live["checkpoint_rel"])
        candidate["initial_checkpoint_mode"] = "live_lineage"
        candidate["lineage_id"] = lineage_id
        candidate["lineage_parent_model_sha256"] = live_model_sha

    run_search = (
        (bool(force_search) or decision.get("action") == "architecture_search")
        and len(candidates) >= 2
    )

    base.update({
        "version": ARCHITECTURE_SEARCH_VERSION,
        "run_search": bool(run_search),
        "meta_controller": decision,
        "architecture_parent": architecture_manifest(parent),
        "architecture_proposals": verified_proposals,
        "architecture_static_rejections": rejected_static,
        "architecture_negative_memory": negative_memory,
        "architecture_incumbent": incumbent_report,
        "architecture_proposals_skipped_by_memory": [
            proposal
            for proposal in raw_proposals
            if str(proposal.get("fingerprint") or "") in negative_fingerprints
        ],
        "mathesis_architecture_hypotheses": mathesis_hypotheses,
        "candidates": candidates,
        "matrix": {
            "include": [
                {"index": int(row["index"])}
                for row in candidates
            ]
        },
        "limits": {
            **dict(base.get("limits") or {}),
            "max_params": int(parameter_cap),
            "max_context": int(max_context),
            "max_width": int(max_width),
            "max_layers": int(max_layers),
        },
        "architecture_policy": {
            "single_active_lineage": True,
            "research_challengers_are_temporary": True,
            "direct_research_checkpoint_promotion": False,
            "live_lineage_weight_seed": True,
            "requested_parameter_cap": int(requested_parameter_cap),
            "effective_parameter_cap": int(parameter_cap),
            "forced_search": bool(force_search),
            "same_budget_control_required": True,
            "capacity_slots_reserved": True,
            "attention_structure_slots_reserved": 2,
            "mathesis_hypotheses_linked_to_candidates": True,
            "same_parent_negative_memory": True,
            "persistent_research_incumbent": True,
            "static_verifier_required": True,
            "external_pretrained_weights": False,
            "arbitrary_generated_python": False,
            "mathesis_can_propose": True,
            "mathesis_can_promote": False,
            "promotion_uses_existing_generalist_gates": True,
            "parameter_cap": int(parameter_cap),
        },
    })
    _atomic_json(output, base)
    try:
        base_path.unlink()
    except FileNotFoundError:
        pass
    return base


def run_architecture_candidate(
    plan_path: str | Path,
    state_dir: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    candidate_index: int,
    stage: int,
    steps: int,
    pretrain_steps: int,
    repeat_seeds: int = 1,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    plan = _read(Path(plan_path), {})
    if not plan.get("run_search"):
        return {"ok": False, "reason": "meta_controller_did_not_request_search"}
    return run_candidate(
        plan_path,
        state_dir,
        repo_root,
        output_dir,
        candidate_index=int(candidate_index),
        stage=int(stage),
        steps=int(steps),
        repeat_seeds=int(repeat_seeds),
        pretrain_steps=int(pretrain_steps),
        max_repo_bytes=20_000_000,
        max_external_bytes=50_000_000,
        source_checkpoint=source_checkpoint,
    )


def _result_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        candidates = (
            [path]
            if path.is_file()
            else list(path.rglob("result.json"))
        )
        for item in candidates:
            if item in seen or not item.is_file():
                continue
            seen.add(item)
            try:
                row = json.loads(item.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(row, dict):
                row = dict(row)
                row["_result_path"] = str(item)
                rows.append(row)
    return rows


def finalize_architecture_search(
    state_dir: str | Path,
    plan_path: str | Path,
    result_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    plan = _read(Path(plan_path), {})
    result_paths = list(result_paths)

    base_result = finalize_swarm(
        root,
        plan_path,
        result_paths,
        output_path,
        allow_direct_promotion=False,
        persist_research_checkpoints=False,
    )

    rows = [row for row in _result_rows(result_paths) if row.get("ok")]
    rank = sorted(rows, key=_research_quality_key)

    migration_result: dict[str, Any] = {
        "accepted": False,
        "reason": "no externally approved architecture candidate",
    }
    approved = base_result.get("promotion_candidate")
    if isinstance(approved, dict) and approved.get("candidate_id"):
        approved_row = next(
            (
                row for row in rows
                if str(row.get("candidate_id") or "")
                == str(approved.get("candidate_id") or "")
            ),
            None,
        )
        if approved_row is None:
            migration_result = {
                "accepted": False,
                "reason": "approved architecture result could not be resolved",
                "candidate_id": approved.get("candidate_id"),
            }
        else:
            approved_result_path = Path(
                str(approved_row.get("_result_path") or "")
            )
            approved_checkpoint = (
                approved_result_path.parent
                / str(approved_row.get("checkpoint_dir") or "best-checkpoint")
            )
            if not approved_checkpoint.is_dir():
                migration_result = {
                    "accepted": False,
                    "reason": "approved architecture checkpoint is missing",
                    "candidate_id": approved.get("candidate_id"),
                }
            else:
                try:
                    migration_result = adopt_verified_descendant(
                        root,
                        candidate_checkpoint=approved_checkpoint,
                        candidate_genome=dict(approved_row["genome"]),
                        cycle=int(plan.get("cycle", 0) or 0),
                        candidate_id=str(approved.get("candidate_id") or ""),
                        research_kind=str(approved.get("kind") or ""),
                        expected_parent_model_sha256=str(
                            approved_row.get("lineage_parent_model_sha256")
                            or approved.get("lineage_parent_model_sha256")
                            or ""
                        ),
                    )
                except Exception as exc:
                    migration_result = {
                        "accepted": False,
                        "reason": f"lineage_migration_error:{type(exc).__name__}:{exc}",
                        "candidate_id": approved.get("candidate_id"),
                    }

    base_result["promoted"] = bool(migration_result.get("accepted"))
    base_result["promotion_reason"] = (
        "verified architecture descendant adopted into the live AIRI lineage"
        if migration_result.get("accepted")
        else str(migration_result.get("reason") or "lineage adoption rejected")
    )
    base_result["lineage_migration"] = migration_result

    architecture_root = root / "architecture-research"
    architecture_root.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    for position, row in enumerate(rank, start=1):
        genome = GeneralistGenome(**dict(row["genome"])).validate()
        spec = ArchitectureSpec.from_genome(
            genome,
            target_vocab_size=int(row.get("tokenizer_vocab_size", 384) or 384),
        )
        leaderboard.append({
            "rank": position,
            "candidate_id": row.get("candidate_id"),
            "kind": row.get("kind"),
            "architecture": spec.to_dict(),
            "fingerprint": spec.fingerprint(),
            "parameters": int(row.get("parameters", 0) or 0),
            "mean_nll_per_byte": float(row.get("mean_nll_per_byte", 0.0) or 0.0),
            "mean_generation_similarity": float(
                row.get("mean_generation_similarity", 0.0) or 0.0
            ),
            "mean_generation_repetition_rate": float(
                row.get("mean_generation_repetition_rate", 0.0) or 0.0
            ),
            "pathological_repetition": bool(
                row.get("any_generation_pathological_repetition")
            ),
            "all_seed_eligible": bool(row.get("all_seed_eligible")),
            "any_seed_eligible": bool(row.get("any_seed_eligible")),
            "score": float(row.get("score", 0.0) or 0.0),
            "stage": int(row.get("stage", 0) or 0),
            "steps": int(row.get("steps", 0) or 0),
            "pretrain_steps": int(row.get("pretrain_steps", 0) or 0),
            "cumulative_steps": int(row.get("cumulative_steps", 0) or 0),
        })

        if not row.get("all_seed_eligible"):
            safe_id = str(row.get("candidate_id") or f"candidate-{position}")
            safe_id = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in safe_id
            )[:120]
            _atomic_json(
                architecture_root / "rejected" / f"{safe_id}.json",
                {
                    **leaderboard[-1],
                    "cycle": int(plan.get("cycle", 0) or 0),
                    "parent_fingerprint": str(
                        ((plan.get("architecture_parent") or {}).get("fingerprint"))
                        or ""
                    ),
                    "rejection_scope": "same-parent topology memory",
                },
            )

    incumbent_root = architecture_root / "incumbent"
    incumbent_summary_path = incumbent_root / "summary.json"
    incumbent_decision: dict[str, Any] = {
        "action": "unchanged",
        "reason": "no_stage3_candidate",
    }
    parent_fingerprint = str(
        ((plan.get("architecture_parent") or {}).get("fingerprint"))
        or ""
    )

    # Under the single-lineage policy architecture challengers are laboratory
    # artifacts only. Persist their leaderboard/specification, never a second
    # checkpoint that can drift into an alternate AIRI.
    shutil.rmtree(incumbent_root, ignore_errors=True)
    if bool(base_result.get("promoted")):
        incumbent_decision = {
            "action": "evidence_only_after_migration",
            "reason": "live_lineage_architecture_changed",
        }
    elif rank:
        incumbent_decision = {
            "action": "evidence_only",
            "reason": "single_lineage_policy_discards_research_weights",
            "candidate_id": rank[0].get("candidate_id"),
            "parameters": rank[0].get("parameters"),
        }
    if False and rank:
        best_row = rank[0]
        best_result_path = Path(str(best_row.get("_result_path") or ""))
        best_checkpoint = (
            best_result_path.parent
            / str(best_row.get("checkpoint_dir") or "best-checkpoint")
        )
        if best_checkpoint.is_dir():
            best_genome = GeneralistGenome(
                **dict(best_row["genome"])
            ).validate()
            best_spec = ArchitectureSpec.from_genome(
                best_genome,
                target_vocab_size=int(
                    best_row.get("tokenizer_vocab_size", 384) or 384
                ),
            )
            existing_summary = _read(incumbent_summary_path, {})
            same_parent = (
                isinstance(existing_summary, dict)
                and str(existing_summary.get("parent_fingerprint") or "")
                    == parent_fingerprint
            )
            keep_existing = (
                same_parent
                and existing_summary
                and _research_quality_key(existing_summary)
                    <= _research_quality_key(best_row)
            )
            if keep_existing:
                incumbent_decision = {
                    "action": "retained",
                    "reason": "existing_same_parent_incumbent_ranked_better",
                    "candidate_id": existing_summary.get("candidate_id"),
                    "parameters": existing_summary.get("parameters"),
                }
            else:
                previous_rounds = (
                    int(existing_summary.get("incubation_rounds", 0) or 0)
                    if same_parent
                    and existing_summary.get("candidate_id")
                        == best_row.get("candidate_id")
                    else 0
                )
                checkpoint_target = incumbent_root / "checkpoint"
                shutil.rmtree(checkpoint_target, ignore_errors=True)
                incumbent_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(best_checkpoint, checkpoint_target)
                summary = {
                    "schema": 1,
                    "version": "airi-architecture-incumbent-v1",
                    "candidate_id": str(best_row.get("candidate_id") or ""),
                    "kind": best_row.get("kind"),
                    "cycle": int(plan.get("cycle", 0) or 0),
                    "stage": int(best_row.get("stage", 0) or 0),
                    "parameters": int(best_row.get("parameters", 0) or 0),
                    "mean_nll_per_byte": float(
                        best_row.get("mean_nll_per_byte", 0.0) or 0.0
                    ),
                    "mean_generation_repetition_rate": float(
                        best_row.get("mean_generation_repetition_rate", 1.0)
                        or 1.0
                    ),
                    "mean_generation_similarity": float(
                        best_row.get("mean_generation_similarity", 0.0) or 0.0
                    ),
                    "mean_generation_nonempty_rate": float(
                        best_row.get("mean_generation_nonempty_rate", 0.0)
                        or 0.0
                    ),
                    "pathological_repetition": bool(
                        best_row.get("any_generation_pathological_repetition")
                    ),
                    "worst_domain_regression": float(
                        best_row.get("worst_domain_regression", 0.0) or 0.0
                    ),
                    "all_seed_eligible": bool(
                        best_row.get("all_seed_eligible")
                    ),
                    "cumulative_steps": int(
                        best_row.get("cumulative_steps", 0) or 0
                    ),
                    "incubation_rounds": previous_rounds + 1,
                    "genome": best_genome.to_dict(),
                    "architecture": best_spec.to_dict(),
                    "fingerprint": best_spec.fingerprint(),
                    "parent_fingerprint": parent_fingerprint,
                    "model_sha256": checkpoint_model_sha256(checkpoint_target),
                    "research_only": True,
                    "production_qualified": False,
                    "recovered": False,
                    "external_pretrained": False,
                }
                _atomic_json(incumbent_summary_path, summary)
                incumbent_decision = {
                    "action": "replaced" if existing_summary else "created",
                    "reason": "best_stage3_research_checkpoint",
                    "candidate_id": summary["candidate_id"],
                    "parameters": summary["parameters"],
                    "incubation_rounds": summary["incubation_rounds"],
                }

    leaderboard_payload = {
        "schema": 1,
        "version": ARCHITECTURE_SEARCH_VERSION,
        "cycle": int(plan.get("cycle", 0) or 0),
        "meta_controller": plan.get("meta_controller"),
        "mathesis_hypotheses": plan.get("mathesis_architecture_hypotheses") or [],
        "entries": leaderboard,
        "winner_promoted": bool(base_result.get("promoted")),
        "promotion_reason": base_result.get("promotion_reason") or base_result.get("reason"),
        "lineage_migration": migration_result,
        "single_active_lineage": True,
        "existing_generalist_gates_preserved": True,
        "incumbent": incumbent_decision,
    }
    _atomic_json(architecture_root / "leaderboard.json", leaderboard_payload)
    _atomic_json(
        architecture_root / "last-plan.json",
        {
            "cycle": plan.get("cycle"),
            "signals": plan.get("signals") or [],
            "architecture_parent": plan.get("architecture_parent"),
            "architecture_proposals": plan.get("architecture_proposals") or [],
            "static_rejections": plan.get("architecture_static_rejections") or [],
            "negative_memory": plan.get("architecture_negative_memory") or {},
            "incumbent": plan.get("architecture_incumbent") or {},
            "incumbent_decision": incumbent_decision,
            "proposals_skipped_by_memory": (
                plan.get("architecture_proposals_skipped_by_memory") or []
            ),
            "mathesis_hypotheses": plan.get("mathesis_architecture_hypotheses") or [],
        },
    )

    champion_raw = _read(root / "champion-genome.json", {})
    if champion_raw:
        champion = GeneralistGenome(**champion_raw).validate()
        champion_cfg = _read(root / "champion" / "config.json", {})
        champion_vocab = int(champion_cfg.get("vocab_size", 384) or 384)
        champion_spec = ArchitectureSpec.from_genome(
            champion,
            target_vocab_size=champion_vocab,
        )
        _atomic_json(
            architecture_root / "champion-architecture.json",
            architecture_manifest(champion_spec),
        )

    lineage_raw = _read(root / "lineage.json", {})
    if isinstance(lineage_raw, dict) and isinstance(
        lineage_raw.get("active_architecture"), dict
    ):
        _atomic_json(
            architecture_root / "active-lineage-architecture.json",
            dict(lineage_raw["active_architecture"]),
        )

    history_row = {
        "cycle": int(plan.get("cycle", 0) or 0),
        "evaluated": len(leaderboard),
        "promoted": bool(base_result.get("promoted")),
        "best_candidate": leaderboard[0] if leaderboard else None,
    }
    history_path = architecture_root / "experiments.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_row, ensure_ascii=False, sort_keys=True) + "\n")

    base_result["architecture_search"] = leaderboard_payload
    base_result["architecture_incumbent"] = incumbent_decision
    _atomic_json(Path(output_path), base_result)
    return base_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIRI autonomous architecture search")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("state_dir")
    prep.add_argument("output")
    prep.add_argument("--repo-root", required=True)
    prep.add_argument("--mathesis-state")
    prep.add_argument("--population-size", type=int, default=8)
    prep.add_argument("--parameter-cap", type=int, default=7_000_000)
    prep.add_argument("--force-search", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("plan")
    worker.add_argument("state_dir")
    worker.add_argument("repo_root")
    worker.add_argument("output_dir")
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--stage", type=int, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--pretrain-steps", type=int, required=True)
    worker.add_argument("--repeat-seeds", type=int, default=1)
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
        result = prepare_architecture_search(
            args.state_dir,
            args.output,
            repo_root=args.repo_root,
            mathesis_state_dir=args.mathesis_state,
            population_size=args.population_size,
            parameter_cap=args.parameter_cap,
            force_search=args.force_search,
        )
    elif args.cmd == "worker":
        result = run_architecture_candidate(
            args.plan,
            args.state_dir,
            args.repo_root,
            args.output_dir,
            candidate_index=args.index,
            stage=args.stage,
            steps=args.steps,
            pretrain_steps=args.pretrain_steps,
            repeat_seeds=args.repeat_seeds,
            source_checkpoint=args.source_checkpoint,
        )
    elif args.cmd == "select":
        result = select_survivors(
            args.results,
            args.output,
            survivors=args.survivors,
        )
    elif args.cmd == "finalize":
        result = finalize_architecture_search(
            args.state_dir,
            args.plan,
            args.results,
            args.output,
        )
    else:
        raise AssertionError(args.cmd)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
