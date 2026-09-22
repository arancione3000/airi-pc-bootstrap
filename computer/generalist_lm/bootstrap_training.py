from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from collections import Counter
import shutil
import time
from typing import Any

from .airi_pc_lab import (
    build_airi_pc_lab_rows,
    load_verified_lab_experiences,
    snapshot_airi_pc_lab,
    run_airi_pc_lab_probe,
)
from .bootstrap_data import build_bootstrap_bundle, write_bootstrap_replay
from .curriculum import train_rows, validation_rows
from .curriculum_memory import CurriculumMemory, canary_rows
from .evolution import GeneralistGenome, progressive_scale_candidate
from .model import CausalTransformerLM, parameter_count
from .phase5_diagnostics import (
    degeneration_gate,
    evaluate_phase5_language,
    evaluate_sft_validation,
    protected_bootstrap_texts,
    sft_validation_gate,
)
from .pretraining import (
    corpus_loss,
    load_packed_block_cache,
    pack_causal_blocks,
    save_packed_block_cache,
)
from .research_cycle import (
    _continual_candidate_genome,
    _grouped_validation,
    _load_champion,
    _research_eligible,
    _research_score,
    _save_champion,
    _transfer_compatible_weights,
)
from .runtime import GeneralistRuntime
from .lineage_migration import refresh_live_lineage_manifest
from .tokenizer import PAD
from .training import causal_training_objective, train_sft


PHASE5_BOOTSTRAP_VERSION = "phase5-language-bootstrap-v3"
PHASE5_SFT_GUARD_VERSION = "phase5-sft-guard-v2"
PHASE5_ANTICOLLAPSE_VERSION = "phase5-anticollapse-v2"
PHASE5_CAPACITY_GROWTH_VERSION = "phase5-capacity-growth-v2"
PHASE5_BASE_UNIQUE_CORPUS_TOKENS = 5_000_000
PHASE5_100M_UNIQUE_CORPUS_TOKENS = 20_000_000
PHASE5_250M_UNIQUE_CORPUS_TOKENS = 40_000_000
PHASE5_500M_UNIQUE_CORPUS_TOKENS = 60_000_000
PHASE5_1B_UNIQUE_CORPUS_TOKENS = 100_000_000

# One-time human-assisted capacity gift for the current AIRI lineage only.
# Once this lineage reaches the requested tier the marker is persisted and this
# path becomes permanently dormant; future growth returns to AIRI's own search.
ASSISTED_CAPACITY_LINEAGE_ID = "airi-5d3d25177d2e83f7"
ASSISTED_CAPACITY_TARGET_PARAMETERS = 50_000_000
ASSISTED_CAPACITY_COMPLETION_FLOOR = 49_000_000


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return dict(default or {})
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_corpus_target(training_target_tokens: int) -> int:
    """Grow reviewed unique text together with cumulative optimization.

    AIRI keeps one checkpoint lineage, but larger rungs should not merely loop
    over the same small corpus forever. The streaming FineWeb/FineWeb2 sources
    already support bounded deterministic growth, so increase unique language
    gradually while keeping the corpus much smaller than the optimization
    budget for resumable CPU training.
    """
    target = max(100_000, int(training_target_tokens))
    if target >= 1_000_000_000:
        ceiling = PHASE5_1B_UNIQUE_CORPUS_TOKENS
    elif target >= 500_000_000:
        ceiling = PHASE5_500M_UNIQUE_CORPUS_TOKENS
    elif target >= 250_000_000:
        ceiling = PHASE5_250M_UNIQUE_CORPUS_TOKENS
    elif target >= 100_000_000:
        ceiling = PHASE5_100M_UNIQUE_CORPUS_TOKENS
    else:
        ceiling = PHASE5_BASE_UNIQUE_CORPUS_TOKENS
    return min(target, ceiling)

def _bootstrap_capacity_target(training_target_tokens: int) -> int | None:
    """Capacity ladder paired with cumulative language optimization.

    The 100M rung is the requested conversational/knowledge bootstrap budget.
    Larger explicit targets remain supported so the language lane does not
    become a permanent capacity ceiling after that rung.
    """
    target = max(0, int(training_target_tokens))
    if target >= 1_000_000_000:
        return 32_000_000
    if target >= 500_000_000:
        return 20_000_000
    if target >= 250_000_000:
        return 12_000_000
    if target >= 100_000_000:
        return 7_000_000
    if target >= 50_000_000:
        return 3_000_000
    if target >= 20_000_000:
        return 1_250_000
    return None


def _assisted_capacity_target(
    progress: dict[str, Any],
    *,
    current_parameters: int,
) -> int | None:
    """Return the one-time 50M assist only for Thomas's current AIRI lineage."""
    if str(progress.get("lineage_id") or "") != ASSISTED_CAPACITY_LINEAGE_ID:
        return None
    if bool(progress.get("assisted_capacity_growth_completed")):
        return None
    if int(current_parameters) >= ASSISTED_CAPACITY_COMPLETION_FLOOR:
        return None
    return ASSISTED_CAPACITY_TARGET_PARAMETERS


def _grow_bootstrap_runtime(
    base_genome: GeneralistGenome,
    runtime: GeneralistRuntime,
    *,
    target_parameters: int,
) -> tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]]:
    """Grow a Phase-5 candidate before long training and retain compatible weights."""
    source_parameters = parameter_count(runtime.model)
    target = max(source_parameters + 1, int(target_parameters))
    grown_genome = progressive_scale_candidate(
        base_genome,
        target_parameters=target,
        vocab_size=runtime.tokenizer.vocab_size,
        max_width=512,
        max_layers=12,
        prefer_function_preserving=True,
    )
    config = grown_genome.model_config(runtime.tokenizer.vocab_size)
    model = CausalTransformerLM(config)
    transfer = _transfer_compatible_weights(
        runtime.model,
        model,
        source_tokenizer=runtime.tokenizer,
        target_tokenizer=runtime.tokenizer,
    )
    if not bool(transfer.get("function_preserving_growth")):
        raise RuntimeError(
            "Phase-5 capacity growth must preserve the learned function; "
            "refusing destructive width/topology reset"
        )
    grown = GeneralistRuntime(
        model,
        config,
        tokenizer=runtime.tokenizer,
        device="cpu",
    )
    return grown_genome, grown, {
        "version": PHASE5_CAPACITY_GROWTH_VERSION,
        "requested_parameters": int(target_parameters),
        "source_parameters": int(source_parameters),
        "parameters": int(parameter_count(model)),
        "genome": grown_genome.to_dict(),
        "weight_transfer": transfer,
    }


def _batch(blocks: list[list[int]], indices: list[int], *, device):
    import torch
    ids = torch.tensor([blocks[i] for i in indices], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[labels == PAD] = -100
    return ids, labels


def _effective_bootstrap_target(
    requested_target: int,
    persisted_progress: dict[str, Any] | None,
) -> int:
    """Never shrink an already-started cumulative bootstrap rung.

    Push-triggered workflow invocations historically defaulted to 1M tokens.
    After a 5M rung completed, those maintenance runs could rebuild the
    displayed manifest at 1M even though the persisted candidate/progress still
    represented 5M.  The cumulative target is part of state and therefore must
    be monotonic.
    """
    requested = max(100_000, int(requested_target))
    persisted = int((persisted_progress or {}).get("target_tokens", 0) or 0)
    return max(requested, persisted)


def _learning_rate(
    *,
    base_lr: float,
    processed_tokens: int,
    target_tokens: int,
    warmup_tokens: int,
) -> float:
    processed = max(0, int(processed_tokens))
    target = max(1, int(target_tokens))
    warmup = max(1, min(int(warmup_tokens), target // 3))
    if processed < warmup:
        return float(base_lr) * max(0.05, processed / warmup)
    progress = min(1.0, (processed - warmup) / max(1, target - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (0.10 + 0.90 * cosine)


def _phase5_success(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(after.get("language_nll", float("inf"))) >= float(before.get("language_nll", float("inf"))) - 0.02:
        reasons.append("language NLL did not improve by at least 0.02 byte-NLL")
    if bool(after.get("pathological_repetition")):
        reasons.append("pathological repetition remains")
    if float(after.get("repetition_rate", 1.0)) > min(
        0.60,
        float(before.get("repetition_rate", 0.0)) + 0.02,
    ):
        reasons.append("repetition rate did not improve enough")
    if float(after.get("non_empty_rate", 0.0)) < 0.70:
        reasons.append("non-empty rate is below 70%")
    if float(after.get("word_output_rate", 0.0)) < 0.40:
        reasons.append("fewer than 40% of held-out prompts produce word-like output")
    if float(after.get("multiword_output_rate", 0.0)) < 0.40:
        reasons.append("fewer than 40% of held-out prompts produce multi-word output")
    return (not reasons), reasons


def _write_latest_research(
    root: Path,
    runtime: GeneralistRuntime,
    *,
    report: dict[str, Any],
    cycle: int,
    base_model_sha256: str,
) -> dict[str, Any]:
    latest = root / "latest-research"
    tmp = root / ".phase5-latest-research"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    runtime.save_checkpoint(
        tmp,
        metadata={
            "role": "phase5_language_bootstrap",
            "research_only": True,
            "production_qualified": False,
            "cycle": int(cycle),
            "base_champion_model_sha256": base_model_sha256,
        },
    )
    _atomic_json(tmp / "research-metrics.json", report)
    summary = {
        "candidate_id": f"phase5-bootstrap-{cycle}",
        "kind": "language_bootstrap",
        "cycle": int(cycle),
        "stage": 5,
        "parameters": int(report.get("parameters", 0) or 0),
        "score": float(report.get("score", 0.0) or 0.0),
        "mean_nll_per_byte": float(report.get("nll_per_byte", 0.0) or 0.0),
        "mean_generation_accuracy": float(report.get("generation_exact_accuracy", 0.0) or 0.0),
        "mean_generation_similarity": float(report.get("generation_similarity", 0.0) or 0.0),
        "mean_generation_nonempty_rate": float(report.get("generation_nonempty_rate", 0.0) or 0.0),
        "worst_domain_regression": float(report.get("worst_domain_regression", 0.0) or 0.0),
        "all_seed_eligible": bool(report.get("research_gate_passed")),
        "any_seed_eligible": bool(report.get("research_gate_passed")),
        "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
        "research_only": True,
        "external_pretrained": False,
    }
    _atomic_json(tmp / "research-summary.json", summary)
    shutil.rmtree(latest, ignore_errors=True)
    tmp.replace(latest)
    return summary


def _frequent_word_documents(documents):
    counts: Counter[str] = Counter()
    parsed: list[tuple[Any, list[str]]] = []
    for document in documents:
        words = re.findall(r"[^\\W\\d_]+", document.text.casefold(), flags=re.UNICODE)
        parsed.append((document, words))
        counts.update(words)
    frequent = {word for word, _count in counts.most_common(768)}
    selected = [
        document
        for document, words in parsed
        if 1 <= len(words) <= 12
        and words
        and sum(word in frequent for word in words) / len(words) >= 0.70
        and len(document.text) <= 100
    ]
    return selected or [row for row in documents if len(row.text) <= 100] or list(documents)


def _causal_curriculum_stage(processed: int, target: int) -> str:
    ratio = max(0.0, min(1.0, processed / max(1, target)))
    if ratio < 0.15:
        return "A_frequent_word_contexts"
    if ratio < 0.40:
        return "B_short_sentence_completion"
    return "C_causal_next_sentence"


def _anti_collapse_weights(
    stage: str,
    diagnostics: dict[str, Any],
) -> tuple[float, float]:
    pathological = bool(diagnostics.get("pathological_repetition"))
    repetition = float(diagnostics.get("repetition_rate", 0.0) or 0.0)
    if not pathological and repetition < 0.60:
        return 0.0, 1.0
    if stage == "A_frequent_word_contexts":
        return 0.0, 1.20
    if stage == "B_short_sentence_completion":
        return 0.050, 1.60
    return 0.100, 2.00


def _anti_collapse_rescue_gate(
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    before_rep = float(before.get("repetition_rate", 1.0) or 1.0)
    after_rep = float(after.get("repetition_rate", 1.0) or 1.0)
    before_nll = float(before.get("language_nll", float("inf")))
    after_nll = float(after.get("language_nll", float("inf")))

    if after_nll > before_nll + 0.08:
        reasons.append("language NLL regressed by more than 0.08")
    if float(after.get("non_empty_rate", 0.0) or 0.0) < max(
        0.50,
        float(before.get("non_empty_rate", 0.0) or 0.0) - 0.15,
    ):
        reasons.append("non-empty generation rate regressed")
    if float(after.get("token_entropy", 0.0) or 0.0) < max(
        1.50,
        float(before.get("token_entropy", 0.0) or 0.0) - 0.50,
    ):
        reasons.append("token entropy collapsed")
    if float(after.get("generation_similarity", 0.0) or 0.0) < (
        float(before.get("generation_similarity", 0.0) or 0.0) - 0.05
    ):
        reasons.append("generation similarity regressed")
    if (
        bool(after.get("pathological_repetition"))
        and before_rep - after_rep < 0.025
    ):
        reasons.append("pathological repetition did not improve by at least 0.025")
    if int(after.get("longest_repeated_token_run", 0) or 0) > (
        int(before.get("longest_repeated_token_run", 0) or 0) + 3
    ):
        reasons.append("longest repeated-token run worsened")
    return (not reasons), reasons



def _language_quality(report: dict[str, Any]) -> float:
    """Small deterministic scalar used only to compare the same held-out suite."""
    nll = float(report.get("language_nll", 1_000_000.0) or 1_000_000.0)
    repetition = float(report.get("repetition_rate", 1.0) or 1.0)
    similarity = float(report.get("generation_similarity", 0.0) or 0.0)
    multiword = float(report.get("multiword_output_rate", 0.0) or 0.0)
    nonempty = float(report.get("non_empty_rate", 0.0) or 0.0)
    entropy = min(5.0, max(0.0, float(report.get("token_entropy", 0.0) or 0.0)))
    pathological = 1.0 if bool(report.get("pathological_repetition")) else 0.0
    return (
        -nll
        - 1.50 * repetition
        + 1.50 * similarity
        + 0.75 * multiword
        + 0.25 * nonempty
        + 0.10 * entropy
        - 0.50 * pathological
    )


def _language_guard_violations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    """Return material held-out language regressions versus a durable anchor."""
    reasons: list[str] = []
    ref_nll = float(reference.get("language_nll", float("inf")))
    cand_nll = float(candidate.get("language_nll", float("inf")))
    ref_rep = float(reference.get("repetition_rate", 1.0) or 1.0)
    cand_rep = float(candidate.get("repetition_rate", 1.0) or 1.0)
    ref_similarity = float(reference.get("generation_similarity", 0.0) or 0.0)
    cand_similarity = float(candidate.get("generation_similarity", 0.0) or 0.0)
    ref_multi = float(reference.get("multiword_output_rate", 0.0) or 0.0)
    cand_multi = float(candidate.get("multiword_output_rate", 0.0) or 0.0)
    ref_nonempty = float(reference.get("non_empty_rate", 0.0) or 0.0)
    cand_nonempty = float(candidate.get("non_empty_rate", 0.0) or 0.0)
    ref_entropy = float(reference.get("token_entropy", 0.0) or 0.0)
    cand_entropy = float(candidate.get("token_entropy", 0.0) or 0.0)

    if cand_nll > ref_nll + 0.15:
        reasons.append("language NLL exceeds durable anchor by more than 0.15")
    if cand_rep > ref_rep + 0.08:
        reasons.append("repetition rate exceeds durable anchor by more than 0.08")
    if (
        not bool(reference.get("pathological_repetition"))
        and bool(candidate.get("pathological_repetition"))
    ):
        reasons.append("pathological repetition reappeared")
    if cand_similarity < ref_similarity - 0.05:
        reasons.append("generation similarity fell below durable anchor")
    if cand_multi < max(0.0, ref_multi - 0.15):
        reasons.append("multiword output rate fell below durable anchor")
    if cand_nonempty < max(0.50, ref_nonempty - 0.10):
        reasons.append("non-empty generation rate fell below durable anchor")
    if cand_entropy < max(1.25, ref_entropy - 0.60):
        reasons.append("token entropy fell below durable anchor")
    return reasons


def _segment_language_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    anchor: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Fail closed when a training segment makes the live AIRI language worse.

    If the lineage is already below a historical best anchor, enter recovery
    mode: a segment may be accepted only when it moves held-out quality toward
    that anchor without introducing a new material regression.
    """
    before_anchor = _language_guard_violations(anchor, before)
    after_anchor = _language_guard_violations(anchor, after)
    local_reasons: list[str] = []

    before_nll = float(before.get("language_nll", float("inf")))
    after_nll = float(after.get("language_nll", float("inf")))
    before_rep = float(before.get("repetition_rate", 1.0) or 1.0)
    after_rep = float(after.get("repetition_rate", 1.0) or 1.0)
    before_similarity = float(before.get("generation_similarity", 0.0) or 0.0)
    after_similarity = float(after.get("generation_similarity", 0.0) or 0.0)

    if after_nll > before_nll + 0.08:
        local_reasons.append("segment language NLL regressed by more than 0.08")
    if after_rep > before_rep + 0.06:
        local_reasons.append("segment repetition rate regressed by more than 0.06")
    if (
        not bool(before.get("pathological_repetition"))
        and bool(after.get("pathological_repetition"))
    ):
        local_reasons.append("segment introduced pathological repetition")
    if after_similarity < before_similarity - 0.04:
        local_reasons.append("segment generation similarity regressed")
    if float(after.get("multiword_output_rate", 0.0) or 0.0) < max(
        0.0,
        float(before.get("multiword_output_rate", 0.0) or 0.0) - 0.15,
    ):
        local_reasons.append("segment multiword output rate regressed")

    before_quality = _language_quality(before)
    after_quality = _language_quality(after)
    recovery_mode = bool(before_anchor)

    if recovery_mode:
        improved = (
            after_quality >= before_quality + 0.005
            or len(after_anchor) < len(before_anchor)
        )
        if not improved:
            local_reasons.append(
                "lineage is below its durable language anchor and this segment "
                "did not measurably recover quality"
            )
        accepted = not local_reasons
    else:
        accepted = not local_reasons and not after_anchor

    return bool(accepted), {
        "accepted": bool(accepted),
        "recovery_mode": recovery_mode,
        "before_quality": before_quality,
        "after_quality": after_quality,
        "anchor_quality": _language_quality(anchor),
        "before_anchor_violations": before_anchor,
        "after_anchor_violations": after_anchor,
        "local_reasons": local_reasons,
        "validation_before": before,
        "validation_after": after,
        "anchor": anchor,
    }


def _run_anti_collapse_rescue(
    runtime: GeneralistRuntime,
    stage_blocks: dict[str, list[list[int]]],
    *,
    bootstrap_root: Path,
    base_model_sha: str,
    seed: int,
    steps: int = 128,
    batch_size: int = 16,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    import torch

    before = evaluate_phase5_language(runtime)
    if (
        not bool(before.get("pathological_repetition"))
        and float(before.get("repetition_rate", 0.0) or 0.0) < 0.60
    ):
        return runtime, {
            "accepted": False,
            "rolled_back": False,
            "reason": "anti-collapse rescue not needed",
            "validation_before": before,
            "validation_after": before,
            "steps": 0,
            "tokens_processed": 0,
        }

    pre_dir = bootstrap_root / "pre-anticollapse"
    trial_dir = bootstrap_root / ".anticollapse-selected"
    shutil.rmtree(pre_dir, ignore_errors=True)
    shutil.rmtree(trial_dir, ignore_errors=True)
    runtime.save_checkpoint(
        pre_dir,
        metadata={
            "role": "phase5_pre_anticollapse_checkpoint",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
        },
    )
    trial = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
    optimizer = torch.optim.AdamW(
        trial.model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
    )
    rng = random.Random(int(seed))
    objective_tail: dict[str, Any] = {}
    losses: list[float] = []
    trained_tokens = 0
    trial.model.train()

    for step in range(max(1, int(steps))):
        stage = (
            "B_short_sentence_completion"
            if step % 3 != 2
            else "C_causal_next_sentence"
        )
        blocks = stage_blocks[stage]
        indices = [
            rng.randrange(len(blocks))
            for _ in range(max(1, int(batch_size)))
        ]
        ids, labels = _batch(blocks, indices, device=trial.device)
        optimizer.zero_grad(set_to_none=True)
        result = trial.model(ids)
        loss, objective_tail = causal_training_objective(
            result["logits"],
            labels,
            ids,
            eos_loss_weight=2.0,
            repetition_unlikelihood_weight=0.08,
            repetition_window=16,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite anti-collapse rescue loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trial.model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        trained_tokens += int((labels[:, 1:] != -100).sum().item())

    after = evaluate_phase5_language(trial)
    accepted, reasons = _anti_collapse_rescue_gate(before, after)
    report = {
        "accepted": bool(accepted),
        "rolled_back": not bool(accepted),
        "reason": (
            "held-out anti-collapse rescue improved degeneration safely"
            if accepted
            else "anti-collapse rescue rejected by held-out safety gate"
        ),
        "gate_reasons": reasons,
        "validation_before": before,
        "validation_after": after,
        "steps": max(1, int(steps)),
        "tokens_processed": int(trained_tokens),
        "mean_objective_loss": sum(losses) / max(1, len(losses)),
        "last_objective": objective_tail,
        "external_pretrained_weights": False,
        "decoding_modified": False,
    }

    if not accepted:
        restored = GeneralistRuntime.from_checkpoint(pre_dir, device="cpu")
        shutil.rmtree(trial_dir, ignore_errors=True)
        return restored, report

    trial.save_checkpoint(
        trial_dir,
        metadata={
            "role": "phase5_anticollapse_rescue_selected",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "steps": max(1, int(steps)),
            "tokens_processed": int(trained_tokens),
        },
    )
    selected = GeneralistRuntime.from_checkpoint(trial_dir, device="cpu")
    shutil.rmtree(trial_dir, ignore_errors=True)
    return selected, report


def _sft_row_fingerprint(row) -> str:
    raw = json.dumps(row.messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normal_content(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def _filter_protected_replay(rows, *, heldout_sft=()):
    heldout_hashes = {
        _sft_row_fingerprint(row)
        for row in heldout_sft
    }
    protected_texts = protected_bootstrap_texts()
    unique = {}
    filtered = 0
    for row in rows:
        key = _sft_row_fingerprint(row)
        contents = {
            _normal_content(message.get("content", ""))
            for message in row.messages
        }
        if key in heldout_hashes or bool(contents & protected_texts):
            filtered += 1
            continue
        unique.setdefault(key, row)
    return list(unique.values()), int(filtered)


def _mixed_replay_rows(
    root: Path,
    bootstrap_sft,
    repo: Path,
    *,
    heldout_sft=(),
) -> tuple[list, dict[str, int]]:
    one_turn = [row for row in bootstrap_sft if len(row.messages) <= 2]
    multi_turn = [row for row in bootstrap_sft if len(row.messages) > 2]
    base = [row.sft() for row in train_rows()]
    lab = build_airi_pc_lab_rows(snapshot_airi_pc_lab(repo), max_rows=18)
    verified = load_verified_lab_experiences(root, max_rows=32)

    stage_rows = {
        "D_simple_prompt_response": one_turn[:900] + base[:200],
        "E_simple_multiturn_chat": multi_turn[:600] + one_turn[:300],
        "F_airi_pc_lab_tool_use": [row.sft() for row in (lab + verified)] + base,
    }
    rows = []
    counts: dict[str, int] = {}
    for stage, stage_items in stage_rows.items():
        counts[stage] = len(stage_items)
        rows.extend(stage_items)
    try:
        memory = CurriculumMemory(root, max_rows=4000)
        memory_rows = [row.sft() for row in memory.rows()[:2400]]
        rows.extend(memory_rows)
        counts["persistent_replay"] = len(memory_rows)
    except Exception:
        counts["persistent_replay"] = 0

    # Deterministic de-duplication plus structural exclusion of every
    # held-out SFT row and the canonical Phase-5 evaluation strings.
    filtered_rows, filtered = _filter_protected_replay(
        rows,
        heldout_sft=heldout_sft,
    )
    counts["held_out_filtered"] = int(filtered)
    return filtered_rows, counts


def _sft_attempt_score(
    before: dict[str, Any],
    after: dict[str, Any],
) -> float:
    return float(
        (float(before.get("language_nll", 0.0)) - float(after.get("language_nll", 0.0)))
        + 0.75 * (
            float(before.get("repetition_rate", 0.0))
            - float(after.get("repetition_rate", 0.0))
        )
        + 0.25 * (
            float(after.get("generation_similarity", 0.0))
            - float(before.get("generation_similarity", 0.0))
        )
    )


def _run_guarded_sft(
    runtime: GeneralistRuntime,
    replay,
    heldout_sft,
    *,
    bootstrap_root: Path,
    base_model_sha: str,
    target_tokens: int,
    seed: int,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    pre_sft_dir = bootstrap_root / "pre-sft"
    selected_dir = bootstrap_root / ".sft-selected"
    shutil.rmtree(pre_sft_dir, ignore_errors=True)
    shutil.rmtree(selected_dir, ignore_errors=True)
    runtime.save_checkpoint(
        pre_sft_dir,
        metadata={
            "role": "phase5_pre_sft_checkpoint",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "target_tokens": int(target_tokens),
        },
    )

    if not replay:
        return runtime, {
            "accepted": False,
            "rolled_back": True,
            "reason": "no replay rows",
            "attempts": [],
        }
    heldout = list(heldout_sft)
    if not heldout:
        return runtime, {
            "accepted": False,
            "rolled_back": True,
            "reason": "no held-out SFT validation rows",
            "attempts": [],
        }

    before = evaluate_sft_validation(
        runtime,
        heldout,
        max_examples=24,
        max_new_tokens=32,
    )
    collapsed = bool(before.get("pathological_repetition")) or (
        float(before.get("repetition_rate", 0.0) or 0.0) >= 0.60
    )
    parameters = parameter_count(runtime.model)
    sft_anti_collapse_weight = 0.08 if collapsed else 0.0
    sft_eos_loss_weight = 1.75 if collapsed else 1.0
    if int(target_tokens) >= 100_000_000 or parameters >= 3_000_000:
        attempts = (
            {"steps": 1600, "learning_rate": 4e-5},
            {"steps": 800, "learning_rate": 2.5e-5},
            {"steps": 320, "learning_rate": 1.25e-5},
        )
    elif int(target_tokens) >= 20_000_000 or parameters >= 750_000:
        attempts = (
            {"steps": 800, "learning_rate": 5e-5},
            {"steps": 400, "learning_rate": 3e-5},
            {"steps": 160, "learning_rate": 1.5e-5},
        )
    elif parameters >= 250_000:
        attempts = (
            {"steps": 320, "learning_rate": 5e-5},
            {"steps": 160, "learning_rate": 3e-5},
            {"steps": 80, "learning_rate": 1.5e-5},
        )
    else:
        attempts = (
            {"steps": 120, "learning_rate": 5e-5},
            {"steps": 60, "learning_rate": 3e-5},
            {"steps": 30, "learning_rate": 1.5e-5},
        )
    reports: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_index: int | None = None

    for index, config in enumerate(attempts):
        trial = GeneralistRuntime.from_checkpoint(pre_sft_dir, device="cpu")
        train_report = train_sft(
            trial.model,
            trial.tokenizer,
            replay,
            steps=int(config["steps"]),
            batch_size=8,
            learning_rate=float(config["learning_rate"]),
            weight_decay=0.01,
            seed=int(seed + index * 101),
            device="cpu",
            gradient_accumulation_steps=1,
            precision="fp32",
            repetition_unlikelihood_weight=sft_anti_collapse_weight,
            eos_loss_weight=sft_eos_loss_weight,
        )
        after = evaluate_sft_validation(
            trial,
            heldout,
            max_examples=24,
            max_new_tokens=32,
        )
        gate_ok, gate_reasons = sft_validation_gate(before, after)
        score = _sft_attempt_score(before, after)
        report = {
            "index": index,
            "steps": int(config["steps"]),
            "learning_rate": float(config["learning_rate"]),
            "training": train_report,
            "validation": after,
            "gate_passed": bool(gate_ok),
            "gate_reasons": list(gate_reasons),
            "selection_score": float(score),
            "anti_collapse_objective": {
                "enabled": bool(sft_anti_collapse_weight > 0.0),
                "repetition_unlikelihood_weight": float(sft_anti_collapse_weight),
                "eos_loss_weight": float(sft_eos_loss_weight),
                "decoding_modified": False,
            },
        }
        reports.append(report)
        if gate_ok and score > best_score:
            best_score = float(score)
            best_index = index
            shutil.rmtree(selected_dir, ignore_errors=True)
            trial.save_checkpoint(
                selected_dir,
                metadata={
                    "role": "phase5_guarded_sft_selected",
                    "production_qualified": False,
                    "base_champion_model_sha256": base_model_sha,
                    "target_tokens": int(target_tokens),
                    "attempt_index": int(index),
                },
            )

    if best_index is None:
        restored = GeneralistRuntime.from_checkpoint(pre_sft_dir, device="cpu")
        shutil.rmtree(selected_dir, ignore_errors=True)
        return restored, {
            "accepted": False,
            "rolled_back": True,
            "reason": "all SFT attempts failed held-out degeneration gate",
            "validation_before": before,
            "attempts": reports,
        }

    selected = GeneralistRuntime.from_checkpoint(selected_dir, device="cpu")
    shutil.rmtree(selected_dir, ignore_errors=True)
    return selected, {
        "accepted": True,
        "rolled_back": False,
        "reason": "best held-out-safe SFT attempt selected",
        "validation_before": before,
        "selected_attempt": int(best_index),
        "selected_score": float(best_score),
        "attempts": reports,
    }


def run_segment(
    state_dir: str | Path,
    repo_root: str | Path,
    cache_dir: str | Path,
    *,
    target_tokens: int,
    segment_tokens: int = 250_000,
    batch_size: int = 16,
    base_learning_rate: float = 3e-4,
    eval_every_steps: int = 32,
) -> dict[str, Any]:
    import torch

    root = Path(state_dir).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    bootstrap_root = root / "bootstrap-data"
    candidate_dir = bootstrap_root / "candidate"
    progress_path = bootstrap_root / "progress.json"
    optimizer_path = bootstrap_root / "optimizer.pt"
    manifest_path = bootstrap_root / "manifest.json"
    before_path = bootstrap_root / "before.json"
    after_path = bootstrap_root / "after.json"
    language_guard_path = bootstrap_root / "language-guard.json"
    segment_guard_path = bootstrap_root / "segment-guard-last.json"
    segment_best_dir = bootstrap_root / ".segment-best"
    bootstrap_root.mkdir(parents=True, exist_ok=True)

    status = _load_json(root / "status.json")
    cycle = int(status.get("cycle", 0) or 0)
    loaded = _load_champion(root, device="cpu")
    if loaded is None:
        raise RuntimeError("Phase 5 requires an existing champion checkpoint")
    champion_genome, champion_runtime = loaded
    champion_model_path = root / "champion" / "model.pt"
    base_model_sha = _sha256_file(champion_model_path)

    persisted_progress = _load_json(progress_path, {}) if progress_path.is_file() else {}
    target_tokens = _effective_bootstrap_target(target_tokens, persisted_progress)

    previous_manifest = _load_json(manifest_path) if manifest_path.is_file() else None
    corpus_target_tokens = _bootstrap_corpus_target(target_tokens)
    bundle = build_bootstrap_bundle(
        champion_runtime.tokenizer,
        cache_dir=cache,
        target_tokens=int(corpus_target_tokens),
        previous_manifest=previous_manifest,
    )
    if int(bundle.manifest.get("actual_selected_tokens", 0) or 0) < int(corpus_target_tokens * 0.95):
        raise RuntimeError(
            "bootstrap corpus coverage is below 95% of reviewed unique-corpus target: "
            f"selected={bundle.manifest.get('actual_selected_tokens')} "
            f"target={corpus_target_tokens}"
        )
    _atomic_json(manifest_path, bundle.manifest)
    replay_manifest = write_bootstrap_replay(
        bundle,
        champion_runtime.tokenizer,
        output_dir=bootstrap_root,
        max_tokens=min(1_000_000, max(250_000, int(target_tokens // 4))),
        max_sft_conversations=512,
    )

    if not before_path.is_file():
        _atomic_json(before_path, evaluate_phase5_language(champion_runtime))
    before = _load_json(before_path)

    progress = _load_json(progress_path, {
        "schema": 1,
        "version": PHASE5_BOOTSTRAP_VERSION,
        "base_champion_model_sha256": base_model_sha,
        "target_tokens": int(target_tokens),
        "tokens_processed": 0,
        "steps": 0,
        "best_validation_loss": None,
        "bad_eval_count": 0,
        "completed_rungs": [],
        "sft_completed_rungs": [],
    })
    progress["schema"] = 1
    progress["version"] = PHASE5_BOOTSTRAP_VERSION
    rebase_to_champion = False
    if str(progress.get("base_champion_model_sha256")) != base_model_sha:
        # A completed previous rung may have gone through the converged swarm
        # and promoted a jointly-trained descendant. Resume the next language
        # rung from that stronger champion, never from stale pre-fusion weights.
        # A mid-rung champion change still fails closed.
        previous_target = int(progress.get("target_tokens", 0) or 0)
        previous_processed = int(progress.get("tokens_processed", 0) or 0)
        previous_sft = {
            int(value)
            for value in (progress.get("sft_completed_rungs") or [])
            if isinstance(value, (int, float))
        }
        if (
            previous_processed < previous_target
            or previous_target not in previous_sft
        ):
            raise RuntimeError("champion changed during an incomplete Phase 5 bootstrap")
        progress["base_champion_model_sha256"] = base_model_sha
        rebase_to_champion = True

    progress["target_tokens"] = max(int(progress.get("target_tokens", 0) or 0), int(target_tokens))
    target_tokens = int(progress["target_tokens"])
    progress["unique_corpus_target_tokens"] = int(corpus_target_tokens)
    progress["optimization_passes_target"] = float(
        target_tokens / max(1, corpus_target_tokens)
    )

    candidate_genome = champion_genome
    capacity_genome_raw = progress.get("capacity_genome")
    if not rebase_to_champion and isinstance(capacity_genome_raw, dict):
        try:
            candidate_genome = GeneralistGenome(**capacity_genome_raw).validate()
        except Exception:
            candidate_genome = champion_genome

    if rebase_to_champion:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        optimizer_path.unlink(missing_ok=True)
        shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-sft", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-anticollapse", ignore_errors=True)
        progress["capacity_genome"] = champion_genome.to_dict()
        progress["capacity_rebased_from_champion"] = {
            "model_sha256": base_model_sha,
            "parameters": int(parameter_count(champion_runtime.model)),
            "training_target_tokens": int(target_tokens),
        }

    if candidate_dir.is_dir():
        runtime = GeneralistRuntime.from_checkpoint(candidate_dir, device="cpu")
    else:
        runtime = GeneralistRuntime(
            copy.deepcopy(champion_runtime.model),
            champion_runtime.config,
            tokenizer=champion_runtime.tokenizer,
            device="cpu",
        )
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "rebased_from_converged_champion": bool(rebase_to_champion),
            },
        )

    desired_capacity = _bootstrap_capacity_target(target_tokens)
    current_capacity = int(parameter_count(runtime.model))
    assisted_capacity = _assisted_capacity_target(
        progress,
        current_parameters=current_capacity,
    )
    if assisted_capacity is not None:
        desired_capacity = max(int(desired_capacity or 0), int(assisted_capacity))

    if desired_capacity is not None and current_capacity < int(desired_capacity):
        candidate_genome, runtime, growth = _grow_bootstrap_runtime(
            candidate_genome,
            runtime,
            target_parameters=int(desired_capacity),
        )
        progress["capacity_target_parameters"] = int(desired_capacity)
        progress["capacity_genome"] = candidate_genome.to_dict()
        history = list(progress.get("capacity_growth_history") or [])
        history.append({
            **growth,
            "training_target_tokens": int(target_tokens),
            "tokens_processed_before_growth": int(progress.get("tokens_processed", 0) or 0),
        })
        progress["capacity_growth_history"] = history
        grown_parameters = int(parameter_count(runtime.model))
        if (
            assisted_capacity is not None
            and grown_parameters >= ASSISTED_CAPACITY_COMPLETION_FLOOR
        ):
            progress["assisted_capacity_growth_completed"] = True
            progress["assisted_capacity_growth"] = {
                "lineage_id": ASSISTED_CAPACITY_LINEAGE_ID,
                "requested_parameters": int(ASSISTED_CAPACITY_TARGET_PARAMETERS),
                "actual_parameters": grown_parameters,
                "tokens_processed_before_growth": int(
                    progress.get("tokens_processed", 0) or 0
                ),
                "generation": int(candidate_genome.generation),
                "genome_id": str(candidate_genome.genome_id),
                "function_preserving_growth": bool(
                    (growth.get("weight_transfer") or {}).get(
                        "function_preserving_growth"
                    )
                ),
            }
        progress["best_validation_loss"] = None
        progress["bad_eval_count"] = 0
        optimizer_path.unlink(missing_ok=True)
        shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-sft", ignore_errors=True)
        shutil.rmtree(bootstrap_root / "pre-anticollapse", ignore_errors=True)
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "capacity_growth": growth,
            },
        )

    # Packing a 20M-token corpus is expensive and previously repeated for
    # every resumable segment.  Cache the exact fixed-width token blocks in the
    # job-local data cache.  The cache identity includes the reviewed manifest,
    # tokenizer and context length, so stale data fails closed and is rebuilt.
    manifest_identity = str(bundle.manifest.get("manifest_content_sha256") or "")
    packed_cache_root = cache / "phase5-packed-blocks-v1"
    packed_cache_root.mkdir(parents=True, exist_ok=True)
    packed_cache_base = {
        "version": "phase5-stage-block-cache-v1",
        "manifest_content_sha256": manifest_identity,
        "tokenizer_version": str(runtime.tokenizer.version),
        "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
        "context_length": int(runtime.config.context_length),
    }

    def _packed_identity(split: str) -> dict[str, Any]:
        return {**packed_cache_base, "split": str(split)}

    def _packed_path(split: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(split))
        digest = hashlib.sha256(
            json.dumps(_packed_identity(split), sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        return packed_cache_root / f"{safe}-{digest}.bin"

    stage_blocks: dict[str, Any] = {}
    packed_cache_hits: dict[str, bool] = {}
    stage_names = (
        "A_frequent_word_contexts",
        "B_short_sentence_completion",
        "C_causal_next_sentence",
    )
    for stage in stage_names:
        cached = load_packed_block_cache(
            _packed_path(stage),
            expected_identity=_packed_identity(stage),
        )
        if cached is not None:
            stage_blocks[stage] = cached
            packed_cache_hits[stage] = True
        else:
            packed_cache_hits[stage] = False

    missing_stages = [stage for stage in stage_names if stage not in stage_blocks]
    if "A_frequent_word_contexts" in missing_stages:
        documents = _frequent_word_documents(bundle.train_documents)
        packed = pack_causal_blocks(
            documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("A_frequent_word_contexts"),
            packed,
            block_size=runtime.config.context_length,
            identity=_packed_identity("A_frequent_word_contexts"),
        )
        stage_blocks["A_frequent_word_contexts"] = packed

    if "B_short_sentence_completion" in missing_stages:
        documents = [
            row for row in bundle.train_documents
            if len(row.text) <= 220
        ] or list(bundle.train_documents)
        packed = pack_causal_blocks(
            documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("B_short_sentence_completion"),
            packed,
            block_size=runtime.config.context_length,
            identity=_packed_identity("B_short_sentence_completion"),
        )
        stage_blocks["B_short_sentence_completion"] = packed

    if "C_causal_next_sentence" in missing_stages:
        packed = pack_causal_blocks(
            bundle.train_documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("C_causal_next_sentence"),
            packed,
            block_size=runtime.config.context_length,
            identity=_packed_identity("C_causal_next_sentence"),
        )
        stage_blocks["C_causal_next_sentence"] = packed

    validation_identity = _packed_identity("validation")
    validation_blocks = load_packed_block_cache(
        _packed_path("validation"),
        expected_identity=validation_identity,
    )
    packed_cache_hits["validation"] = validation_blocks is not None
    if validation_blocks is None:
        validation_blocks = pack_causal_blocks(
            bundle.validation_documents,
            runtime.tokenizer,
            context_length=runtime.config.context_length,
        )
        save_packed_block_cache(
            _packed_path("validation"),
            validation_blocks,
            block_size=runtime.config.context_length,
            identity=validation_identity,
        )

    if any(not rows for rows in stage_blocks.values()) or not validation_blocks:
        raise RuntimeError("bootstrap corpus did not produce curriculum train/validation blocks")

    progress["packed_block_cache"] = {
        "version": "phase5-stage-block-cache-v1",
        "manifest_content_sha256": manifest_identity,
        "hits": packed_cache_hits,
        "block_counts": {
            stage: len(rows) for stage, rows in stage_blocks.items()
        },
        "validation_blocks": len(validation_blocks),
    }

    progress["bootstrap_replay"] = replay_manifest
    progress["curriculum_schedule"] = [
        "A_frequent_word_contexts",
        "B_short_sentence_completion",
        "C_causal_next_sentence",
        "D_simple_prompt_response",
        "E_simple_multiturn_chat",
        "F_airi_pc_lab_tool_use",
    ]

    optimizer = torch.optim.AdamW(
        runtime.model.parameters(),
        lr=float(base_learning_rate),
        weight_decay=0.01,
    )
    if optimizer_path.is_file():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))

    segment_language_before = evaluate_phase5_language(runtime)
    guard_payload = _load_json(language_guard_path) if language_guard_path.is_file() else {}
    guard_anchor = (
        dict(guard_payload.get("report") or {})
        if isinstance(guard_payload, dict)
        else {}
    )
    if not guard_anchor:
        historical = _load_json(after_path) if after_path.is_file() else {}
        if (
            isinstance(historical, dict)
            and historical
            and _language_quality(historical) > _language_quality(segment_language_before)
        ):
            guard_anchor = historical
            guard_source = "best_completed_rung"
        else:
            guard_anchor = segment_language_before
            guard_source = "current_live_lineage"
        _atomic_json(
            language_guard_path,
            {
                "schema": 1,
                "version": "phase5-segment-language-guard-v1",
                "source": guard_source,
                "report": guard_anchor,
                "created_at_unix": int(time.time()),
            },
        )

    progress_before_segment = copy.deepcopy(progress)
    consecutive_rejections = int(
        progress.get("segment_guard_consecutive_rejections", 0) or 0
    )
    segment_lr_scale = max(
        0.125,
        min(1.0, float(progress.get("segment_guard_lr_scale", 1.0) or 1.0)),
    )
    retry_seed_offset = int(consecutive_rejections) * 1_000_003
    shutil.rmtree(segment_best_dir, ignore_errors=True)

    processed_before_segment = int(progress.get("tokens_processed", 0) or 0)
    segment_budget = max(1, int(segment_tokens))
    eval_every_steps = max(8, int(eval_every_steps))
    best_loss = progress.get("best_validation_loss")
    best_loss = float(best_loss) if best_loss is not None else float("inf")
    bad_eval_count = int(progress.get("bad_eval_count", 0) or 0)
    early_stopped = False
    last_train_loss = None
    last_validation_loss = None

    runtime.model.train()
    while (
        int(progress["tokens_processed"]) < target_tokens
        and int(progress["tokens_processed"]) - processed_before_segment < segment_budget
    ):
        step = int(progress["steps"])
        stage = _causal_curriculum_stage(
            int(progress["tokens_processed"]),
            target_tokens,
        )
        progress["curriculum_stage"] = stage
        train_blocks = stage_blocks[stage]
        rng = random.Random(5_000_000 + step + retry_seed_offset)
        indices = [rng.randrange(len(train_blocks)) for _ in range(max(1, int(batch_size)))]
        ids, labels = _batch(train_blocks, indices, device=runtime.device)
        optimizer.zero_grad(set_to_none=True)

        lr = _learning_rate(
            base_lr=float(base_learning_rate) * segment_lr_scale,
            processed_tokens=int(progress["tokens_processed"]),
            target_tokens=target_tokens,
            warmup_tokens=min(100_000, max(20_000, target_tokens // 10)),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        result = runtime.model(ids)
        anti_weight, eos_weight = _anti_collapse_weights(stage, before)
        loss, objective_stats = causal_training_objective(
            result["logits"],
            labels,
            ids,
            eos_loss_weight=eos_weight,
            repetition_unlikelihood_weight=anti_weight,
            repetition_window=16,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Phase 5 bootstrap loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0)
        optimizer.step()

        supervised = int((labels[:, 1:] != -100).sum().item())
        progress["tokens_processed"] = int(progress["tokens_processed"]) + supervised
        progress["steps"] = step + 1
        progress["learning_rate"] = float(lr)
        last_train_loss = float(loss.detach().cpu())
        progress["last_train_loss"] = last_train_loss
        progress["last_objective"] = objective_stats
        progress["anti_collapse_training_enabled"] = bool(anti_weight > 0.0)

        if int(progress["steps"]) % eval_every_steps == 0:
            last_validation_loss = corpus_loss(
                runtime.model,
                validation_blocks,
                device="cpu",
                batch_size=max(2, int(batch_size) // 2),
                max_blocks=min(96, len(validation_blocks)),
                seed=991,
            )
            progress["last_validation_loss"] = float(last_validation_loss)
            if last_validation_loss + 1e-4 < best_loss:
                best_loss = float(last_validation_loss)
                bad_eval_count = 0
                runtime.save_checkpoint(
                    segment_best_dir,
                    metadata={
                        "role": "phase5_language_bootstrap_segment_best",
                        "validation_loss": best_loss,
                        "tokens_processed": int(progress["tokens_processed"]),
                    },
                )
            else:
                bad_eval_count += 1
            progress["best_validation_loss"] = best_loss
            progress["bad_eval_count"] = bad_eval_count
            # A freshly grown network needs a substantial fraction of the
            # rescue rung before plateau logic is allowed to stop it.
            minimum_before_early_stop = max(
                600_000,
                int(target_tokens * 0.75),
            )
            if (
                bad_eval_count >= 4
                and int(progress["tokens_processed"]) >= minimum_before_early_stop
            ):
                early_stopped = True
                break
            runtime.model.train()

    segment_language_after = evaluate_phase5_language(runtime)
    segment_accepted, segment_guard = _segment_language_gate(
        segment_language_before,
        segment_language_after,
        guard_anchor,
    )
    attempted_tokens = int(progress["tokens_processed"]) - processed_before_segment
    attempted_steps = int(progress["steps"]) - int(
        progress_before_segment.get("steps", 0) or 0
    )
    segment_guard.update({
        "attempted_tokens": int(attempted_tokens),
        "attempted_steps": int(attempted_steps),
        "learning_rate_scale": float(segment_lr_scale),
        "consecutive_rejections_before": int(consecutive_rejections),
    })
    _atomic_json(segment_guard_path, segment_guard)

    if not segment_accepted:
        # candidate_dir and optimizer.pt still contain the exact pre-segment
        # state because they are written only after this gate passes.
        shutil.rmtree(segment_best_dir, ignore_errors=True)
        progress = progress_before_segment
        history = list(progress.get("segment_guard_rejections") or [])
        history.append({
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "attempted_tokens": int(attempted_tokens),
            "attempted_steps": int(attempted_steps),
            "before_quality": float(segment_guard["before_quality"]),
            "after_quality": float(segment_guard["after_quality"]),
            "recovery_mode": bool(segment_guard["recovery_mode"]),
            "reasons": list(segment_guard.get("local_reasons") or [])
                + list(segment_guard.get("after_anchor_violations") or []),
            "learning_rate_scale": float(segment_lr_scale),
        })
        progress["segment_guard_rejections"] = history[-32:]
        progress["segment_guard_consecutive_rejections"] = int(
            consecutive_rejections + 1
        )
        progress["segment_guard_lr_scale"] = max(
            0.125,
            float(segment_lr_scale) * 0.5,
        )
        progress["segment_guard_last_accepted"] = False
        progress["segment_guard_recovery_mode"] = bool(
            segment_guard.get("recovery_mode")
        )
        progress["early_stopped"] = False
        progress["rung_complete"] = False
        progress["updated_at_unix"] = int(time.time())
        _atomic_json(progress_path, progress)
        lineage_manifest = refresh_live_lineage_manifest(
            root,
            reason="phase5_segment_language_guard_rollback",
        )
        return {
            "ok": True,
            "lineage_id": lineage_manifest.get("lineage_id"),
            "active_lineage_checkpoint": lineage_manifest.get("active_checkpoint"),
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
            "segment_tokens_processed": 0,
            "steps": int(progress.get("steps", 0) or 0),
            "rung_complete": False,
            "early_stopped": False,
            "segment_rejected": True,
            "segment_guard_path": str(segment_guard_path),
            "next_learning_rate_scale": float(progress["segment_guard_lr_scale"]),
        }

    if segment_best_dir.is_dir():
        shutil.rmtree(bootstrap_root / "best", ignore_errors=True)
        segment_best_dir.rename(bootstrap_root / "best")

    progress["segment_guard_consecutive_rejections"] = 0
    progress["segment_guard_lr_scale"] = min(
        1.0,
        max(0.125, float(segment_lr_scale) * 1.20),
    )
    progress["segment_guard_last_accepted"] = True
    progress["segment_guard_recovery_mode"] = bool(segment_guard.get("recovery_mode"))
    accepted_history = list(progress.get("segment_guard_acceptances") or [])
    accepted_history.append({
        "tokens_processed": int(progress["tokens_processed"]),
        "segment_tokens": int(attempted_tokens),
        "before_quality": float(segment_guard["before_quality"]),
        "after_quality": float(segment_guard["after_quality"]),
        "recovery_mode": bool(segment_guard["recovery_mode"]),
        "learning_rate_scale": float(segment_lr_scale),
    })
    progress["segment_guard_acceptances"] = accepted_history[-32:]

    if _language_quality(segment_language_after) > _language_quality(guard_anchor) + 0.005:
        runtime.save_checkpoint(
            bootstrap_root / "language-guard-best",
            metadata={
                "role": "phase5_language_guard_best",
                "tokens_processed": int(progress["tokens_processed"]),
                "language_quality": float(_language_quality(segment_language_after)),
            },
        )
        _atomic_json(
            language_guard_path,
            {
                "schema": 1,
                "version": "phase5-segment-language-guard-v1",
                "source": "accepted_segment_improvement",
                "report": segment_language_after,
                "tokens_processed": int(progress["tokens_processed"]),
                "updated_at_unix": int(time.time()),
            },
        )

    runtime.save_checkpoint(
        candidate_dir,
        metadata={
            "role": "phase5_language_bootstrap_candidate",
            "production_qualified": False,
            "base_champion_model_sha256": base_model_sha,
            "tokens_processed": int(progress["tokens_processed"]),
            "target_tokens": target_tokens,
            "segment_language_guard": True,
        },
    )
    torch.save(optimizer.state_dict(), optimizer_path)

    rung_complete = int(progress["tokens_processed"]) >= target_tokens or early_stopped
    progress["early_stopped"] = bool(early_stopped)
    progress["rung_complete"] = bool(rung_complete)
    progress["updated_at_unix"] = int(time.time())

    rescue_accepted = False
    rescue_due = bool(
        rung_complete
        and (
            str(progress.get("anti_collapse_rescue_version") or "")
            != PHASE5_ANTICOLLAPSE_VERSION
            or str(progress.get("anti_collapse_base_model_sha256") or "")
            != base_model_sha
        )
    )
    if rescue_due:
        runtime_parameters = parameter_count(runtime.model)
        rescue_steps = (
            1024 if runtime_parameters >= 3_000_000
            else 512 if runtime_parameters >= 750_000
            else 256 if runtime_parameters >= 250_000
            else 128
        )
        runtime, rescue_report = _run_anti_collapse_rescue(
            runtime,
            stage_blocks,
            bootstrap_root=bootstrap_root,
            base_model_sha=base_model_sha,
            seed=71 + int(cycle),
            steps=rescue_steps,
        )
        rescue_accepted = bool(rescue_report.get("accepted"))
        progress["anti_collapse_rescue"] = rescue_report
        progress["anti_collapse_rescue_version"] = PHASE5_ANTICOLLAPSE_VERSION
        progress["anti_collapse_base_model_sha256"] = base_model_sha
        progress["anti_collapse_rescue_tokens"] = int(
            progress.get("anti_collapse_rescue_tokens", 0) or 0
        ) + int(rescue_report.get("tokens_processed", 0) or 0)
        if rescue_accepted:
            optimizer_path.unlink(missing_ok=True)
            progress["optimizer_reset_after_anticollapse"] = True
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "tokens_processed": int(progress["tokens_processed"]),
                "target_tokens": target_tokens,
                "anti_collapse_rescue": bool(rescue_accepted),
            },
        )

    completed_sft_before = set(
        int(x) for x in (progress.get("sft_completed_rungs") or [])
    )
    sft_guard_migration = bool(
        rung_complete
        and target_tokens in completed_sft_before
        and str(progress.get("sft_guard_version") or "") != PHASE5_SFT_GUARD_VERSION
    )
    needs_sft = bool(
        rung_complete
        and (
            target_tokens not in completed_sft_before
            or sft_guard_migration
            or rescue_accepted
        )
    )

    if needs_sft:
        progress["curriculum_stage"] = "D_to_F_supervised_replay"
        if sft_guard_migration:
            causal_best_dir = bootstrap_root / "best"
            if not (causal_best_dir / "model.pt").is_file():
                raise RuntimeError(
                    "cannot migrate legacy SFT state without causal-best checkpoint"
                )
            runtime = GeneralistRuntime.from_checkpoint(
                causal_best_dir,
                device="cpu",
            )
            progress["sft_guard_migrated_from_causal_best"] = True
        replay, replay_counts = _mixed_replay_rows(
            root,
            bundle.sft_train,
            repo,
            heldout_sft=bundle.sft_validation,
        )
        progress["curriculum_replay_rows"] = replay_counts
        runtime, sft_report = _run_guarded_sft(
            runtime,
            replay,
            bundle.sft_validation,
            bootstrap_root=bootstrap_root,
            base_model_sha=base_model_sha,
            target_tokens=target_tokens,
            seed=51 + len(progress.get("completed_rungs") or []),
        )
        progress["sft_report"] = sft_report
        completed_sft = list(progress.get("sft_completed_rungs") or [])
        completed_sft.append(target_tokens)
        progress["sft_completed_rungs"] = sorted(set(int(x) for x in completed_sft))
        if bool(sft_report.get("accepted")):
            accepted_sft = list(progress.get("sft_accepted_rungs") or [])
            accepted_sft.append(target_tokens)
            progress["sft_accepted_rungs"] = sorted(set(int(x) for x in accepted_sft))
        runtime.save_checkpoint(
            candidate_dir,
            metadata={
                "role": "phase5_language_bootstrap_candidate",
                "production_qualified": False,
                "base_champion_model_sha256": base_model_sha,
                "tokens_processed": int(progress["tokens_processed"]),
                "target_tokens": target_tokens,
                "sft_replay": bool(sft_report.get("accepted")),
                "sft_guarded": True,
                "sft_rolled_back": bool(sft_report.get("rolled_back")),
            },
        )
        # Only accepted SFT changes invalidate the causal Adam moments. If every
        # SFT attempt is rejected, the restored pre-SFT checkpoint still matches
        # the persisted causal optimizer exactly.
        progress["sft_guard_version"] = PHASE5_SFT_GUARD_VERSION
        progress["sft_guard_migration_applied"] = bool(sft_guard_migration)
        if bool(sft_report.get("accepted")) or sft_guard_migration:
            optimizer_path.unlink(missing_ok=True)
            progress["optimizer_reset_after_sft"] = True
        else:
            progress["optimizer_reset_after_sft"] = False

    if rung_complete:
        after = evaluate_phase5_language(runtime)
        _atomic_json(bootstrap_root / "after.json", after)

        champion_report = _grouped_validation(
            champion_runtime.model,
            champion_runtime.tokenizer,
            validation_rows(),
            device="cpu",
        )
        champion_report["parameters"] = sum(
            int(p.numel()) for p in champion_runtime.model.parameters() if p.requires_grad
        )
        champion_report["score"] = _research_score(
            champion_report,
            champion_report["parameters"],
        )
        candidate_report = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            validation_rows(),
            device="cpu",
        )
        candidate_report["parameters"] = sum(
            int(p.numel()) for p in runtime.model.parameters() if p.requires_grad
        )
        candidate_report["score"] = _research_score(
            candidate_report,
            candidate_report["parameters"],
        )
        rotating = canary_rows(max(1, cycle))
        champion_report["canary_cycle"] = max(1, cycle)
        champion_report["canary"] = _grouped_validation(
            champion_runtime.model,
            champion_runtime.tokenizer,
            rotating,
            device="cpu",
        )
        candidate_report["canary_cycle"] = max(1, cycle)
        candidate_report["canary"] = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            rotating,
            device="cpu",
        )
        eligible, eligible_reason = _research_eligible(
            champion_report,
            candidate_report,
            minimum_loss_gain=0.01,
            max_domain_regression=0.08,
        )
        degeneration_ok, degeneration_reason = degeneration_gate(before, after)
        phase5_ok, phase5_reasons = _phase5_success(before, after)

        old_domains = champion_report.get("domain_nll_per_byte") or {}
        new_domains = candidate_report.get("domain_nll_per_byte") or {}
        regressions = [
            float(new_domains[name]) - float(old)
            for name, old in old_domains.items()
            if name in new_domains
        ]
        worst_domain_regression = max(regressions, default=0.0)
        candidate_report["worst_domain_regression"] = float(worst_domain_regression)
        candidate_report["research_gate_passed"] = bool(eligible)
        candidate_report["phase5_degeneration_gate_passed"] = bool(degeneration_ok)
        candidate_report["phase5_minimum_success"] = bool(phase5_ok)
        candidate_report["phase5_diagnostics"] = after

        latest_summary = _write_latest_research(
            root,
            runtime,
            report=candidate_report,
            cycle=cycle,
            base_model_sha256=base_model_sha,
        )

        promoted = bool(eligible and degeneration_ok and phase5_ok)
        promotion_reason = (
            "existing research gates + Phase 5 degeneration/language gates passed"
            if promoted
            else "; ".join(
                [
                    f"research={eligible_reason}" if not eligible else "",
                    f"degeneration={degeneration_reason}" if not degeneration_ok else "",
                    ("phase5=" + ", ".join(phase5_reasons)) if not phase5_ok else "",
                ]
            ).strip("; ")
        )
        if promoted:
            genome = _continual_candidate_genome(candidate_genome, cycle + 1)
            _save_champion(root, genome, runtime, candidate_report)
            status["champion"] = genome.to_dict()
            status["champion_report"] = candidate_report
            status["promoted"] = True
            status["promotion_reason"] = "phase5_bootstrap:" + promotion_reason

        status["latest_research"] = latest_summary
        status["phase5_bootstrap"] = {
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "unique_corpus_target_tokens": int(corpus_target_tokens),
            "optimization_passes_target": float(
                target_tokens / max(1, corpus_target_tokens)
            ),
            "capacity_target_parameters": progress.get("capacity_target_parameters"),
            "capacity_growth_history": progress.get("capacity_growth_history") or [],
            "tokens_processed": int(progress["tokens_processed"]),
            "steps": int(progress["steps"]),
            "before": before,
            "after": after,
            "research_gate_passed": bool(eligible),
            "research_gate_reason": eligible_reason,
            "degeneration_gate_passed": bool(degeneration_ok),
            "degeneration_gate_reason": degeneration_reason,
            "minimum_success": bool(phase5_ok),
            "minimum_success_reasons": phase5_reasons,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
            "manifest_sha256": bundle.manifest.get("manifest_content_sha256"),
            "anti_collapse": {
                "version": PHASE5_ANTICOLLAPSE_VERSION,
                "rescue": progress.get("anti_collapse_rescue"),
                "rescue_tokens": int(progress.get("anti_collapse_rescue_tokens", 0) or 0),
                "last_objective": progress.get("last_objective"),
                "decoding_modified": False,
            },
        }
        _atomic_json(root / "status.json", status)

        snapshot = snapshot_airi_pc_lab(repo)
        lab_probe = run_airi_pc_lab_probe(runtime, snapshot)
        phase5_lab = {
            "version": "airi-pc-lab-phase5-v1",
            "mode": snapshot.get("mode"),
            "candidate": lab_probe,
            "denied_capabilities": snapshot.get("denied_capabilities") or [],
            "capabilities": snapshot.get("capabilities") or [],
        }
        _atomic_json(bootstrap_root / "airi-pc-lab-after.json", phase5_lab)
        existing_lab = _load_json(root / "airi-pc-lab-report.json")
        existing_lab["phase5"] = phase5_lab
        _atomic_json(root / "airi-pc-lab-report.json", existing_lab)

        completed = list(progress.get("completed_rungs") or [])
        completed.append(target_tokens)
        progress["completed_rungs"] = sorted(set(int(x) for x in completed))
        report = {
            "schema": 1,
            "version": PHASE5_BOOTSTRAP_VERSION,
            "target_tokens": target_tokens,
            "unique_corpus_target_tokens": int(corpus_target_tokens),
            "optimization_passes_target": float(
                target_tokens / max(1, corpus_target_tokens)
            ),
            "capacity_target_parameters": progress.get("capacity_target_parameters"),
            "capacity_growth_history": progress.get("capacity_growth_history") or [],
            "tokens_processed": int(progress["tokens_processed"]),
            "steps": int(progress["steps"]),
            "model_parameters": int(candidate_report["parameters"]),
            "tokenizer_version": str(runtime.tokenizer.version),
            "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
            "before": before,
            "after": after,
            "validation": {
                "champion": champion_report,
                "candidate": candidate_report,
            },
            "research_gate_passed": bool(eligible),
            "research_gate_reason": eligible_reason,
            "degeneration_gate_passed": bool(degeneration_ok),
            "degeneration_gate_reason": degeneration_reason,
            "minimum_success": bool(phase5_ok),
            "minimum_success_reasons": phase5_reasons,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
            "airi_pc_lab_after": lab_probe,
            "manifest_sha256": bundle.manifest.get("manifest_content_sha256"),
            "anti_collapse": {
                "version": PHASE5_ANTICOLLAPSE_VERSION,
                "rescue": progress.get("anti_collapse_rescue"),
                "rescue_tokens": int(progress.get("anti_collapse_rescue_tokens", 0) or 0),
                "last_objective": progress.get("last_objective"),
                "decoding_modified": False,
            },
        }
        _atomic_json(bootstrap_root / "report.json", report)

    _atomic_json(progress_path, progress)
    lineage_manifest = refresh_live_lineage_manifest(
        root,
        reason=(
            "phase5_rung_complete"
            if rung_complete
            else "phase5_segment_checkpoint"
        ),
    )
    return {
        "ok": True,
        "lineage_id": lineage_manifest.get("lineage_id"),
        "active_lineage_checkpoint": lineage_manifest.get("active_checkpoint"),
        "version": PHASE5_BOOTSTRAP_VERSION,
        "target_tokens": target_tokens,
        "tokens_processed": int(progress["tokens_processed"]),
        "segment_tokens_processed": int(progress["tokens_processed"]) - processed_before_segment,
        "steps": int(progress["steps"]),
        "rung_complete": bool(rung_complete),
        "early_stopped": bool(early_stopped),
        "last_train_loss": last_train_loss,
        "last_validation_loss": last_validation_loss,
        "best_validation_loss": progress.get("best_validation_loss"),
        "report_path": str(bootstrap_root / "report.json") if rung_complete else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIRI Generalist Phase 5 language bootstrap")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--segment-tokens", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args(argv)
    result = run_segment(
        args.state_dir,
        args.repo_root,
        args.cache_dir,
        target_tokens=args.target_tokens,
        segment_tokens=args.segment_tokens,
        batch_size=args.batch_size,
        base_learning_rate=args.learning_rate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
