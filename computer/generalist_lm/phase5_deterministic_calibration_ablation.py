from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .bootstrap_data import load_bootstrap_replay
from .bootstrap_training import (
    _elementary_rehabilitation_rows,
    _filter_protected_replay,
    _segment_language_gate,
    _sft_row_fingerprint,
)
from .phase5_diagnostics import evaluate_phase5_language
from .runtime import GeneralistRuntime
from .training import _batch, causal_training_objective


def _diversity(report: dict[str, Any]) -> dict[str, float | int]:
    outputs = [
        re.sub(r"\s+", " ", str(row.get("raw_output") or row.get("output") or ""))
        .strip()
        .lower()
        for row in (report.get("traces") or [])
    ]
    counts: dict[str, int] = {}
    for output in outputs:
        counts[output] = counts.get(output, 0) + 1
    pairs = 0
    jaccard = 0.0
    sequence = 0.0
    for left in range(len(outputs)):
        for right in range(left + 1, len(outputs)):
            pairs += 1
            sequence += SequenceMatcher(None, outputs[left], outputs[right]).ratio()
            a = set(outputs[left].split())
            b = set(outputs[right].split())
            union = a | b
            jaccard += len(a & b) / len(union) if union else 1.0
    unique = len(set(outputs))
    dominant = max(counts.values(), default=0)
    return {
        "prompt_count": len(outputs),
        "unique_generation_count": unique,
        "exact_duplicate_rate": 1.0 - unique / len(outputs) if outputs else 0.0,
        "dominant_generation_fraction": dominant / len(outputs) if outputs else 0.0,
        "mean_pairwise_token_jaccard": jaccard / pairs if pairs else 0.0,
        "mean_pairwise_sequence_similarity": sequence / pairs if pairs else 0.0,
    }


def _r2_rows() -> list[Any]:
    curriculum = _elementary_rehabilitation_rows()
    rows: list[Any] = []
    for language in ("it", "en"):
        rows.extend(curriculum["R2_simple_responses"][language])
    return sorted(rows, key=_sft_row_fingerprint)


def _deterministic_residual_calibration(
    runtime,
    reference,
    tokenizer,
    examples,
    anchors,
    *,
    source_d_ff: int,
    steps: int,
    learning_rate: float,
    kl_weight: float,
    anchor_offset: int,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    model = runtime.model
    teacher = reference.model
    device_obj = torch.device(device)
    model.to(device_obj)
    teacher.to(device_obj)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_d_ff = int(model.config.d_ff)
    masks: list[tuple[Any, Any]] = []
    trainable = []
    coordinates = 0
    for block in model.blocks:
        down = block.ff.down
        down.weight.requires_grad_(True)
        mask = torch.zeros_like(down.weight, device=device_obj)
        mask[:, int(source_d_ff):target_d_ff] = 1
        masks.append((down.weight, mask))
        trainable.append(down.weight)
        coordinates += int(mask.sum().item())

    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=0.0)
    example_indices = list(range(len(examples)))
    anchor_batch = min(len(anchors), max(12, len(examples)))
    losses: list[float] = []
    kl_losses: list[float] = []
    gradient_norms: list[float] = []

    for step in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        ids, labels = _batch(
            examples,
            tokenizer,
            model.config.context_length,
            example_indices,
        )
        ids, labels = ids.to(device_obj), labels.to(device_obj)
        logits = model(ids)["logits"]
        supervised_loss, _ = causal_training_objective(
            logits,
            labels,
            ids,
            eos_loss_weight=1.10,
            repetition_unlikelihood_weight=0.02,
            repetition_window=16,
        )

        start = (int(anchor_offset) + step * anchor_batch) % len(anchors)
        anchor_indices = [
            (start + offset) % len(anchors)
            for offset in range(anchor_batch)
        ]
        anchor_ids, anchor_labels = _batch(
            anchors,
            tokenizer,
            model.config.context_length,
            anchor_indices,
        )
        anchor_ids = anchor_ids.to(device_obj)
        anchor_labels = anchor_labels.to(device_obj)
        student_logits = model(anchor_ids)["logits"][:, :-1, :].float()
        with torch.no_grad():
            teacher_logits = teacher(anchor_ids)["logits"][:, :-1, :].float()
        valid = anchor_labels[:, 1:] != -100
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        token_kl = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
        kl_loss = (
            token_kl.masked_select(valid).mean().clamp_min(0.0)
            if int(valid.sum().item())
            else token_kl.mean() * 0.0
        )
        total = supervised_loss + float(kl_weight) * kl_loss
        total.backward()
        for parameter, mask in masks:
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        losses.append(float(total.detach().cpu()))
        kl_losses.append(float(kl_loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))

    return {
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "kl_weight": float(kl_weight),
        "anchor_offset": int(anchor_offset),
        "r2_rows_per_step": len(examples),
        "anchor_rows_per_step": anchor_batch,
        "trainable_coordinate_count": coordinates,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "mean_kl_loss": sum(kl_losses) / max(1, len(kl_losses)),
        "mean_gradient_norm": sum(gradient_norms) / max(1, len(gradient_norms)),
        "max_gradient_norm": max(gradient_norms, default=0.0),
    }


def run_ablation(
    state_dir: str | Path,
    *,
    policy_filter: str | None = None,
) -> dict[str, Any]:
    state = Path(state_dir).expanduser().resolve()
    bootstrap = state / "bootstrap-data"
    candidate = bootstrap / "candidate"
    progress = json.loads((bootstrap / "progress.json").read_text(encoding="utf-8"))
    guard = json.loads((bootstrap / "language-guard.json").read_text(encoding="utf-8"))
    anchor = dict(guard.get("report") or {})
    revival = dict(progress.get("dead_capacity_revival") or {})
    source_d_ff = int(revival.get("source_d_ff", 0) or 0)
    if not bool(revival.get("completed")) or source_d_ff <= 0:
        raise RuntimeError("missing revived residual source")

    baseline = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    before = evaluate_phase5_language(baseline)
    before_diversity = _diversity(before)
    ceiling = float(anchor["repetition_rate"]) + 0.08
    headroom_before = float(ceiling - before["repetition_rate"])

    replay = load_bootstrap_replay(bootstrap)
    elementary_fps = {
        _sft_row_fingerprint(row)
        for stage in _elementary_rehabilitation_rows().values()
        for language in ("it", "en")
        for row in stage[language]
    }
    safe_replay, _ = _filter_protected_replay(list(replay.sft_train), heldout_sft=())
    anchors = sorted(
        [
            row for row in safe_replay
            if _sft_row_fingerprint(row) not in elementary_fps
        ],
        key=_sft_row_fingerprint,
    )
    examples = _r2_rows()
    if not anchors or len(examples) != 12:
        raise RuntimeError("invalid deterministic calibration pools")

    policies = [
        {"policy": "DET_R2_S8_LR625E7_KL5", "steps": 8, "learning_rate": 6.25e-7, "kl_weight": 5.0},
        {"policy": "DET_R2_S12_LR625E7_KL5", "steps": 12, "learning_rate": 6.25e-7, "kl_weight": 5.0},
        {"policy": "DET_R2_S8_LR1E6_KL5", "steps": 8, "learning_rate": 1.0e-6, "kl_weight": 5.0},
        {"policy": "DET_R2_S8_LR625E7_KL10", "steps": 8, "learning_rate": 6.25e-7, "kl_weight": 10.0},
    ]
    if policy_filter:
        policies = [p for p in policies if p["policy"] == policy_filter]
        if not policies:
            raise ValueError(f"unknown policy: {policy_filter}")
    offsets = [0, 17, 37, 71]
    results: list[dict[str, Any]] = []

    for policy in policies:
        for offset in offsets:
            runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
            reference = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
            training = _deterministic_residual_calibration(
                runtime,
                reference,
                runtime.tokenizer,
                examples,
                anchors,
                source_d_ff=source_d_ff,
                steps=int(policy["steps"]),
                learning_rate=float(policy["learning_rate"]),
                kl_weight=float(policy["kl_weight"]),
                anchor_offset=offset,
            )
            after = evaluate_phase5_language(runtime)
            after_diversity = _diversity(after)
            gate_ok, gate = _segment_language_gate(
                before,
                after,
                anchor,
                attempted_tokens=0,
            )
            headroom_after = float(ceiling - after["repetition_rate"])
            nll_delta = float(after["language_nll"] - before["language_nll"])
            duplicate_delta = float(
                after_diversity["exact_duplicate_rate"]
                - before_diversity["exact_duplicate_rate"]
            )
            calibration_ok = bool(
                gate_ok
                and headroom_after > headroom_before
                and nll_delta <= 0.015
                and duplicate_delta <= 1e-12
            )
            results.append({
                "policy": policy["policy"],
                "anchor_offset": offset,
                "calibration_ok": calibration_ok,
                "guard_ok": bool(gate_ok),
                "training": training,
                "nll_delta": nll_delta,
                "repetition_delta": float(after["repetition_rate"] - before["repetition_rate"]),
                "headroom_before": headroom_before,
                "headroom_after": headroom_after,
                "headroom_gain": float(headroom_after - headroom_before),
                "diversity_before": before_diversity,
                "diversity_after": after_diversity,
                "reasons": gate["local_reasons"] + gate["after_anchor_violations"],
            })
            del runtime, reference

    summaries: list[dict[str, Any]] = []
    for policy in policies:
        name = policy["policy"]
        rows = [row for row in results if row["policy"] == name]
        gains = [row["headroom_gain"] for row in rows]
        nlls = [row["nll_delta"] for row in rows]
        passes = sum(int(row["calibration_ok"]) for row in rows)
        summaries.append({
            "policy": name,
            "trials": len(rows),
            "calibration_passes": passes,
            "calibration_pass_rate": passes / max(1, len(rows)),
            "mean_headroom_gain": sum(gains) / max(1, len(gains)),
            "worst_headroom_gain": min(gains, default=0.0),
            "mean_nll_delta": sum(nlls) / max(1, len(nlls)),
            "worst_nll_delta": max(nlls, default=0.0),
            "stable_candidate": bool(
                len(rows) == len(offsets)
                and passes == len(rows)
                and min(gains, default=-1.0) > 0.0
                and max(nlls, default=1.0) <= 0.015
            ),
        })

    return {
        "schema": 1,
        "version": "phase5-deterministic-residual-calibration-v1",
        "read_only": True,
        "lineage_id": str(progress.get("lineage_id") or ""),
        "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
        "source_d_ff": source_d_ff,
        "target_d_ff": int(baseline.config.d_ff),
        "baseline_headroom": headroom_before,
        "policy_summaries": summaries,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()
    report = run_ablation(args.state_dir, policy_filter=args.policy)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "lineage_id": report["lineage_id"],
        "tokens_processed": report["tokens_processed"],
        "baseline_headroom": report["baseline_headroom"],
        "policy_summaries": report["policy_summaries"],
        "variants": [
            {
                "policy": row["policy"],
                "offset": row["anchor_offset"],
                "calibration_ok": row["calibration_ok"],
                "guard_ok": row["guard_ok"],
                "nll_delta": row["nll_delta"],
                "repetition_delta": row["repetition_delta"],
                "headroom_after": row["headroom_after"],
                "headroom_gain": row["headroom_gain"],
                "duplicate_rate_after": row["diversity_after"]["exact_duplicate_rate"],
                "reasons": row["reasons"],
            }
            for row in report["results"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
