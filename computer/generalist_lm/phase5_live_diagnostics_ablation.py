from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .bootstrap_data import load_bootstrap_replay
from .bootstrap_training import (
    _anti_collapse_weights,
    _learning_rate,
    _protected_causal_replay_rows,
    _protected_continual_plan,
    _protected_optimizer,
    _segment_language_gate,
    _sft_training_batch,
)
from .phase5_diagnostics import evaluate_phase5_language
from .pretraining import _batch as causal_batch, pack_causal_blocks
from .runtime import GeneralistRuntime
from .training import causal_training_objective, reference_kl_loss


def run_ablation(
    state_dir: str | Path,
    *,
    diagnostics_mode: str,
    seed_index: int,
) -> dict[str, Any]:
    import torch

    state = Path(state_dir).expanduser().resolve()
    bootstrap = state / "bootstrap-data"
    candidate = bootstrap / "candidate"
    progress = json.loads((bootstrap / "progress.json").read_text(encoding="utf-8"))
    guard = json.loads((bootstrap / "language-guard.json").read_text(encoding="utf-8"))
    stale_before = json.loads((bootstrap / "before.json").read_text(encoding="utf-8"))
    anchor = dict(guard.get("report") or {})
    if not anchor:
        raise RuntimeError("missing durable language anchor")

    runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    reference = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)

    live_before = evaluate_phase5_language(runtime)
    replay = load_bootstrap_replay(bootstrap)
    short_documents = [row for row in replay.documents if len(row.text) <= 220] or list(replay.documents)
    blocks = pack_causal_blocks(
        short_documents,
        runtime.tokenizer,
        context_length=runtime.config.context_length,
    )
    if len(blocks) < 32:
        raise RuntimeError("insufficient causal blocks")

    protected_plan = _protected_continual_plan(
        live_before,
        anchor,
        consecutive_rejections=int(progress.get("segment_guard_consecutive_rejections", 0) or 0),
        success_streak=int(progress.get("segment_guard_success_streak", 0) or 0),
        trust_region_recovery=True,
    )
    protected_rows, row_counts = _protected_causal_replay_rows(
        replay.sft_train,
        heldout_sft=(),
        elementary_only=True,
    )
    if len(protected_rows) < 2:
        raise RuntimeError("insufficient elementary protected rows")

    mode = str(diagnostics_mode).strip().lower()
    if mode == "stale":
        objective_diagnostics = stale_before
    elif mode == "live":
        objective_diagnostics = live_before
    else:
        raise ValueError("diagnostics_mode must be stale or live")

    anti_weight, eos_weight = _anti_collapse_weights(
        "B_short_sentence_completion",
        objective_diagnostics,
    )
    anti_weight = max(
        float(anti_weight),
        float(protected_plan.get("anti_repetition_weight", 0.0) or 0.0),
    )

    base_scale = max(
        1.0 / 128.0,
        min(1.0, float(progress.get("segment_guard_lr_scale", 1.0 / 128.0) or 1.0 / 128.0)),
    )
    optimizer, optimizer_report = _protected_optimizer(
        runtime.model,
        learning_rate=3e-4 * base_scale,
    )

    step = int(progress.get("steps", 0) or 0)
    seed_offset = int(seed_index) * 1_000_003
    rng = random.Random(5_000_000 + step + seed_offset)
    indices = [rng.randrange(len(blocks)) for _ in range(32)]

    target_tokens = int(progress.get("target_tokens", 100_000_000) or 100_000_000)
    processed_tokens = int(progress.get("tokens_processed", 0) or 0)
    lr = _learning_rate(
        base_lr=3e-4 * base_scale,
        processed_tokens=processed_tokens,
        target_tokens=target_tokens,
        warmup_tokens=min(100_000, max(20_000, target_tokens // 10)),
    )
    lr *= 0.25
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_multiplier", 1.0) or 1.0)

    optimizer.zero_grad(set_to_none=True)
    prepared = []
    supervised = 0
    for offset in range(0, len(indices), 2):
        ids, labels = causal_batch(blocks, indices[offset:offset + 2], device=runtime.device)
        count = int((labels[:, 1:] != -100).sum().item())
        prepared.append((ids, labels, count))
        supervised += count
    if supervised <= 0:
        raise RuntimeError("causal proposal has no supervised tokens")

    causal_loss_weighted = 0.0
    objective_tail: dict[str, Any] = {}
    for ids, labels, count in prepared:
        result = runtime.model(ids)
        loss, objective = causal_training_objective(
            result["logits"],
            labels,
            ids,
            eos_loss_weight=float(eos_weight),
            repetition_unlikelihood_weight=float(anti_weight),
            repetition_window=16,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite causal loss")
        (loss * (count / supervised)).backward()
        causal_loss_weighted += float(loss.detach().cpu()) * count
        objective_tail = objective

    replay_rng = random.Random(9_000_000 + step * 97 + seed_offset)
    replay_indices = [replay_rng.randrange(len(protected_rows)) for _ in range(2)]
    replay_ids, replay_labels = _sft_training_batch(runtime, protected_rows, replay_indices)
    replay_supervised = int((replay_labels[:, 1:] != -100).sum().item())
    replay_result = runtime.model(replay_ids)
    replay_loss, _ = causal_training_objective(
        replay_result["logits"],
        replay_labels,
        replay_ids,
        eos_loss_weight=1.0,
        repetition_unlikelihood_weight=float(anti_weight),
        repetition_window=16,
    )
    with torch.no_grad():
        reference_logits = reference.model(replay_ids)["logits"]
    kl_loss, kl_tokens = reference_kl_loss(
        replay_result["logits"],
        reference_logits,
        replay_labels,
        temperature=float(protected_plan.get("reference_kl_temperature", 1.0) or 1.0),
    )
    protected_loss = (
        float(protected_plan["replay_loss_weight"]) * replay_loss
        + float(protected_plan["reference_kl_weight"]) * kl_loss
    )
    protected_loss.backward()

    global_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0)
    optimizer.step()
    after = evaluate_phase5_language(runtime)
    accepted, gate = _segment_language_gate(
        live_before,
        after,
        anchor,
        attempted_tokens=supervised,
    )

    return {
        "schema": 1,
        "version": "phase5-live-diagnostics-objective-ablation-v1",
        "read_only": True,
        "diagnostics_mode": mode,
        "seed_index": int(seed_index),
        "lineage_id": str(progress.get("lineage_id") or ""),
        "tokens_processed": processed_tokens,
        "attempted_tokens": int(supervised),
        "actual_objective": {
            "repetition_unlikelihood_weight": float(anti_weight),
            "eos_loss_weight": float(eos_weight),
            "protected_replay_loss_weight": float(protected_plan["replay_loss_weight"]),
            "reference_kl_weight": float(protected_plan["reference_kl_weight"]),
            "effective_learning_rate": float(lr),
        },
        "diagnostics_source": {
            "suite": str(objective_diagnostics.get("suite") or ""),
            "prompt_count": int(objective_diagnostics.get("prompt_count", 0) or 0),
            "repetition_rate": float(objective_diagnostics.get("repetition_rate", 0.0) or 0.0),
            "pathological_repetition": bool(objective_diagnostics.get("pathological_repetition")),
        },
        "live_before": {
            "language_nll": float(live_before["language_nll"]),
            "repetition_rate": float(live_before["repetition_rate"]),
            "pathological_repetition": bool(live_before["pathological_repetition"]),
        },
        "after": {
            "language_nll": float(after["language_nll"]),
            "repetition_rate": float(after["repetition_rate"]),
            "pathological_repetition": bool(after["pathological_repetition"]),
        },
        "delta": {
            "language_nll": float(after["language_nll"] - live_before["language_nll"]),
            "repetition_rate": float(after["repetition_rate"] - live_before["repetition_rate"]),
        },
        "accepted_by_unchanged_guard": bool(accepted),
        "gate_reasons": list(gate["local_reasons"]) + list(gate["after_anchor_violations"]),
        "causal_loss": float(causal_loss_weighted / max(1, supervised)),
        "protected_replay_loss": float(replay_loss.detach().cpu()),
        "reference_kl_loss": float(kl_loss.detach().cpu()),
        "reference_kl_tokens": int(kl_tokens),
        "replay_supervised_tokens": int(replay_supervised),
        "global_gradient_norm_before_clip": float(global_norm.detach().cpu()),
        "protected_rows": row_counts,
        "optimizer": optimizer_report,
        "objective_tail": objective_tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only stale-vs-live diagnostics objective ablation")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--mode", choices=("stale", "live"), required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_ablation(args.state_dir, diagnostics_mode=args.mode, seed_index=args.seed_index)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
