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
    _protected_optimizer,
    _segment_language_gate,
    _sft_training_batch,
)
from .phase5_ablation import (
    _cross_prompt_diversity,
    _group_gradient_stats,
    _group_update_stats,
    _set_optimizer_policy,
)
from .phase5_diagnostics import evaluate_phase5_language
from .pretraining import _batch as causal_batch, pack_causal_blocks
from .runtime import GeneralistRuntime
from .training import causal_training_objective, reference_kl_loss


def run_ablation(state_dir: str | Path) -> dict[str, Any]:
    import torch

    torch.manual_seed(17)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    state = Path(state_dir).expanduser().resolve()
    bootstrap = state / "bootstrap-data"
    candidate = bootstrap / "candidate"
    progress = json.loads((bootstrap / "progress.json").read_text(encoding="utf-8"))
    guard = json.loads((bootstrap / "language-guard.json").read_text(encoding="utf-8"))
    anchor = dict(guard.get("report") or {})
    if not anchor:
        raise RuntimeError("missing durable language anchor")

    baseline = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    before = evaluate_phase5_language(baseline)
    replay = load_bootstrap_replay(bootstrap)
    short_documents = [row for row in replay.documents if len(row.text) <= 220] or list(replay.documents)
    blocks = pack_causal_blocks(
        short_documents,
        baseline.tokenizer,
        context_length=baseline.config.context_length,
    )
    if len(blocks) < 128:
        raise RuntimeError("insufficient causal blocks for 8k transition ablation")

    protected_rows, protected_counts = _protected_causal_replay_rows(
        replay.sft_train,
        heldout_sft=(),
        elementary_only=True,
    )
    if len(protected_rows) < 2:
        raise RuntimeError("insufficient elementary replay rows")

    reference = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)

    base_scale = max(
        1.0 / 128.0,
        min(1.0, float(progress.get("segment_guard_lr_scale", 1.0 / 128.0) or 1.0 / 128.0)),
    )
    target_tokens = int(progress.get("target_tokens", 100_000_000) or 100_000_000)
    processed_tokens = int(progress.get("tokens_processed", 0) or 0)
    start_step = int(progress.get("steps", 0) or 0)
    ceiling = float(anchor["repetition_rate"]) + 0.08

    policies = [
        {
            "policy": "ACTUAL_8K_B32_S2_R025",
            "effective_batch_size": 32,
            "steps": 2,
            "replay_loss_weight": 0.25,
            "kl_weight": 0.50,
        },
        {
            "policy": "TRUST_8K_B32_S2_R4",
            "effective_batch_size": 32,
            "steps": 2,
            "replay_loss_weight": 4.0,
            "kl_weight": 0.50,
        },
        {
            "policy": "ONE_STEP_8K_B64_R4",
            "effective_batch_size": 64,
            "steps": 1,
            "replay_loss_weight": 4.0,
            "kl_weight": 0.50,
        },
        {
            "policy": "ONE_STEP_8K_B64_R1",
            "effective_batch_size": 64,
            "steps": 1,
            "replay_loss_weight": 1.0,
            "kl_weight": 0.50,
        },
    ]
    seeds = [0, 1, 2, 3]
    results: list[dict[str, Any]] = []

    for policy in policies:
        for seed_index in seeds:
            runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
            model = runtime.model
            model.train()
            optimizer, optimizer_report = _protected_optimizer(
                model,
                learning_rate=3e-4 * base_scale,
            )
            multipliers = {
                "embeddings": 0.25,
                "attention_and_norm": 0.35,
                "ffn": 1.0,
                "lm_head": 0.50,
            }
            _set_optimizer_policy(
                optimizer,
                learning_rate=3e-4 * base_scale,
                multipliers=multipliers,
            )

            supervised_total = 0
            replay_supervised_total = 0
            grad_rows: list[dict[str, Any]] = []
            effective_batch_size = int(policy["effective_batch_size"])
            local_steps = int(policy["steps"])
            seed_offset = int(seed_index) * 1_000_003

            for local_step in range(local_steps):
                step = start_step + local_step
                rng = random.Random(5_000_000 + step + seed_offset)
                indices = [rng.randrange(len(blocks)) for _ in range(effective_batch_size)]
                optimizer.zero_grad(set_to_none=True)

                lr = _learning_rate(
                    base_lr=3e-4 * base_scale,
                    processed_tokens=processed_tokens + supervised_total,
                    target_tokens=target_tokens,
                    warmup_tokens=min(100_000, max(20_000, target_tokens // 10)),
                )
                lr *= min(1.0, float(local_step + 1) / 4.0)
                _set_optimizer_policy(optimizer, learning_rate=lr, multipliers=multipliers)

                anti_weight, eos_weight = _anti_collapse_weights(
                    "B_short_sentence_completion",
                    before,
                )
                anti_weight = max(float(anti_weight), 0.02)

                prepared = []
                supervised = 0
                for offset in range(0, len(indices), 2):
                    ids, labels = causal_batch(
                        blocks,
                        indices[offset:offset + 2],
                        device=runtime.device,
                    )
                    count = int((labels[:, 1:] != -100).sum().item())
                    prepared.append((ids, labels, count))
                    supervised += count
                for ids, labels, count in prepared:
                    logits = model(ids)["logits"]
                    loss, _ = causal_training_objective(
                        logits,
                        labels,
                        ids,
                        eos_loss_weight=eos_weight,
                        repetition_unlikelihood_weight=anti_weight,
                        repetition_window=16,
                    )
                    (loss * (count / max(1, supervised))).backward()
                supervised_total += supervised

                replay_rng = random.Random(9_000_000 + step * 97 + seed_offset)
                replay_indices = [
                    replay_rng.randrange(len(protected_rows))
                    for _ in range(2)
                ]
                replay_ids, replay_labels = _sft_training_batch(
                    runtime,
                    protected_rows,
                    replay_indices,
                )
                replay_count = int((replay_labels[:, 1:] != -100).sum().item())
                replay_supervised_total += replay_count
                replay_logits = model(replay_ids)["logits"]
                replay_loss, _ = causal_training_objective(
                    replay_logits,
                    replay_labels,
                    replay_ids,
                    eos_loss_weight=1.0,
                    repetition_unlikelihood_weight=anti_weight,
                    repetition_window=16,
                )
                with torch.no_grad():
                    ref_logits = reference.model(replay_ids)["logits"]
                kl_loss, _ = reference_kl_loss(
                    replay_logits,
                    ref_logits,
                    replay_labels,
                    temperature=1.0,
                )
                (
                    float(policy["replay_loss_weight"]) * replay_loss
                    + float(policy["kl_weight"]) * kl_loss
                ).backward()

                gradient_before_clip = _group_gradient_stats(model)
                global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                grad_rows.append({
                    "step": local_step,
                    "effective_lr": float(lr),
                    "global_gradient_norm_before_clip": float(global_norm.detach().cpu()),
                    "gradient_before_clip": gradient_before_clip,
                })

            after = evaluate_phase5_language(runtime)
            accepted, gate = _segment_language_gate(
                before,
                after,
                anchor,
                attempted_tokens=supervised_total,
            )
            before_diversity = _cross_prompt_diversity(before)
            after_diversity = _cross_prompt_diversity(after)
            results.append({
                "name": f"{policy['policy']}_SEED_{seed_index}",
                "policy": policy["policy"],
                "seed": seed_index,
                "accepted_by_unchanged_guard": bool(accepted),
                "effective_batch_size": effective_batch_size,
                "optimizer_steps": local_steps,
                "attempted_tokens": int(supervised_total),
                "replay_supervised_tokens": int(replay_supervised_total),
                "replay_loss_weight": float(policy["replay_loss_weight"]),
                "reference_kl_weight": float(policy["kl_weight"]),
                "before": {
                    "language_nll": before["language_nll"],
                    "repetition_rate": before["repetition_rate"],
                },
                "after": {
                    "language_nll": after["language_nll"],
                    "repetition_rate": after["repetition_rate"],
                },
                "delta": {
                    "language_nll": float(after["language_nll"] - before["language_nll"]),
                    "repetition_rate": float(after["repetition_rate"] - before["repetition_rate"]),
                },
                "headroom": {
                    "before": float(ceiling - before["repetition_rate"]),
                    "after": float(ceiling - after["repetition_rate"]),
                },
                "cross_prompt_diversity": {
                    "before": before_diversity,
                    "after": after_diversity,
                },
                "gate": {
                    "local_reasons": gate["local_reasons"],
                    "after_anchor_violations": gate["after_anchor_violations"],
                },
                "gradient_steps": grad_rows,
                "parameter_updates": _group_update_stats(model, reference.model),
                "optimizer_report": optimizer_report,
            })
            del runtime, optimizer

    policy_summaries = []
    for policy in policies:
        name = str(policy["policy"])
        rows = [row for row in results if row["policy"] == name]
        rep_deltas = [float(row["delta"]["repetition_rate"]) for row in rows]
        nll_deltas = [float(row["delta"]["language_nll"]) for row in rows]
        passes = sum(int(row["accepted_by_unchanged_guard"]) for row in rows)
        policy_summaries.append({
            "policy": name,
            "trials": len(rows),
            "guard_passes": passes,
            "guard_pass_rate": passes / max(1, len(rows)),
            "mean_repetition_delta": sum(rep_deltas) / max(1, len(rep_deltas)),
            "worst_repetition_delta": max(rep_deltas) if rep_deltas else 0.0,
            "mean_nll_delta": sum(nll_deltas) / max(1, len(nll_deltas)),
            "worst_nll_delta": max(nll_deltas) if nll_deltas else 0.0,
        })

    return {
        "schema": 1,
        "version": "phase5-8k-transition-ablation-v1",
        "read_only": True,
        "lineage_id": str(progress.get("lineage_id") or ""),
        "tokens_processed": processed_tokens,
        "parameters": int(sum(p.numel() for p in reference.model.parameters())),
        "baseline_headroom": float(ceiling - before["repetition_rate"]),
        "protected_replay_rows": protected_counts,
        "policy_summaries": policy_summaries,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Phase-5 8k transition ablation")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_ablation(args.state_dir)
    target = Path(args.output)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "lineage_id": report["lineage_id"],
        "tokens_processed": report["tokens_processed"],
        "baseline_headroom": report["baseline_headroom"],
        "policy_summaries": report["policy_summaries"],
        "variants": [
            {
                "name": row["name"],
                "accepted": row["accepted_by_unchanged_guard"],
                "attempted_tokens": row["attempted_tokens"],
                "optimizer_steps": row["optimizer_steps"],
                "nll_delta": row["delta"]["language_nll"],
                "repetition_delta": row["delta"]["repetition_rate"],
                "headroom_after": row["headroom"]["after"],
                "reasons": row["gate"]["local_reasons"] + row["gate"]["after_anchor_violations"],
            }
            for row in report["results"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
