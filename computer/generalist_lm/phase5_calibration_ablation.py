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
    _language_guard_violations,
    _segment_language_gate,
    _sft_row_fingerprint,
)
from .phase5_diagnostics import evaluate_phase5_language
from .runtime import GeneralistRuntime
from .training import train_sft_residual_recovery


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
    sequence_sum = 0.0
    jaccard_sum = 0.0
    pairs = 0
    for left in range(len(outputs)):
        for right in range(left + 1, len(outputs)):
            pairs += 1
            sequence_sum += SequenceMatcher(
                None, outputs[left], outputs[right]
            ).ratio()
            a = set(outputs[left].split())
            b = set(outputs[right].split())
            union = a | b
            jaccard_sum += len(a & b) / len(union) if union else 1.0
    unique = len(set(outputs))
    dominant = max(counts.values(), default=0)
    return {
        "prompt_count": len(outputs),
        "unique_generation_count": unique,
        "exact_duplicate_rate": (
            1.0 - unique / len(outputs) if outputs else 0.0
        ),
        "dominant_generation_fraction": (
            dominant / len(outputs) if outputs else 0.0
        ),
        "mean_pairwise_sequence_similarity": (
            sequence_sum / pairs if pairs else 0.0
        ),
        "mean_pairwise_token_jaccard": (
            jaccard_sum / pairs if pairs else 0.0
        ),
    }


def _elementary_rows() -> tuple[list[Any], list[Any]]:
    curriculum = _elementary_rehabilitation_rows()
    r2: list[Any] = []
    all_rows: list[Any] = []
    for stage in sorted(curriculum):
        for language in ("it", "en"):
            rows = list(curriculum[stage][language])
            all_rows.extend(rows)
            if stage == "R2_simple_responses":
                r2.extend(rows)
    return r2, all_rows


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
    if not anchor:
        raise RuntimeError("missing durable language anchor")

    revival = dict(progress.get("dead_capacity_revival") or {})
    source_d_ff = int(revival.get("source_d_ff", 0) or 0)
    if not bool(revival.get("completed")) or source_d_ff <= 0:
        raise RuntimeError("live checkpoint has no verified revived residual source")

    baseline = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    before = evaluate_phase5_language(baseline)
    before_diversity = _diversity(before)
    ceiling = float(anchor["repetition_rate"]) + 0.08

    replay = load_bootstrap_replay(bootstrap)
    r2_rows, all_elementary = _elementary_rows()
    elementary_fps = {
        _sft_row_fingerprint(row) for row in all_elementary
    }
    safe_replay, filtered = _filter_protected_replay(
        list(replay.sft_train),
        heldout_sft=(),
    )
    anchor_rows = [
        row for row in safe_replay
        if _sft_row_fingerprint(row) not in elementary_fps
    ]
    if not anchor_rows:
        raise RuntimeError("calibration ablation has no protected KL anchors")

    # Pre-registered variance-reduction matrix.  These policies were fixed
    # before looking at their validation outcomes.  Larger batches reduce
    # stochastic coverage variance while stronger teacher KL constrains the
    # revived residual branch against language drift.
    policies = [
        {
            "policy": "R2_B12_S16_LR1E6_KL5",
            "rows": r2_rows,
            "steps": 16,
            "batch_size": 12,
            "learning_rate": 1.0e-6,
            "kl_weight": 5.0,
        },
        {
            "policy": "R2_B24_S12_LR1E6_KL5",
            "rows": r2_rows,
            "steps": 12,
            "batch_size": 24,
            "learning_rate": 1.0e-6,
            "kl_weight": 5.0,
        },
        {
            "policy": "R2_B12_S16_LR625E7_KL5",
            "rows": r2_rows,
            "steps": 16,
            "batch_size": 12,
            "learning_rate": 6.25e-7,
            "kl_weight": 5.0,
        },
        {
            "policy": "R2_B24_S8_LR125E6_KL5",
            "rows": r2_rows,
            "steps": 8,
            "batch_size": 24,
            "learning_rate": 1.25e-6,
            "kl_weight": 5.0,
        },
    ]
    seeds = [0, 1, 2, 3]
    if policy_filter:
        policies = [
            policy for policy in policies
            if str(policy["policy"]) == str(policy_filter)
        ]
        if not policies:
            raise ValueError(f"unknown calibration policy: {policy_filter}")
    results: list[dict[str, Any]] = []

    for policy in policies:
        for seed in seeds:
            runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
            reference = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
            training = train_sft_residual_recovery(
                runtime.model,
                reference.model,
                runtime.tokenizer,
                list(policy["rows"]),
                anchor_examples=anchor_rows,
                source_d_ff=source_d_ff,
                steps=int(policy["steps"]),
                batch_size=int(policy["batch_size"]),
                learning_rate=float(policy["learning_rate"]),
                seed=70_000 + int(seed),
                device="cpu",
                repetition_unlikelihood_weight=0.02,
                eos_loss_weight=1.10,
                repetition_window=16,
                kl_weight=float(policy["kl_weight"]),
                train_upstream=False,
            )
            after = evaluate_phase5_language(runtime)
            after_diversity = _diversity(after)
            gate_ok, gate = _segment_language_gate(
                before,
                after,
                anchor,
                attempted_tokens=0,
            )
            repetition_delta = float(
                after["repetition_rate"] - before["repetition_rate"]
            )
            headroom_before = float(ceiling - before["repetition_rate"])
            headroom_after = float(ceiling - after["repetition_rate"])
            calibration_ok = bool(
                gate_ok
                and headroom_after > headroom_before
                and after_diversity["exact_duplicate_rate"]
                <= before_diversity["exact_duplicate_rate"] + 1e-12
                and after_diversity["mean_pairwise_token_jaccard"]
                <= before_diversity["mean_pairwise_token_jaccard"] + 0.03
            )
            results.append({
                "name": f"{policy['policy']}_SEED_{seed}",
                "policy": policy["policy"],
                "seed": seed,
                "steps": int(policy["steps"]),
                "batch_size": int(policy["batch_size"]),
                "learning_rate": float(policy["learning_rate"]),
                "kl_weight": float(policy["kl_weight"]),
                "calibration_ok": calibration_ok,
                "passes_unchanged_language_guard": bool(gate_ok),
                "training": training,
                "before": {
                    "language_nll": before["language_nll"],
                    "repetition_rate": before["repetition_rate"],
                    "token_entropy": before["token_entropy"],
                    "generation_similarity": before["generation_similarity"],
                    "multiword_output_rate": before["multiword_output_rate"],
                },
                "after": {
                    "language_nll": after["language_nll"],
                    "repetition_rate": after["repetition_rate"],
                    "token_entropy": after["token_entropy"],
                    "generation_similarity": after["generation_similarity"],
                    "multiword_output_rate": after["multiword_output_rate"],
                },
                "delta": {
                    "language_nll": float(
                        after["language_nll"] - before["language_nll"]
                    ),
                    "repetition_rate": repetition_delta,
                    "token_entropy": float(
                        after["token_entropy"] - before["token_entropy"]
                    ),
                    "generation_similarity": float(
                        after["generation_similarity"]
                        - before["generation_similarity"]
                    ),
                },
                "headroom": {
                    "before": headroom_before,
                    "after": headroom_after,
                    "gain": float(headroom_after - headroom_before),
                },
                "cross_prompt_diversity": {
                    "before": before_diversity,
                    "after": after_diversity,
                },
                "gate": {
                    "local_reasons": gate["local_reasons"],
                    "after_anchor_violations": gate["after_anchor_violations"],
                    "before_quality": gate["before_quality"],
                    "after_quality": gate["after_quality"],
                },
            })
            del runtime, reference

    policy_summaries: list[dict[str, Any]] = []
    for policy in policies:
        name = str(policy["policy"])
        rows = [row for row in results if row["policy"] == name]
        headroom_gains = [float(row["headroom"]["gain"]) for row in rows]
        nll_deltas = [float(row["delta"]["language_nll"]) for row in rows]
        guard_passes = sum(int(row["passes_unchanged_language_guard"]) for row in rows)
        calibration_passes = sum(int(row["calibration_ok"]) for row in rows)
        duplicate_deltas = [
            float(
                row["cross_prompt_diversity"]["after"]["exact_duplicate_rate"]
                - row["cross_prompt_diversity"]["before"]["exact_duplicate_rate"]
            )
            for row in rows
        ]
        policy_summaries.append({
            "policy": name,
            "trials": len(rows),
            "guard_passes": guard_passes,
            "calibration_passes": calibration_passes,
            "guard_pass_rate": guard_passes / max(1, len(rows)),
            "calibration_pass_rate": calibration_passes / max(1, len(rows)),
            "mean_headroom_gain": sum(headroom_gains) / max(1, len(headroom_gains)),
            "worst_headroom_gain": min(headroom_gains) if headroom_gains else 0.0,
            "best_headroom_gain": max(headroom_gains) if headroom_gains else 0.0,
            "mean_nll_delta": sum(nll_deltas) / max(1, len(nll_deltas)),
            "worst_nll_delta": max(nll_deltas) if nll_deltas else 0.0,
            "worst_duplicate_rate_delta": max(duplicate_deltas) if duplicate_deltas else 0.0,
            "stable_candidate": bool(
                len(rows) == len(seeds)
                and guard_passes == len(rows)
                and min(headroom_gains, default=-1.0) > 0.0
                and max(nll_deltas, default=1.0) <= 0.015
                and max(duplicate_deltas, default=1.0) <= 1e-12
            ),
        })

    return {
        "schema": 1,
        "version": "phase5-residual-calibration-ablation-v2",
        "read_only": True,
        "lineage_id": str(progress.get("lineage_id") or ""),
        "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
        "source_d_ff": source_d_ff,
        "target_d_ff": int(baseline.config.d_ff),
        "anchor_repetition": float(anchor["repetition_rate"]),
        "repetition_ceiling": ceiling,
        "baseline_headroom": float(ceiling - before["repetition_rate"]),
        "baseline_diversity": before_diversity,
        "protected_anchor_rows": len(anchor_rows),
        "protected_rows_filtered": int(filtered),
        "policy_summaries": policy_summaries,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only residual calibration ablation for Phase 5"
    )
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()
    report = run_ablation(args.state_dir, policy_filter=args.policy)
    target = Path(args.output)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "lineage_id": report["lineage_id"],
        "tokens_processed": report["tokens_processed"],
        "source_d_ff": report["source_d_ff"],
        "target_d_ff": report["target_d_ff"],
        "baseline_headroom": report["baseline_headroom"],
        "policy_summaries": report["policy_summaries"],
        "variants": [
            {
                "name": row["name"],
                "calibration_ok": row["calibration_ok"],
                "guard_ok": row["passes_unchanged_language_guard"],
                "nll_delta": row["delta"]["language_nll"],
                "repetition_delta": row["delta"]["repetition_rate"],
                "headroom_after": row["headroom"]["after"],
                "headroom_gain": row["headroom"]["gain"],
                "entropy_delta": row["delta"]["token_entropy"],
                "similarity_delta": row["delta"]["generation_similarity"],
                "duplicate_rate_after": row["cross_prompt_diversity"]["after"]["exact_duplicate_rate"],
                "jaccard_after": row["cross_prompt_diversity"]["after"]["mean_pairwise_token_jaccard"],
                "reasons": (
                    row["gate"]["local_reasons"]
                    + row["gate"]["after_anchor_violations"]
                ),
            }
            for row in report["results"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
