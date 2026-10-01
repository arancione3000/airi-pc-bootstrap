"""Read-only Phase 5 forensic audit. Missing historical measurements stay null."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot_baseline(state_dir, *, state_sha, code_sha):
    from .qualification import checkpoint_digest
    from .runtime import checkpoint_model_digest

    root = Path(state_dir)
    b = root / "bootstrap-data"
    p = read_json(b / "progress.json")
    e = read_json(b / "learning-efficiency.json")
    s = read_json(b / "segment-guard-last.json")
    lineage = read_json(root / "lineage.json")
    live = s["validation_after"] if s["accepted"] else s["validation_before"]
    anchor = s["anchor"]
    checkpoint = b / "candidate"
    model_sha = hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest()
    if model_sha != lineage["active_model_sha256"]:
        raise ValueError("checkpoint does not match pinned live lineage")
    baseline = {
        "schema": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "state_updated_at": p["updated_at_unix"],
        "state_commit_sha": state_sha,
        "code_commit_sha": code_sha,
        "lineage_id": lineage["lineage_id"],
        "checkpoint_digest": checkpoint_digest(checkpoint),
        "model_sha256": model_sha,
        "verified_model_payload_digest": checkpoint_model_digest(checkpoint),
        "architecture_genome": lineage["active_genome"],
        "parameters": lineage["parameters"],
        "tokenizer": lineage["tokenizer_version"],
        "vocab_size": lineage["tokenizer_vocab_size"],
        "context_length": lineage["active_genome"]["context_length"],
        "train_loss": p["last_train_loss"],
        "validation_loss": p["last_validation_loss"],
        "durable_anchor_repetition_rate": anchor["repetition_rate"],
        "hard_repetition_boundary": anchor["repetition_rate"] + 0.08,
        "repetition_headroom": anchor["repetition_rate"]
        + 0.08
        - live["repetition_rate"],
        "language_quality_metrics": {k: v for k, v in live.items() if k != "traces"},
        "current_segment_size": p["segment_guard_last_attempted_tokens"],
        "effective_budget_tokens": p["segment_guard_effective_budget_tokens"],
        "requested_segment_size": p["segment_guard_requested_budget_tokens"],
        "lr_scale": p["segment_guard_lr_scale"],
        "effective_lr": p["learning_rate"],
        "recovery_mode": p["segment_guard_recovery_mode"],
        "success_streak": p["segment_guard_success_streak"],
        "rejection_streak": p["segment_guard_consecutive_rejections"],
        "replay_configuration": p["protected_continual_training"],
        "actual_replay_supervised_tokens": s["replay_supervised_tokens"],
        "actual_new_supervised_tokens": s["new_supervised_tokens"],
        "actual_replay_token_fraction": s["actual_replay_token_fraction"],
        "replay_loss_weight": s["replay_loss_weight"],
        "reference_kl_weight": s["reference_kl_weight"],
        "optimizer": p["protected_optimizer"],
        "optimizer_storage": p.get("optimizer_storage"),
        "throughput_scope": "existing telemetry: training/evaluation plus reported local checkpoint persistence; excludes workflow acquisition/git push/startup",
        "gradient_pressure_ratio": None,
        "gradient_pressure_unavailable_reason": "not recorded by the live workflow",
    }
    for k in ("tokens_processed", "valid_tokens_processed", "target_tokens"):
        baseline[k] = p[k]
    for k in (
        "language_nll",
        "repetition_rate",
        "multiword_output_rate",
        "pathological_repetition",
        "unique_token_ratio",
        "longest_repeated_token_run",
    ):
        baseline[k] = live[k]
    for k in (
        "attempted_segments",
        "accepted_segments",
        "rollback_segments",
        "acceptance_rate",
        "rollback_rate",
        "attempted_tokens",
        "accepted_tokens",
        "effective_new_tokens",
        "attempted_tokens_per_hour",
        "accepted_tokens_per_hour",
        "cumulative_language_quality_gain",
        "learning_gain_per_100k_tokens",
        "elapsed_training_seconds",
        "checkpoint_persist_seconds",
    ):
        baseline[k] = e[k]
    baseline["minimum_target_accepted_tokens_per_hour"] = (
        e["accepted_tokens_per_hour"] * 3
    )
    baseline["strong_target_accepted_tokens_per_hour"] = (
        e["accepted_tokens_per_hour"] * 5
    )
    return baseline


def forensic_rollbacks(progress, efficiency):
    rejections = list(progress.get("segment_guard_rejections") or [])
    accepted = list(progress.get("segment_guard_acceptances") or [])
    causes = Counter(reason for row in rejections for reason in row["reasons"])
    buckets = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    events = []
    for success, rows in ((False, rejections), (True, accepted)):
        for row in rows:
            tokens = row.get("segment_tokens", row.get("attempted_tokens"))
            lr = row.get("learning_rate_scale")
            buckets[str((tokens, lr))]["accepted" if success else "rejected"] += 1
            events.append({"accepted": success, **row})
    for group in buckets.values():
        total = group["accepted"] + group["rejected"]
        group["rejection_rate_in_separately_capped_history"] = group["rejected"] / total
    history = efficiency.get("history") or []
    span = (
        history[-1]["updated_at_unix"] - history[0]["updated_at_unix"]
        if len(history) > 1
        else 0
    )
    wall_tokens = sum(r["accepted_tokens"] for r in history[1:])
    return {
        "scope": "last 32 rejections and last 32 acceptances are separately capped, not a chronological unbiased window",
        "causes_in_retained_rejections": dict(causes),
        "by_segment_tokens_and_lr": dict(buckets),
        "retained_events": events,
        "chronological_efficiency_history": history,
        "recent_window": {
            "attempts": len(history),
            "accepted": sum(bool(r["accepted"]) for r in history),
            "rollback_rate": sum(not r["accepted"] for r in history)
            / max(1, len(history)),
            "calendar_wall_seconds": span,
            "calendar_accepted_tokens_excluding_first_event": wall_tokens,
            "calendar_accepted_tokens_per_hour": (
                wall_tokens / span * 3600 if span > 0 else None
            ),
            "training_evaluation_seconds_excluding_first_event": sum(
                r["elapsed_training_seconds"] for r in history[1:]
            ),
        },
        "not_recoverable_from_existing_history": [
            "per-objective gradient norms",
            "optimizer moment norms",
            "absolute before/after repetition and NLL for all trials",
            "causes for all cumulative rollbacks",
            "architecture causal effect",
        ],
    }


def dataset_audit(documents, tokenizer):
    """Audit only supplied training data, with explicit measurable proxies."""
    exact = Counter(d.sha256 for d in documents)
    normalized = Counter(" ".join(d.text.casefold().split()) for d in documents)
    tokens = Counter()
    grams = Counter()
    domains = Counter()
    sources = Counter()
    languages = Counter()
    lengths = []
    repeated = 0
    short = 0
    total = 0
    unique_mass = 0
    seen = set()
    for d in documents:
        ids = tokenizer.encode(d.text)
        tokens.update(ids)
        grams.update(zip(ids, ids[1:], ids[2:], ids[3:]))
        lengths.append(len(ids))
        total += len(ids)
        if d.sha256 not in seen:
            unique_mass += len(ids)
            seen.add(d.sha256)
        domains[d.domain] += len(ids)
        sources[d.source.split(":")[0]] += len(ids)
        languages[
            (
                d.source.rsplit(":", 1)[-1]
                if d.source.rsplit(":", 1)[-1] in {"en", "it"}
                else "unattributed"
            )
        ] += len(ids)
        short += int(len(ids) < 8)
        repeated += int(len(ids) >= 12 and len(set(ids)) / len(ids) < 0.2)
    lengths.sort()
    template = Counter(re.sub(r"\d+", "<number>", d.text.casefold()) for d in documents)
    buckets = defaultdict(list)
    shingle_sets = []
    near_docs = set()
    comparisons = 0
    truncated_buckets = 0
    # Deterministic, bounded MinHash candidate search; exact Jaccard verifies
    # each proposed match. This is an estimate with unmeasured LSH recall.
    for index, document in enumerate(documents):
        text = " ".join(document.text.casefold().split())
        shingles = {text[i : i + 5] for i in range(max(1, len(text) - 4))}
        hashes = [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
            for s in shingles
        ]
        signature = tuple(
            min(((h * (2 * salt + 1) + salt) & ((1 << 64) - 1)) for h in hashes)
            for salt in range(1, 9)
        )
        candidates = set()
        for band in range(4):
            key = (band, signature[2 * band : 2 * band + 2])
            candidates.update(buckets[key][-64:])
            truncated_buckets += int(len(buckets[key]) > 64)
            buckets[key].append(index)
        for old in candidates:
            comparisons += 1
            prior = shingle_sets[old]
            if len(shingles & prior) / max(1, len(shingles | prior)) >= 0.9:
                near_docs.add(index)
                break
        shingle_sets.append(shingles)
    return {
        "scope": "supplied training documents only; persisted replay is a bounded sample of the 32M source corpus",
        "documents": len(documents),
        "exact_duplicate_rate": 1 - len(exact) / max(1, len(documents)),
        "normalized_duplicate_rate": 1 - len(normalized) / max(1, len(documents)),
        "near_duplicate_method": "case/whitespace normalization only; semantic near-duplicates unmeasured",
        "near_duplicate_char5_jaccard_90_rate_estimate": len(near_docs)
        / max(1, len(documents)),
        "near_duplicate_search": {
            "method": "8-value MinHash, 4 bands, exact Jaccard >=.9 verification; last 64 candidates per bucket",
            "candidate_comparisons": comparisons,
            "truncated_bucket_observations": truncated_buckets,
            "recall_measured": False,
            "semantic_similarity_measured": False,
        },
        "domain_supervised_tokens": dict(domains),
        "source_supervised_tokens": dict(sources),
        "source_attributed_language_tokens": dict(languages),
        "language_balance_method": "immutable source attribution; no language-ID inference",
        "token_count": total,
        "distinct_token_ids": len(tokens),
        "top_token_frequency": tokens.most_common(20),
        "top_4grams": [(list(g), n) for g, n in grams.most_common(20)],
        "length_quantiles": {
            str(q): (
                lengths[min(len(lengths) - 1, int(q * (len(lengths) - 1)))]
                if lengths
                else None
            )
            for q in (0, 0.25, 0.5, 0.75, 0.95, 1)
        },
        "extremely_short_documents": short,
        "low_diversity_documents": repeated,
        "numeric_template_duplicate_rate": 1 - len(template) / max(1, len(documents)),
        "unique_document_token_mass": unique_mass,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--state-sha", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.state_dir).resolve()
    out = Path(args.output_dir).resolve()
    if out == root or root in out.parents:
        raise ValueError("audit output must be outside source state")
    from .bootstrap_data import load_bootstrap_replay
    from .runtime import GeneralistRuntime

    baseline = snapshot_baseline(root, state_sha=args.state_sha, code_sha=args.code_sha)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_start.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n"
    )
    b = root / "bootstrap-data"
    forensic = forensic_rollbacks(
        read_json(b / "progress.json"), read_json(b / "learning-efficiency.json")
    )
    (out / "forensic_rollbacks.json").write_text(
        json.dumps(forensic, indent=2, sort_keys=True) + "\n"
    )
    runtime = GeneralistRuntime.from_checkpoint(b / "candidate", device="cpu")
    replay = load_bootstrap_replay(b)
    data = dataset_audit(replay.documents, runtime.tokenizer)
    (out / "dataset_audit.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "baseline": baseline["tokens_processed"],
                "accepted_per_hour": baseline["accepted_tokens_per_hour"],
                "rollback_rate": baseline["rollback_rate"],
                "causes": forensic["causes_in_retained_rejections"],
                "documents": data["documents"],
            }
        )
    )


if __name__ == "__main__":
    main()
