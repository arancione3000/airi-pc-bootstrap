"""Paired offline ablations on a pinned AIRI checkpoint; never promotes weights.

The default causal slice is the persisted training-only replay, NOT the full
live corpus. Results are diagnostic accepted-equivalent tokens, not durable
live learning. Each independent trial reloads the same model and optimizer.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

from .bootstrap_data import load_bootstrap_replay
from .bootstrap_training import (
    _batch,
    _filter_protected_replay,
    _learning_rate,
    _load_optimizer_checkpoint,
    _protected_causal_replay_rows,
    _protected_optimizer,
    _save_optimizer_checkpoint,
    _segment_language_gate,
    _language_quality,
    _sft_training_batch,
)
from .learning_efficiency_audit import read_json
from .phase5_diagnostics import (
    evaluate_conversational_probes,
    evaluate_phase5_language,
    protected_bootstrap_texts,
)
from .pretraining import CorpusDocument, pack_causal_blocks
from .qualification import checkpoint_digest
from .runtime import GeneralistRuntime
from .training import causal_training_objective, encode_sft_example, reference_kl_loss

_DATA_CACHE = {}
_ROW_COUNT_CACHE = {}


@dataclass(frozen=True)
class Trial:
    name: str = "baseline"
    segment_tokens: int = 4064
    base_learning_rate: float = 3e-4
    replay_examples: int = 2
    replay_token_fraction: float | None = None
    replay_weight: float = 4.0
    elementary_only: bool = True
    kl_weight: float = 0.5
    kl_source: str = "segment"
    kl_scope: str = "replay"
    anti_weight: float = 0.05
    eos_weight: float = 1.6
    warmup_steps: int = 4
    clip_norm: float = 1.0
    lr_multiplier: float = 1.0
    lr_schedule: str = "constant"
    optimizer_state: str = "reset"
    optimizer_kind: str = "AdamW"
    gradient_balance: bool = False
    balance_floor: float = 0.25
    repetition_sampling: bool = False
    curriculum: str = "short"
    prefix_acceptance: bool = False


def trial_matrix(segment_tokens=4064):
    base = asdict(Trial(segment_tokens=segment_tokens))
    variants = [
        ("baseline", {}),
        ("elementary32", {"replay_examples": 32}),
        ("broad32", {"replay_examples": 32, "elementary_only": False}),
        ("proportional25", {"replay_token_fraction": 0.25, "elementary_only": False}),
        (
            "token_normalized",
            {"replay_examples": 32, "elementary_only": False, "replay_weight": -1.0},
        ),
        (
            "balanced_gradients",
            {"replay_examples": 32, "elementary_only": False, "gradient_balance": True},
        ),
        (
            "balanced_exact",
            {
                "replay_examples": 32,
                "elementary_only": False,
                "gradient_balance": True,
                "balance_floor": 0.0,
            },
        ),
        ("balanced_elementary", {"gradient_balance": True, "balance_floor": 0.0}),
        ("kl_zero", {"kl_weight": 0.0}),
        ("kl_strong", {"kl_weight": 5.0}),
        (
            "durable_kl",
            {"kl_source": "durable", "kl_scope": "causal", "kl_weight": 5.0},
        ),
        (
            "durable_kl_strong",
            {"kl_source": "durable", "kl_scope": "causal", "kl_weight": 500.0},
        ),
        (
            "durable_kl_broad",
            {
                "kl_source": "durable",
                "kl_scope": "causal",
                "kl_weight": 500.0,
                "elementary_only": False,
                "replay_token_fraction": 0.25,
                "replay_weight": 1.0,
            },
        ),
        ("anti_strong", {"anti_weight": 1.0}),
        ("repetition_sampling", {"repetition_sampling": True}),
        ("cosine_segment", {"lr_schedule": "cosine"}),
        ("resume_momentum", {"optimizer_state": "resume"}),
        ("no_warmup", {"warmup_steps": 1}),
        ("clip_point1", {"clip_norm": 0.1}),
        ("general_mix", {"curriculum": "general"}),
        ("sgd_direction", {"optimizer_kind": "SGD", "lr_multiplier": 1000.0}),
        ("sgd_direction_strong", {"optimizer_kind": "SGD", "lr_multiplier": 10000.0}),
        (
            "sgd_broad",
            {
                "optimizer_kind": "SGD",
                "lr_multiplier": 10000.0,
                "elementary_only": False,
                "replay_token_fraction": 0.25,
                "replay_weight": 1.0,
            },
        ),
        ("prefix_acceptance", {"prefix_acceptance": True}),
    ]
    return [Trial(**{**base, **changes, "name": name}) for name, changes in variants]


def replay_selection(runtime, rows, *, rng, examples, token_fraction, causal_tokens):
    if token_fraction is not None and not 0 < token_fraction < 1:
        raise ValueError("actual replay token fraction must be between zero and one")
    key = (
        runtime.tokenizer.version,
        getattr(runtime.tokenizer, "digest", None),
        runtime.config.context_length,
        hashlib.sha256(
            json.dumps([r.messages for r in rows], sort_keys=True).encode()
        ).hexdigest(),
    )
    if key not in _ROW_COUNT_CACHE:
        _ROW_COUNT_CACHE[key] = [
            int(
                (
                    encode_sft_example(
                        r, runtime.tokenizer, runtime.config.context_length
                    )[1][1:]
                    != -100
                ).sum()
            )
            for r in rows
        ]
    counts = _ROW_COUNT_CACHE[key]
    if not rows or min(counts) <= 0:
        raise ValueError("replay rows must contain supervised targets")
    if token_fraction is None:
        indices = [rng.randrange(len(rows)) for _ in range(examples)]
    else:
        target = math.ceil(causal_tokens * token_fraction / (1 - token_fraction))
        indices = []
        tokens = 0
        # A shuffled cycle gives diverse coverage without replacement until
        # the pool is exhausted. No padding token is counted as supervision.
        while tokens < target:
            pool = list(range(len(rows)))
            rng.shuffle(pool)
            for idx in pool:
                indices.append(idx)
                tokens += counts[idx]
                if tokens >= target:
                    break
    if not indices:
        raise ValueError("empty replay selection")
    return indices, sum(counts[i] for i in indices)


def grad_vector(model):
    import torch

    return {
        n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def grad_norm(grads):
    return math.sqrt(sum(float(g.double().square().sum()) for g in grads.values()))


def grad_relation(left, right):
    ln, rn = grad_norm(left), grad_norm(right)
    dot = sum(float((left[k].double() * right[k].double()).sum()) for k in left)
    return {"left_norm": ln, "right_norm": rn, "cosine": dot / max(1e-30, ln * rn)}


def slice_hash(blocks, replay_rows):
    payload = {
        "causal_blocks": blocks,
        "replay_messages": [r.messages for r in replay_rows],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_digest(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()


def validate_output_path(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("benchmark output must be separate from source state")


def summarize_trials(results):
    groups = {}
    for row in results:
        key = (row["trial"]["name"], row["trial"]["segment_tokens"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (name, size), rows in sorted(groups.items()):
        n = len(rows)
        accepted = sum(r["accepted"] for r in rows)
        elapsed = sum(
            r["training_evaluation_seconds"] + r["checkpoint_seconds"] for r in rows
        )
        tokens = sum(r["accepted_equivalent_tokens"] for r in rows)
        attempted = sum(r["attempted_causal_tokens"] for r in rows)
        replay = sum(r["replay_tokens"] for r in rows)
        p = accepted / n
        z = 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        summary.append(
            {
                "trial": name,
                "requested_segment_tokens": size,
                "attempts": n,
                "acceptance_rate": p,
                "rollback_rate": 1 - p,
                "acceptance_wilson_95": [center - radius, center + radius],
                "accepted_equivalent_tokens_per_hour": tokens
                / max(1e-9, elapsed)
                * 3600,
                "attempted_tokens_per_hour": attempted
                / max(1e-9, sum(r["training_evaluation_seconds"] for r in rows))
                * 3600,
                "training_wall_seconds": sum(
                    r["training_evaluation_seconds"] for r in rows
                ),
                "checkpoint_seconds": sum(r["checkpoint_seconds"] for r in rows),
                "actual_replay_token_fraction": replay / max(1, replay + attempted),
                "mean_repetition_before": sum(
                    r["before"]["repetition_rate"] for r in rows
                )
                / n,
                "mean_repetition_after": sum(
                    r["after"]["repetition_rate"] for r in rows
                )
                / n,
                "mean_nll_before": sum(r["before"]["language_nll"] for r in rows) / n,
                "mean_nll_after": sum(r["after"]["language_nll"] for r in rows) / n,
                "mean_headroom_after": sum(r["headroom_after"] for r in rows) / n,
                "pathological_trials": sum(
                    bool(r["after"]["pathological_repetition"]) for r in rows
                ),
                "corpus_validation_loss": None,
                "corpus_validation_unavailable_reason": "the original held-out corpus is not included in persisted replay",
                "proven_stable_segment": False,
                "stability_limitation": "independent reset trials do not demonstrate sustained accepted live lineage progress",
            }
        )
    return {
        "groups": summary,
        "live_tokens_persisted": 0,
        "success_claim_allowed": False,
        "scope": "diagnostic paired trials; empirical live success remains unproven",
    }


def objective_gradient_probe(
    root, *, seed=5000000, replay_examples=2, elementary_only=True
):
    """Separate full-effective-batch raw/weighted objectives at the same weights.

    This diagnostic is not included in throughput comparisons. Every objective
    gets its own backward pass; no optimizer step or source write is performed.
    """
    import torch

    root = Path(root)
    runtime = GeneralistRuntime.from_checkpoint(root / "bootstrap-data/candidate")
    reference = copy.deepcopy(runtime.model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    replay = load_bootstrap_replay(root / "bootstrap-data")
    protected = protected_bootstrap_texts()
    docs = [
        d
        for d in replay.documents
        if len(d.text) <= 220
        and not any(text in " ".join(d.text.casefold().split()) for text in protected)
    ]
    blocks = pack_causal_blocks(
        docs, runtime.tokenizer, context_length=runtime.config.context_length
    )
    rng = random.Random(seed)
    indices = [rng.randrange(len(blocks)) for _ in range(32)]
    rows, _ = _protected_causal_replay_rows(
        replay.sft_train, elementary_only=elementary_only
    )
    replay_indices, replay_count = replay_selection(
        runtime,
        rows,
        rng=random.Random(seed + 100000),
        examples=replay_examples,
        token_fraction=None,
        causal_tokens=4064,
    )
    batches = {
        "causal": [
            _batch(blocks, indices[i : i + 2], device=runtime.device)
            for i in range(0, 32, 2)
        ],
        "replay": [
            _sft_training_batch(runtime, rows, replay_indices[i : i + 2])
            for i in range(0, len(replay_indices), 2)
        ],
    }
    output = {
        "schema": 1,
        "checkpoint_hash": checkpoint_digest(root / "bootstrap-data/candidate"),
        "seed": seed,
        "elementary_only": elementary_only,
        "replay_examples": replay_examples,
        "new_tokens": sum(
            int((labels[:, 1:] != -100).sum()) for _, labels in batches["causal"]
        ),
        "replay_tokens": replay_count,
        "objectives": {},
        "scope": "separate raw gradients at initial checkpoint; KL uses the live pre-segment self-reference",
    }
    causal = None
    for scope, kind, weight in [
        ("causal", "ce", 1.0),
        ("causal", "anti", 0.05),
        ("replay", "ce", 4.0),
        ("replay", "anti", 0.2),
        ("replay", "kl", 0.5),
        ("causal", "kl", 0.5),
    ]:
        runtime.model.zero_grad(set_to_none=True)
        total = sum(int((labels[:, 1:] != -100).sum()) for _, labels in batches[scope])
        value = 0.0
        for ids, labels in batches[scope]:
            count = int((labels[:, 1:] != -100).sum())
            logits = runtime.model(ids)["logits"]
            if kind == "kl":
                with torch.no_grad():
                    ref_logits = reference(ids)["logits"]
                loss, _ = reference_kl_loss(logits, ref_logits, labels)
            else:
                ce, _ = causal_training_objective(
                    logits, labels, ids, eos_loss_weight=1.6 if scope == "causal" else 1
                )
                if kind == "anti":
                    full, _ = causal_training_objective(
                        logits,
                        labels,
                        ids,
                        eos_loss_weight=1.6 if scope == "causal" else 1,
                        repetition_unlikelihood_weight=1,
                    )
                    loss = full - ce
                else:
                    loss = ce
            (loss * count / total).backward()
            value += float(loss.detach()) * count / total
        gradients = grad_vector(runtime.model)
        norm = grad_norm(gradients)
        name = scope + "_" + kind
        output["objectives"][name] = {
            "raw_gradient_norm": norm,
            "weight": weight,
            "weighted_gradient_norm": norm * weight,
            "loss": value,
        }
        if causal is None:
            causal = gradients
        else:
            output["objectives"][name]["relation_to_causal_ce"] = grad_relation(
                causal, gradients
            )
            del gradients
    return output


def architecture_geometry_probe(root):
    """Test inherited candidates before training; incompatible geometry cannot promote."""
    import torch
    from .model import CausalTransformerLM, GeneralistLMConfig, parameter_count
    from .research_cycle import _transfer_compatible_weights

    root = Path(root)
    runtime = GeneralistRuntime.from_checkpoint(root / "bootstrap-data/candidate")
    baseline = evaluate_phase5_language(runtime)
    anchor = read_json(root / "bootstrap-data/language-guard.json")["report"]
    rows = []
    configurations = [
        (256, 16, 3712, "mha", 4),
        (384, 12, 3072, "mha", 6),
        (512, 12, 2048, "mha", 8),
        (512, 12, 2304, "gqa", 8),
    ]
    ids = torch.tensor(
        [
            runtime.tokenizer.encode(
                "Una persona apre la finestra e osserva la strada."
            )[: runtime.config.context_length]
        ]
    )
    runtime.model.eval()
    with torch.no_grad():
        source_logits = runtime.model(ids)["logits"]
    for width, depth, ff, attention, heads in configurations:
        config = GeneralistLMConfig(
            **{
                **runtime.config.to_dict(),
                "d_model": width,
                "n_layers": depth,
                "d_ff": ff,
                "n_heads": heads,
                "attention_type": attention,
                "n_kv_heads": 2 if attention == "gqa" else heads,
            }
        ).validate()
        torch.manual_seed(9001)
        model = CausalTransformerLM(config)
        transfer = _transfer_compatible_weights(
            runtime.model,
            model,
            source_tokenizer=runtime.tokenizer,
            target_tokenizer=runtime.tokenizer,
        )
        candidate = GeneralistRuntime(model, config, tokenizer=runtime.tokenizer)
        candidate.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            delta = float((candidate.model(ids)["logits"] - source_logits).abs().max())
        report = evaluate_phase5_language(candidate)
        eligible, gate = _segment_language_gate(
            baseline, report, anchor, attempted_tokens=0
        )
        rows.append(
            {
                "config": config.to_dict(),
                "parameters": parameter_count(model),
                "parameter_ratio": parameter_count(model)
                / parameter_count(runtime.model),
                "transfer": transfer,
                "max_logit_delta": delta,
                "language": {k: v for k, v in report.items() if k != "traces"},
                "inherited_candidate_passes_existing_language_gate": eligible,
                "reasons": gate["local_reasons"] + gate["after_anchor_violations"],
                "eval_seconds": time.perf_counter() - start,
                "training_tokens": 0,
                "promoted": False,
            }
        )
        del candidate, model
    return {
        "schema": 1,
        "checkpoint_hash": checkpoint_digest(root / "bootstrap-data/candidate"),
        "baseline_architecture": runtime.config.to_dict(),
        "baseline_parameters": parameter_count(runtime.model),
        "candidates": rows,
        "scope": "same-checkpoint inheritance feasibility, not a trained geometry efficiency comparison; all candidates remain temporary and unpromoted",
    }


def run_trial(root, trial, *, seed, causal_documents=None, measure_gradients=False):
    import torch

    root = Path(root)
    b = root / "bootstrap-data"
    started = time.perf_counter()
    torch.manual_seed(seed)
    runtime = GeneralistRuntime.from_checkpoint(b / "candidate", device="cpu")
    p = read_json(b / "progress.json")
    anchor = read_json(b / "language-guard.json")["report"]
    before = evaluate_phase5_language(runtime)
    data_key = (
        str(root),
        id(causal_documents),
        trial.curriculum,
        trial.repetition_sampling,
    )
    if data_key not in _DATA_CACHE:
        replay = load_bootstrap_replay(b)
        documents = (
            causal_documents if causal_documents is not None else replay.documents
        )
        protected = protected_bootstrap_texts()
        documents = [
            d
            for d in documents
            if not any(
                text and text in " ".join(d.text.casefold().split())
                for text in protected
            )
        ]
        if trial.curriculum == "short":
            documents = [d for d in documents if len(d.text) <= 220]
        if trial.repetition_sampling:
            documents = [
                d
                for d in documents
                if len(set(runtime.tokenizer.encode(d.text)))
                / max(1, len(runtime.tokenizer.encode(d.text)))
                >= 0.4
            ]
        if not documents:
            raise ValueError("no holdout-disjoint causal documents")
        blocks = pack_causal_blocks(
            documents, runtime.tokenizer, context_length=runtime.config.context_length
        )
        _DATA_CACHE[data_key] = (blocks, replay.sft_train)
    blocks, sft_train = _DATA_CACHE[data_key]
    rows, _ = _protected_causal_replay_rows(
        sft_train, elementary_only=trial.elementary_only
    )
    rows, filtered = _filter_protected_replay(rows)
    initial_digest = checkpoint_digest(b / "candidate")
    reference = (
        copy.deepcopy(runtime.model)
        if trial.kl_source == "segment"
        else GeneralistRuntime.from_checkpoint(
            b / "language-guard-best", device="cpu"
        ).model
    )
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer, optimizer_report = _protected_optimizer(
        runtime.model,
        learning_rate=trial.base_learning_rate * p["segment_guard_lr_scale"],
    )
    if trial.optimizer_kind == "SGD":
        optimizer = torch.optim.SGD(
            optimizer.param_groups, lr=1e-4, momentum=0.0, weight_decay=0.01
        )
        optimizer_report = {
            **optimizer_report,
            "name": "SGD",
            "momentum": 0.0,
            "betas": None,
        }
    elif trial.optimizer_kind != "AdamW":
        raise ValueError("unknown optimizer kind")
    if trial.optimizer_state == "resume":
        if trial.optimizer_kind != "AdamW":
            raise ValueError(
                "live AdamW moments cannot be loaded into a different optimizer"
            )
        _load_optimizer_checkpoint(optimizer, b / "optimizer.pt")
    elif trial.optimizer_state != "reset":
        raise ValueError("unknown optimizer policy")
    attempted = replay_tokens = 0
    steps = []
    selected_blocks = []
    selected_rows = []
    instrument_seconds = 0.0
    prefix = None
    train_started = time.perf_counter()
    runtime.model.train()
    expected_steps = math.ceil(
        trial.segment_tokens / (32 * (runtime.config.context_length - 1))
    )
    while attempted < trial.segment_tokens:
        step = len(steps)
        rng = random.Random(seed + step)
        indices = [rng.randrange(len(blocks)) for _ in range(32)]
        selected_blocks.extend(blocks[i] for i in indices)
        prepared = []
        for offset in range(0, 32, 2):
            ids, labels = _batch(
                blocks, indices[offset : offset + 2], device=runtime.device
            )
            count = int((labels[:, 1:] != -100).sum())
            prepared.append((ids, labels, count))
        causal_count = sum(r[2] for r in prepared)
        lr = _learning_rate(
            base_lr=trial.base_learning_rate
            * p["segment_guard_lr_scale"]
            * trial.lr_multiplier,
            processed_tokens=p["tokens_processed"] + attempted,
            target_tokens=p["target_tokens"],
            warmup_tokens=100000,
        )
        lr *= min(1, (step + 1) / trial.warmup_steps)
        if trial.lr_schedule == "cosine":
            lr *= 0.5 * (1 + math.cos(math.pi * step / max(1, expected_steps)))
        elif trial.lr_schedule != "constant":
            raise ValueError("unknown LR schedule")
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_multiplier"]
        optimizer.zero_grad(set_to_none=True)
        causal_loss = 0.0
        for ids, labels, count in prepared:
            logits = runtime.model(ids)["logits"]
            loss, objective = causal_training_objective(
                logits,
                labels,
                ids,
                eos_loss_weight=trial.eos_weight,
                repetition_unlikelihood_weight=trial.anti_weight,
            )
            (loss * count / causal_count).backward()
            causal_loss += float(loss.detach()) * count / causal_count
        instrument_start = time.perf_counter()
        causal_grads = (
            grad_vector(runtime.model)
            if measure_gradients or trial.gradient_balance
            else None
        )
        instrument_seconds += time.perf_counter() - instrument_start
        replay_indices, replay_count = replay_selection(
            runtime,
            rows,
            rng=random.Random(seed + 100000 + step),
            examples=trial.replay_examples,
            token_fraction=trial.replay_token_fraction,
            causal_tokens=causal_count,
        )
        selected_rows.extend(rows[i] for i in replay_indices)
        weight = (
            trial.replay_weight
            if trial.replay_weight >= 0
            else replay_count / causal_count
        )
        if trial.gradient_balance:
            optimizer.zero_grad(set_to_none=True)
        replay_loss = kl_value = 0.0
        kl_count = 0
        for offset in range(0, len(replay_indices), 2):
            ids, labels = _sft_training_batch(
                runtime, rows, replay_indices[offset : offset + 2]
            )
            count = int((labels[:, 1:] != -100).sum())
            logits = runtime.model(ids)["logits"]
            loss, _ = causal_training_objective(
                logits, labels, ids, repetition_unlikelihood_weight=trial.anti_weight
            )
            with torch.no_grad():
                ref_logits = reference(ids)["logits"]
            kl, nt = reference_kl_loss(logits, ref_logits, labels)
            scale = count / replay_count
            if trial.kl_scope == "replay":
                (weight * loss * scale + trial.kl_weight * kl * scale).backward()
                kl_value += float(kl.detach()) * scale
                kl_count += nt
            else:
                (weight * loss * scale).backward()
            replay_loss += float(loss.detach()) * scale
        if trial.kl_scope == "causal":
            for ids, labels, count in prepared:
                logits = runtime.model(ids)["logits"]
                with torch.no_grad():
                    ref_logits = reference(ids)["logits"]
                kl, nt = reference_kl_loss(logits, ref_logits, labels)
                (trial.kl_weight * kl * count / causal_count).backward()
                kl_value += float(kl.detach()) * count / causal_count
                kl_count += nt
        elif trial.kl_scope != "replay":
            raise ValueError("unknown KL scope")
        instrument_start = time.perf_counter()
        pressure = None
        if causal_grads is not None:
            current = grad_vector(runtime.model)
            replay_grads = (
                current
                if trial.gradient_balance
                else {k: current[k] - causal_grads[k] for k in current}
            )
            pressure = grad_relation(causal_grads, replay_grads)
            pressure["scope"] = (
                "causal CE+anti versus weighted replay CE+anti+KL; individual auxiliary gradients measured separately by objective probe"
            )
            if trial.gradient_balance:
                balance = min(
                    4,
                    max(
                        trial.balance_floor,
                        pressure["left_norm"] / max(1e-20, pressure["right_norm"]),
                    ),
                )
                for name, parameter in runtime.model.named_parameters():
                    if name in causal_grads:
                        parameter.grad = (
                            causal_grads[name] + balance * replay_grads[name]
                        )
                pressure["balance_multiplier"] = balance
            del current, replay_grads, causal_grads
        instrument_seconds += time.perf_counter() - instrument_start
        norm = float(
            torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), trial.clip_norm)
        )
        if not math.isfinite(norm):
            raise ValueError("non-finite gradient")
        optimizer.step()
        attempted += causal_count
        replay_tokens += replay_count
        event = {
            "step": step,
            "effective_lr": lr,
            "causal_tokens": causal_count,
            "replay_tokens": replay_count,
            "replay_weight": weight,
            "causal_loss": causal_loss,
            "replay_loss": replay_loss,
            "kl_loss": kl_value,
            "kl_tokens": kl_count,
            "combined_gradient_norm_before_clipping": norm,
            "gradient_pressure": pressure,
        }
        if trial.prefix_acceptance:
            report = evaluate_phase5_language(runtime)
            ok, guard = _segment_language_gate(
                before, report, anchor, attempted_tokens=attempted
            )
            event["prefix_gate"] = {
                "accepted": ok,
                "reasons": guard["local_reasons"] + guard["after_anchor_violations"],
            }
            if ok:
                prefix = {
                    "tokens": attempted,
                    "model": copy.deepcopy(runtime.model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "report": report,
                }
            runtime.model.train()
        steps.append(event)
    after = evaluate_phase5_language(runtime)
    accepted, gate = _segment_language_gate(
        before, after, anchor, attempted_tokens=attempted
    )
    equivalent = attempted if accepted else 0
    prefix_restored = False
    if trial.prefix_acceptance and not accepted and prefix:
        runtime.model.load_state_dict(prefix["model"])
        optimizer.load_state_dict(prefix["optimizer"])
        after = evaluate_phase5_language(runtime)
        accepted, gate = _segment_language_gate(
            before, after, anchor, attempted_tokens=prefix["tokens"]
        )
        equivalent = prefix["tokens"] if accepted else 0
        prefix_restored = True
    train_seconds = time.perf_counter() - train_started
    conversations = evaluate_conversational_probes(runtime)
    persist_started = time.perf_counter()
    if accepted:
        with tempfile.TemporaryDirectory(prefix="airi-benchmark-checkpoint-") as tmp:
            runtime.save_checkpoint(
                Path(tmp) / "candidate", metadata={"benchmark_only": True}
            )
            _save_optimizer_checkpoint(optimizer, Path(tmp) / "optimizer.pt")
    persist_seconds = time.perf_counter() - persist_started if accepted else 0.0
    total_seconds = time.perf_counter() - started
    return {
        "schema": 1,
        "trial": asdict(trial),
        "seed": seed,
        "checkpoint_hash": initial_digest,
        "architecture": runtime.config.to_dict(),
        "dataset_slice_hash": slice_hash(selected_blocks, selected_rows),
        "causal_slice_hash": slice_hash(selected_blocks, []),
        "data_scope": (
            "provided training-only documents"
            if causal_documents is not None
            else "persisted training-only replay sample; not the full live corpus"
        ),
        "optimizer": optimizer_report,
        "steps": steps,
        "before": before,
        "after": after,
        "gate": gate,
        "accepted": accepted,
        "attempted_causal_tokens": attempted,
        "accepted_equivalent_tokens": equivalent,
        "replay_tokens": replay_tokens,
        "actual_replay_token_fraction": replay_tokens
        / max(1, replay_tokens + attempted),
        "language_quality_before": _language_quality(before),
        "language_quality_after": _language_quality(after),
        "headroom_before": anchor["repetition_rate"] + 0.08 - before["repetition_rate"],
        "headroom_after": anchor["repetition_rate"] + 0.08 - after["repetition_rate"],
        "conversation_probe": conversations,
        "prefix_restored": prefix_restored,
        "training_evaluation_seconds": train_seconds,
        "checkpoint_seconds": persist_seconds,
        "instrumentation_seconds": instrument_seconds,
        "total_wall_seconds": total_seconds,
        "attempted_tokens_per_hour": attempted / max(1e-9, train_seconds) * 3600,
        "accepted_equivalent_tokens_per_hour": equivalent
        / max(1e-9, train_seconds + persist_seconds)
        * 3600,
        "end_to_end_equivalent_tokens_per_hour": equivalent
        / max(1e-9, total_seconds)
        * 3600,
        "source_state_modified": False,
        "live_tokens_persisted": 0,
        "heldout_filtered": filtered,
        "torch_version": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", default="baseline,broad32")
    parser.add_argument("--segments", default="4064")
    parser.add_argument("--seeds", default="5000000,5000001,5000002")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--measure-gradients", action="store_true")
    parser.add_argument("--gradient-probe-only", action="store_true")
    parser.add_argument("--geometry-probe-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--causal-documents",
        help="JSONL of reviewed training-only CorpusDocument objects, never validation",
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    root, out = Path(args.state_dir).resolve(), Path(args.output).resolve()
    validate_output_path(root, out)
    if args.summary_only:
        result = summarize_trials(
            [json.loads(line) for line in out.read_text().splitlines()]
        )
        out.with_suffix(".summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result), flush=True)
        return
    before_hash = source_digest(root)
    if args.gradient_probe_only or args.geometry_probe_only:
        result = (
            architecture_geometry_probe(root)
            if args.geometry_probe_only
            else objective_gradient_probe(root)
        )
        if source_digest(root) != before_hash:
            raise ValueError("objective probe modified source state")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result), flush=True)
        return
    documents = None
    dataset_input_hash = (
        hashlib.sha256(Path(args.causal_documents).read_bytes()).hexdigest()
        if args.causal_documents
        else before_hash
    )
    if args.causal_documents:
        documents = []
        for line in Path(args.causal_documents).read_text().splitlines():
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("causal input requires explicit train split")
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
                raise ValueError("causal document hash mismatch")
            documents.append(
                CorpusDocument(
                    **{
                        k: row[k]
                        for k in ("source", "text", "sha256", "bytes", "domain")
                    }
                )
            )
    completed = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            completed.add(
                (row["trial"]["name"], row["trial"]["segment_tokens"], row["seed"])
            )
            if row["checkpoint_hash"] != checkpoint_digest(
                root / "bootstrap-data/candidate"
            ):
                raise ValueError("cannot resume against a different checkpoint")
            if row["threads"] != args.threads:
                raise ValueError("cannot mix hardware settings in one journal")
            if row.get("input_state_digest", before_hash) != before_hash:
                raise ValueError("cannot resume against changed source state")
            if (
                row.get("input_dataset_digest", dataset_input_hash)
                != dataset_input_hash
            ):
                raise ValueError("cannot resume against changed dataset input")
            expected = next(
                (
                    trial
                    for trial in trial_matrix(row["trial"]["segment_tokens"])
                    if trial.name == row["trial"]["name"]
                ),
                None,
            )
            if expected is None or Trial(**row["trial"]) != expected:
                raise ValueError("cannot resume with changed trial parameters")
    out.parent.mkdir(parents=True, exist_ok=True)
    # flock is released by the kernel after a crash. It prevents two processes
    # from silently mixing timing samples or replacing each other's journal.
    lock = out.with_suffix(out.suffix + ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = set(args.trials.split(","))
    for size in map(int, args.segments.split(",")):
        matrix = trial_matrix(size)
        if names - {t.name for t in matrix}:
            raise ValueError("unknown trial name")
        for seed in map(int, args.seeds.split(",")):
            for trial in matrix:
                if trial.name not in names or (trial.name, size, seed) in completed:
                    continue
                result = run_trial(
                    root,
                    trial,
                    seed=seed,
                    causal_documents=documents,
                    measure_gradients=args.measure_gradients,
                )
                result["threads"] = args.threads
                result["input_state_digest"] = before_hash
                result["input_dataset_digest"] = dataset_input_hash
                if source_digest(root) != before_hash:
                    raise ValueError("source state changed during benchmark")
                existing = out.read_text(encoding="utf-8") if out.exists() else ""
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=out.parent, delete=False
                ) as handle:
                    handle.write(existing + json.dumps(result, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary = handle.name
                os.replace(temporary, out)
                print(
                    json.dumps(
                        {
                            "trial": trial.name,
                            "tokens": size,
                            "seed": seed,
                            "accepted": result["accepted"],
                            "equivalent_per_hour": result[
                                "accepted_equivalent_tokens_per_hour"
                            ],
                            "repetition": result["after"]["repetition_rate"],
                            "headroom": result["headroom_after"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
