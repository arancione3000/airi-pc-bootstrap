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
_REPLAY_BLOCK_CACHE = {}


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
    parameter_policy: str = "all"
    ffn_down_scale: float = 1.0
    autoregressive_ul_weight: float = 0.0
    replay_format: str = "sft"
    autoregressive_prefixes: int = 2
    anchor_decay: float = 0.0


def trial_matrix(segment_tokens=4064):
    base = asdict(Trial(segment_tokens=segment_tokens))
    variants = [
        ("baseline", {}),
        ("eos4", {"eos_weight": 4.0}),
        ("eos16", {"eos_weight": 16.0}),
        ("eos64", {"eos_weight": 64.0}),
        (
            "causal_ar8_eos16",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "autoregressive_ul_weight": 0.1,
                "autoregressive_prefixes": 8,
                "eos_weight": 16.0,
            },
        ),
        ("anchor_decay001", {"anchor_decay": 0.001}),
        ("anchor_decay01", {"anchor_decay": 0.01}),
        ("anchor_decay05", {"anchor_decay": 0.05}),
        (
            "causal_ar8_anchor_decay01",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "autoregressive_ul_weight": 0.1,
                "autoregressive_prefixes": 8,
                "anchor_decay": 0.01,
            },
        ),
        (
            "causal_ar8_anchor_decay05",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "autoregressive_ul_weight": 0.1,
                "autoregressive_prefixes": 8,
                "anchor_decay": 0.05,
            },
        ),
        (
            "causal_replay_ar8",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "autoregressive_ul_weight": 0.1,
                "autoregressive_prefixes": 8,
            },
        ),
        (
            "causal_replay_ar8strong",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "autoregressive_ul_weight": 1.0,
                "autoregressive_prefixes": 8,
            },
        ),
        (
            "ar_ul10_broad",
            {"autoregressive_ul_weight": 10.0, "autoregressive_prefixes": 8},
        ),
        (
            "causal_replay8",
            {"replay_format": "causal", "replay_examples": 8, "replay_weight": 0.25},
        ),
        (
            "causal_replay32",
            {"replay_format": "causal", "replay_examples": 32, "replay_weight": 1.0},
        ),
        (
            "causal_replay8strong",
            {"replay_format": "causal", "replay_examples": 8, "replay_weight": 4.0},
        ),
        (
            "causal_replay_ar",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "autoregressive_ul_weight": 0.1,
            },
        ),
        (
            "causal_replay_anchor",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "kl_source": "durable",
                "kl_weight": 5.0,
            },
        ),
        (
            "causal_replay_anchor_ar",
            {
                "replay_format": "causal",
                "replay_examples": 8,
                "replay_weight": 0.25,
                "kl_source": "durable",
                "kl_weight": 5.0,
                "autoregressive_ul_weight": 0.1,
            },
        ),
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
        ("ar_ul1", {"autoregressive_ul_weight": 1.0}),
        ("ar_ul10", {"autoregressive_ul_weight": 10.0}),
        ("ar_ul100", {"autoregressive_ul_weight": 100.0}),
        (
            "ar_embeddings",
            {
                "autoregressive_ul_weight": 10.0,
                "parameter_policy": "embeddings",
                "lr_multiplier": 16.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        ("anti_10", {"anti_weight": 10.0}),
        ("anti_50", {"anti_weight": 50.0}),
        ("anti_100", {"anti_weight": 100.0}),
        (
            "anti_50_broad",
            {"anti_weight": 50.0, "elementary_only": False, "replay_examples": 32},
        ),
        (
            "anti_50_normalized",
            {
                "anti_weight": 50.0,
                "elementary_only": False,
                "replay_token_fraction": 0.25,
                "replay_weight": -1.0,
            },
        ),
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
        ("attention_only", {"parameter_policy": "attention_and_norm"}),
        (
            "embeddings_normalized64",
            {
                "parameter_policy": "embeddings",
                "lr_multiplier": 64.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "embeddings_normalized128",
            {
                "parameter_policy": "embeddings",
                "lr_multiplier": 128.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "attention_normalized1",
            {
                "parameter_policy": "attention_and_norm",
                "lr_multiplier": 1.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "attention_normalized4",
            {
                "parameter_policy": "attention_and_norm",
                "lr_multiplier": 4.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "attention_normalized8",
            {
                "parameter_policy": "attention_and_norm",
                "lr_multiplier": 8.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "attention_fast16",
            {"parameter_policy": "attention_and_norm", "lr_multiplier": 16.0},
        ),
        (
            "attention_fast128",
            {"parameter_policy": "attention_and_norm", "lr_multiplier": 128.0},
        ),
        (
            "attention_normalized",
            {
                "parameter_policy": "attention_and_norm",
                "lr_multiplier": 16.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "embeddings_normalized",
            {
                "parameter_policy": "embeddings",
                "lr_multiplier": 16.0,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        ("embeddings_only", {"parameter_policy": "embeddings"}),
        ("ffn_only", {"parameter_policy": "ffn"}),
        ("ffn_up_only", {"parameter_policy": "ffn_up"}),
        ("ffn_down_only", {"parameter_policy": "ffn_down"}),
        ("no_ffn_down", {"parameter_policy": "no_ffn_down"}),
        ("down_scale16", {"ffn_down_scale": 1 / 16}),
        ("down_scale64", {"ffn_down_scale": 1 / 64}),
        (
            "down_scale64_normalized",
            {
                "ffn_down_scale": 1 / 64,
                "elementary_only": False,
                "replay_examples": 32,
                "replay_weight": -1.0,
            },
        ),
        (
            "attention_anti50",
            {"parameter_policy": "attention_and_norm", "anti_weight": 50.0},
        ),
        ("embeddings_anti50", {"parameter_policy": "embeddings", "anti_weight": 50.0}),
    ]
    return [Trial(**{**base, **changes, "name": name}) for name, changes in variants]


def configure_trainable_parameters(model, policy):
    """Offline localization of drift; never mutates the source checkpoint.

    Freeze before creating AdamW groups so decoupled weight decay and moments
    cannot silently update excluded tensors. The tied head is an embedding.
    """
    policies = {
        "all",
        "ffn",
        "ffn_up",
        "ffn_down",
        "no_ffn_down",
        "embeddings",
        "attention_and_norm",
    }
    if policy not in policies:
        raise ValueError("unknown trainable parameter policy")
    counts = {"trainable": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        if ".ff." in name:
            group = "ffn_up" if ".ff.up." in name else "ffn_down"
        elif (
            "token_embedding" in name
            or "position_embedding" in name
            or name.startswith("lm_head.")
        ):
            group = "embeddings"
        else:
            group = "attention_and_norm"
        trainable = (
            policy == "all"
            or group == policy
            or (policy == "ffn" and group.startswith("ffn_"))
            or (policy == "no_ffn_down" and group != "ffn_down")
        )
        parameter.requires_grad_(trainable)
        counts["trainable" if trainable else "frozen"] += parameter.numel()
    if not counts["trainable"]:
        raise ValueError("parameter policy selected no tensors")
    return {"policy": policy, **counts}


def scale_ffn_down_groups(optimizer, model, scale):
    """Offline test of wide-FFN output sensitivity; up/attention rates unchanged."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("FFN down scale must be finite in (0, 1]")
    if scale == 1:
        return
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        down = [p for p in group["params"] if ".ff.down." in names[id(p)]]
        other = [p for p in group["params"] if ".ff.down." not in names[id(p)]]
        if other:
            groups.append({**group, "params": other})
        if down:
            groups.append(
                {
                    **group,
                    "params": down,
                    "group_name": "ffn_down",
                    "lr": group["lr"] * scale,
                    "lr_multiplier": group["lr_multiplier"] * scale,
                }
            )
    optimizer.param_groups[:] = groups


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


def slice_hash(blocks, replay_rows, replay_blocks=None):
    payload = {
        "causal_blocks": blocks,
        "replay_messages": [r.messages for r in replay_rows],
    }
    if replay_blocks:
        payload["protected_causal_replay_blocks"] = replay_blocks
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


def anchored_parameter_decay(model, anchor, strength):
    """Bounded proximal regularization to the SAME lineage's immutable anchor.

    This modifies only trained candidate tensors after the normal optimizer
    update. It never changes the anchor, never imports external weights, and
    never modifies counters. Guards must still accept the resulting candidate.
    """
    import torch

    if not math.isfinite(strength) or not 0 <= strength <= 0.05:
        raise ValueError("anchor decay strength must be finite in [0, .05]")
    current = dict(model.named_parameters())
    reference = dict(anchor.named_parameters())
    if set(current) != set(reference) or any(
        current[n].shape != reference[n].shape for n in current
    ):
        raise ValueError("anchor parameter topology mismatch")
    movement = distance_before = distance_after = 0.0
    with torch.no_grad():
        for name, parameter in current.items():
            if not parameter.requires_grad:
                continue
            target = (
                reference[name]
                .detach()
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            distance_before += float((parameter - target).square().sum())
            before = parameter.detach().clone()
            parameter.lerp_(target, strength)
            movement += float((parameter - before).square().sum())
            distance_after += float((parameter - target).square().sum())
    return {
        "strength": strength,
        "actual_parameter_movement_norm": math.sqrt(movement),
        "distance_to_anchor_before": math.sqrt(distance_before),
        "distance_to_anchor_after": math.sqrt(distance_after),
        "scope": "decoupled proximal parameter regularizer, not a replay gradient or live checkpoint replacement",
    }


def protected_causal_replay_blocks(root, runtime):
    """General language replay in the same causal format as held-out prediction.

    The saved foundation corpus is training-only and digest verified by the
    normal loader. No validation rows or external teachers are imported.
    """
    key = (str(Path(root).resolve()), runtime.config.context_length)
    if key not in _REPLAY_BLOCK_CACHE:
        protected = protected_bootstrap_texts()
        documents = [
            d
            for d in load_bootstrap_replay(Path(root) / "bootstrap-data").documents
            if not any(
                text and text in " ".join(d.text.casefold().split())
                for text in protected
            )
        ]
        blocks = pack_causal_blocks(
            documents, runtime.tokenizer, context_length=runtime.config.context_length
        )
        if not blocks:
            raise ValueError("no protected causal replay blocks")
        _REPLAY_BLOCK_CACHE[key] = blocks
    return _REPLAY_BLOCK_CACHE[key]


def causal_replay_selection(blocks, *, rng, examples, token_fraction, causal_tokens):
    from .tokenizer import PAD

    if token_fraction is not None and not 0 < token_fraction < 1:
        raise ValueError("actual replay token fraction must be between zero and one")
    selected = []
    count = 0
    target = (
        causal_tokens * token_fraction / (1 - token_fraction)
        if token_fraction is not None
        else None
    )
    order = list(range(len(blocks)))
    while count < target if target is not None else len(selected) < examples:
        rng.shuffle(order)
        for index in order:
            needed = sum(t not in (PAD, -100) for t in blocks[index][1:])
            if not needed:
                continue
            selected.append(index)
            count += needed
            if count >= target if target is not None else len(selected) >= examples:
                break
        if not selected:
            raise ValueError("causal replay has no supervised labels")
    if count == 0:
        raise ValueError("causal replay budget must be positive")
    return selected, count


def generated_repetition_objective(logits, input_ids, *, prompt_tokens, ngram=4):
    """Negative-only loss on repeated generated ngrams; no pseudo gold labels.

    Callers supply ONLY continuations of training prefixes, never verifier
    prompts. Structural tokens and the prompt itself receive no negative label.
    Auxiliary labels are not counted as new supervised causal training tokens.
    """
    import torch
    from .tokenizer import BYTE_OFFSET

    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("unexpected generated objective tensor shapes")
    if ngram < 2 or prompt_tokens < 1 or prompt_tokens > input_ids.shape[1]:
        raise ValueError("invalid generated objective prompt/ngram length")
    coordinates = []
    for row_index, row in enumerate(input_ids.detach().cpu().tolist()):
        seen = set()
        for end in range(ngram - 1, len(row)):
            gram = tuple(row[end - ngram + 1 : end + 1])
            if (
                end >= prompt_tokens
                and gram in seen
                and all(t >= BYTE_OFFSET for t in gram)
            ):
                coordinates.append((row_index, end - 1, row[end]))
            seen.add(gram)
    if not coordinates:
        return logits.sum() * 0.0, {
            "negative_positions": 0,
            "generated_tokens": input_ids.shape[0]
            * (input_ids.shape[1] - prompt_tokens),
        }
    probs = logits.softmax(-1)
    negative_probs = torch.stack([probs[b, t, v] for b, t, v in coordinates]).clamp(
        max=1 - 1e-6
    )
    return -torch.log1p(-negative_probs).mean(), {
        "negative_positions": len(coordinates),
        "generated_tokens": input_ids.shape[0] * (input_ids.shape[1] - prompt_tokens),
    }


def training_prefix_generated_objective(runtime, prefixes):
    """Generate with unchanged greedy decoding, then differentiate negatives."""
    import torch

    original_mode = runtime.model.training
    terms = []
    stats = {
        "negative_positions": 0,
        "generated_tokens": 0,
        "holdout_filtered_trajectories": 0,
        "prefix_ids_sha256": slice_hash([list(p) for p in prefixes], []),
    }
    try:
        for prompt in prefixes:
            generated = runtime._generate_ids(
                list(prompt), max_new_tokens=40, temperature=0.0, repetition_penalty=1.0
            )
            stats["generated_tokens"] += len(generated)
            normalized = " ".join(
                runtime.tokenizer.decode(list(prompt) + generated).casefold().split()
            )
            if any(text and text in normalized for text in protected_bootstrap_texts()):
                stats["holdout_filtered_trajectories"] += 1
                continue
            ids = torch.tensor([list(prompt) + generated], device=runtime.device)
            runtime.model.train(original_mode)
            logits = runtime.model(ids)["logits"]
            loss, report = generated_repetition_objective(
                logits, ids, prompt_tokens=len(prompt)
            )
            terms.append(loss * report["negative_positions"])
            stats["negative_positions"] += report["negative_positions"]
    finally:
        runtime.model.train(original_mode)
    if not terms:
        zero = (
            next(p for p in runtime.model.parameters() if p.requires_grad).sum() * 0.0
        )
        return zero, stats
    return sum(terms) / max(1, stats["negative_positions"]), stats


def read_split_documents(path, split):
    documents = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("split") != split:
            raise ValueError(f"document input requires explicit {split} split")
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("document hash mismatch")
        documents.append(
            CorpusDocument(
                **{k: row[k] for k in ("source", "text", "sha256", "bytes", "domain")}
            )
        )
    if not documents:
        raise ValueError("empty document input")
    return documents


def exclude_corpus_validation_replay(rows, validation_documents):
    """Corpus and SFT split hashes differ: exclude held-out content across formats."""
    validation_texts = {
        " ".join(d.text.casefold().split()) for d in validation_documents
    }
    kept = [
        row
        for row in rows
        if not any(
            " ".join(m["content"].casefold().split()) in validation_texts
            for m in row.messages
        )
    ]
    return kept, len(rows) - len(kept)


def require_disjoint_documents(training, validation):
    if {d.sha256 for d in training} & {d.sha256 for d in validation}:
        raise ValueError("training/validation document overlap")


def evaluate_corpus_subset(runtime, documents, max_blocks=32):
    """Fixed held-out subset loss, never feeds labels to an optimizer.

    This is not the original whole-corpus validation reported by live telemetry.
    The selected block hash and number of labels describe the measurement.
    """
    import torch
    from torch.nn import functional as F

    blocks = pack_causal_blocks(
        documents, runtime.tokenizer, context_length=runtime.config.context_length
    )
    if not blocks:
        raise ValueError("empty validation blocks")
    if len(blocks) > max_blocks:
        rng = random.Random(1789)
        blocks = [blocks[i] for i in sorted(rng.sample(range(len(blocks)), max_blocks))]
    previous = runtime.model.training
    runtime.model.eval()
    total = 0.0
    labels_count = 0
    try:
        with torch.no_grad():
            for start in range(0, len(blocks), 2):
                ids, labels = _batch(
                    blocks,
                    list(range(start, min(start + 2, len(blocks)))),
                    device=runtime.device,
                )
                logits = runtime.model(ids)["logits"][:, :-1, :]
                targets = labels[:, 1:]
                total += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                )
                labels_count += int((targets != -100).sum())
    finally:
        runtime.model.train(previous)
    return {
        "loss": total / max(1, labels_count),
        "supervised_tokens": labels_count,
        "block_sha256": slice_hash(blocks, []),
        "scope": "fixed recovered corpus holdout subset, not full live validation",
    }


def run_trial(
    root,
    trial,
    *,
    seed,
    causal_documents=None,
    validation_documents=None,
    measure_gradients=False,
    initial_checkpoint=None,
    export_checkpoint=None,
    prior_accepted_tokens=0,
):
    import torch

    root = Path(root)
    b = root / "bootstrap-data"
    started = time.perf_counter()
    torch.manual_seed(seed)
    initial_checkpoint = (
        Path(initial_checkpoint) if initial_checkpoint is not None else b / "candidate"
    )
    if prior_accepted_tokens < 0:
        raise ValueError("negative offline sequence accepted-token offset")
    if initial_checkpoint != b / "candidate" and trial.optimizer_state == "resume":
        raise ValueError(
            "source optimizer moments do not belong to an exported sequence checkpoint"
        )
    if export_checkpoint is not None:
        export_checkpoint = Path(export_checkpoint)
        validate_output_path(root, export_checkpoint)
        validate_output_path(initial_checkpoint, export_checkpoint)
        if export_checkpoint.exists():
            raise ValueError(
                "offline candidate export must not overwrite an existing checkpoint"
            )
    runtime = GeneralistRuntime.from_checkpoint(initial_checkpoint, device="cpu")
    p = read_json(b / "progress.json")
    p["tokens_processed"] += prior_accepted_tokens
    anchor = read_json(b / "language-guard.json")["report"]
    if trial.kl_source == "durable" or trial.anchor_decay:

        def tokenizer_spec(path):
            token_file = path / "tokenizer.json"
            return json.loads(token_file.read_text()) if token_file.exists() else None

        if tokenizer_spec(initial_checkpoint) != tokenizer_spec(
            b / "language-guard-best"
        ):
            raise ValueError(
                "durable anchor tokenizer differs from training checkpoint"
            )
    before = evaluate_phase5_language(runtime)
    validation_before = None
    if validation_documents is not None:
        training_documents = (
            causal_documents
            if causal_documents is not None
            else load_bootstrap_replay(b).documents
        )
        require_disjoint_documents(training_documents, validation_documents)
        validation_before = evaluate_corpus_subset(runtime, validation_documents)
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
    corpus_validation_replay_excluded = 0
    if validation_documents is not None:
        rows, corpus_validation_replay_excluded = exclude_corpus_validation_replay(
            rows, validation_documents
        )
        if not rows and trial.replay_format == "sft":
            raise ValueError("no validation-disjoint SFT replay")
    initial_digest = checkpoint_digest(initial_checkpoint)
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
    parameter_anchor = None
    if trial.anchor_decay:
        parameter_anchor = (
            reference
            if trial.kl_source == "durable"
            else GeneralistRuntime.from_checkpoint(
                b / "language-guard-best", device="cpu"
            ).model
        )
        parameter_anchor.eval()
        for parameter in parameter_anchor.parameters():
            parameter.requires_grad_(False)
    if trial.optimizer_state == "resume" and (
        trial.parameter_policy != "all" or trial.ffn_down_scale != 1
    ):
        raise ValueError("live moments cannot resume with a changed parameter topology")
    parameter_report = configure_trainable_parameters(
        runtime.model, trial.parameter_policy
    )
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
    scale_ffn_down_groups(optimizer, runtime.model, trial.ffn_down_scale)
    optimizer_report["ffn_down_scale"] = trial.ffn_down_scale
    attempted = replay_tokens = 0
    steps = []
    selected_blocks = []
    selected_rows = []
    selected_replay_blocks = []
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
        replay_rng = random.Random(seed + 100000 + step)
        replay_batches = []
        if trial.replay_format == "sft":
            replay_indices, replay_count = replay_selection(
                runtime,
                rows,
                rng=replay_rng,
                examples=trial.replay_examples,
                token_fraction=trial.replay_token_fraction,
                causal_tokens=causal_count,
            )
            selected_rows.extend(rows[i] for i in replay_indices)
            for offset in range(0, len(replay_indices), 2):
                ids, labels = _sft_training_batch(
                    runtime, rows, replay_indices[offset : offset + 2]
                )
                replay_batches.append((ids, labels, int((labels[:, 1:] != -100).sum())))
        elif trial.replay_format == "causal":
            replay_blocks = protected_causal_replay_blocks(root, runtime)
            replay_indices, replay_count = causal_replay_selection(
                replay_blocks,
                rng=replay_rng,
                examples=trial.replay_examples,
                token_fraction=trial.replay_token_fraction,
                causal_tokens=causal_count,
            )
            selected_replay_blocks.extend(replay_blocks[i] for i in replay_indices)
            for offset in range(0, len(replay_indices), 2):
                ids, labels = _batch(
                    replay_blocks,
                    replay_indices[offset : offset + 2],
                    device=runtime.device,
                )
                replay_batches.append((ids, labels, int((labels[:, 1:] != -100).sum())))
        else:
            raise ValueError("unknown protected replay format")
        weight = (
            trial.replay_weight
            if trial.replay_weight >= 0
            else replay_count / causal_count
        )
        if trial.gradient_balance:
            optimizer.zero_grad(set_to_none=True)
        replay_loss = kl_value = 0.0
        kl_count = 0
        for ids, labels, count in replay_batches:
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
        ar_report = None
        if trial.autoregressive_ul_weight > 0:
            auxiliary_start = time.perf_counter()
            previous_grads = grad_vector(runtime.model) if measure_gradients else None
            instrument_seconds += time.perf_counter() - auxiliary_start
            if not 1 <= trial.autoregressive_prefixes <= 32:
                raise ValueError("generated prefix count must be in [1, 32]")
            prefixes = [row[:8].tolist() for ids, _, _ in prepared for row in ids][
                : trial.autoregressive_prefixes
            ]
            ar_loss, ar_report = training_prefix_generated_objective(runtime, prefixes)
            (trial.autoregressive_ul_weight * ar_loss).backward()
            ar_report["raw_loss"] = float(ar_loss.detach())
            ar_report["weight"] = trial.autoregressive_ul_weight
            ar_report["new_supervised_causal_tokens_counted"] = 0
            if previous_grads is not None:
                auxiliary_start = time.perf_counter()
                current_grads = grad_vector(runtime.model)
                auxiliary_grads = {
                    k: current_grads[k] - previous_grads[k] for k in current_grads
                }
                ar_report["gradient_pressure"] = grad_relation(
                    previous_grads, auxiliary_grads
                )
                ar_report["gradient_pressure"][
                    "scope"
                ] = "previous combined gradient versus weighted generated negative-only objective"
                instrument_seconds += time.perf_counter() - auxiliary_start
                del previous_grads, current_grads, auxiliary_grads
        norm = float(
            torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), trial.clip_norm)
        )
        if not math.isfinite(norm):
            raise ValueError("non-finite gradient")
        before_update = (
            {
                n: p.detach().clone()
                for n, p in runtime.model.named_parameters()
                if p.requires_grad
            }
            if parameter_anchor is not None
            else None
        )
        optimizer.step()
        anchor_report = None
        if parameter_anchor is not None:
            update_norm = math.sqrt(
                sum(
                    float((p.detach() - before_update[n]).square().sum())
                    for n, p in runtime.model.named_parameters()
                    if p.requires_grad
                )
            )
            anchor_report = anchored_parameter_decay(
                runtime.model, parameter_anchor, trial.anchor_decay
            )
            anchor_report["optimizer_update_norm"] = update_norm
            del before_update
        attempted += causal_count
        replay_tokens += replay_count
        event = {
            "step": step,
            "effective_lr": lr,
            "effective_group_lrs": {
                g["group_name"]: g["lr"] for g in optimizer.param_groups
            },
            "causal_tokens": causal_count,
            "replay_tokens": replay_count,
            "replay_weight": weight,
            "causal_loss": causal_loss,
            "replay_loss": replay_loss,
            "kl_loss": kl_value,
            "kl_tokens": kl_count,
            "combined_gradient_norm_before_clipping": norm,
            "gradient_pressure": pressure,
            "autoregressive_unlikelihood": ar_report,
            "anchored_parameter_decay": anchor_report,
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
    validation_after = (
        evaluate_corpus_subset(runtime, validation_documents)
        if validation_documents is not None
        else None
    )
    persist_started = time.perf_counter()
    if accepted:
        with tempfile.TemporaryDirectory(prefix="airi-benchmark-checkpoint-") as tmp:
            runtime.save_checkpoint(
                Path(tmp) / "candidate", metadata={"benchmark_only": True}
            )
            _save_optimizer_checkpoint(optimizer, Path(tmp) / "optimizer.pt")
            if export_checkpoint is not None:
                import shutil

                shutil.copytree(Path(tmp) / "candidate", export_checkpoint)
    if checkpoint_digest(initial_checkpoint) != initial_digest:
        raise ValueError("offline trial modified its initial checkpoint")
    persist_seconds = time.perf_counter() - persist_started if accepted else 0.0
    total_seconds = time.perf_counter() - started
    return {
        "schema": 1,
        "trial": asdict(trial),
        "seed": seed,
        "checkpoint_hash": initial_digest,
        "offline_sequence_prior_accepted_tokens": prior_accepted_tokens,
        "exported_checkpoint_hash": (
            checkpoint_digest(export_checkpoint)
            if accepted and export_checkpoint is not None
            else None
        ),
        "architecture": runtime.config.to_dict(),
        "dataset_slice_hash": slice_hash(
            selected_blocks, selected_rows, selected_replay_blocks
        ),
        "causal_slice_hash": slice_hash(selected_blocks, []),
        "data_scope": (
            "provided training-only documents"
            if causal_documents is not None
            else "persisted training-only replay sample; not the full live corpus"
        ),
        "corpus_subset_validation_before": validation_before,
        "corpus_subset_validation_after": validation_after,
        "optimizer": optimizer_report,
        "trainable_parameters": parameter_report,
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
        "corpus_validation_replay_excluded": corpus_validation_replay_excluded,
        "torch_version": torch.__version__,
    }


def run_sequence(
    root,
    trial,
    sizes,
    *,
    seed,
    causal_documents=None,
    validation_documents=None,
    measure_gradients=False,
    on_segment=None,
):
    """Verify sustained OFFLINE accepted updates; rejects retain the prior model.

    All strategies start from the pinned source. This temporary candidate is
    never a live lineage, never writes progress/state, and is discarded after
    the experiment. Interrupted sequences restart reproducibly from the source;
    an optional callback can durably record completed segments.
    """
    from dataclasses import replace

    root = Path(root)
    original = source_digest(root)
    current = root / "bootstrap-data/candidate"
    rows = []
    accepted_tokens = 0
    with tempfile.TemporaryDirectory(prefix="airi-offline-sequence-") as tmp:
        for index, size in enumerate(sizes):
            exported = Path(tmp) / str(index) / "candidate"
            result = run_trial(
                root,
                replace(trial, segment_tokens=size),
                seed=seed + index * 10000,
                causal_documents=causal_documents,
                validation_documents=validation_documents,
                measure_gradients=measure_gradients,
                initial_checkpoint=current,
                export_checkpoint=exported,
                prior_accepted_tokens=accepted_tokens,
            )
            result["sequence_index"] = index
            result["sequence_initial_checkpoint_hash"] = checkpoint_digest(
                root / "bootstrap-data/candidate"
            )
            if result["accepted"]:
                current = exported
                accepted_tokens += result["accepted_equivalent_tokens"]
            result["sequence_accepted_tokens"] = accepted_tokens
            rows.append(result)
            if source_digest(root) != original:
                raise ValueError("offline sequence modified pinned source")
            if on_segment is not None:
                on_segment(result)
    return {
        "schema": 1,
        "scope": "temporary offline candidate sequence; not live persistent learning",
        "segments": rows,
        "accepted_equivalent_tokens": accepted_tokens,
        "live_tokens_persisted": 0,
        "source_state_modified": False,
        "success_claim_allowed": False,
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
    parser.add_argument(
        "--validation-documents",
        help="separate hash-verified recovered holdout JSONL; reported as subset validation",
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
        documents = read_split_documents(args.causal_documents, "train")
    validation_documents = None
    if args.validation_documents:
        validation_documents = read_split_documents(
            args.validation_documents, "validation"
        )
        if documents is not None:
            require_disjoint_documents(documents, validation_documents)
        dataset_input_hash = hashlib.sha256(
            (
                dataset_input_hash
                + hashlib.sha256(
                    Path(args.validation_documents).read_bytes()
                ).hexdigest()
            ).encode()
        ).hexdigest()
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
                    validation_documents=validation_documents,
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
