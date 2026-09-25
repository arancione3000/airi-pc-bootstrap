from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from .bootstrap_data import load_bootstrap_replay
from .bootstrap_training import (
    _anti_collapse_weights,
    _elementary_rehabilitation_rows,
    _filter_protected_replay,
    _learning_rate,
    _load_optimizer_checkpoint,
    _protected_causal_replay_rows,
    _protected_continual_plan,
    _protected_optimizer,
    _segment_language_gate,
    _sft_row_fingerprint,
    _sft_training_batch,
)
from .phase5_diagnostics import evaluate_phase5_language
from .pretraining import _batch as causal_batch, pack_causal_blocks
from .runtime import GeneralistRuntime
from .tokenizer import BYTE_OFFSET
from .training import causal_training_objective, reference_kl_loss


def _mass_unlikelihood_objective(
    logits,
    labels,
    input_ids,
    *,
    eos_loss_weight: float,
    repetition_unlikelihood_weight: float,
    repetition_window: int = 16,
):
    """Experimental sequence-aware UL used only by this read-only ablation.

    The production objective averages over every individual negative token.
    This variant instead penalizes the total probability mass assigned to any
    recently seen non-target token at each supervised position, so the signal
    does not vanish as the negative set grows.
    """
    import torch
    from torch.nn import functional as F

    base, stats = causal_training_objective(
        logits,
        labels,
        input_ids,
        eos_loss_weight=eos_loss_weight,
        repetition_unlikelihood_weight=0.0,
        repetition_window=repetition_window,
    )
    weight = max(0.0, float(repetition_unlikelihood_weight))
    if weight <= 0.0:
        return base, stats

    shifted_logits = logits[:, :-1, :]
    targets = labels[:, 1:]
    valid = targets != -100
    window = max(1, min(int(repetition_window), int(input_ids.shape[1])))
    negative_mask = torch.zeros_like(shifted_logits, dtype=torch.bool)
    for position in range(int(shifted_logits.shape[1])):
        start = max(0, position - window + 1)
        recent = input_ids[:, start:position + 1]
        negative_mask[:, position, :].scatter_(1, recent, True)
    floor = max(0, min(int(BYTE_OFFSET), int(shifted_logits.shape[-1])))
    if floor:
        negative_mask[:, :, :floor] = False
    safe_targets = targets.clamp_min(0).unsqueeze(-1)
    negative_mask.scatter_(2, safe_targets, False)
    negative_mask &= valid.unsqueeze(-1)

    probs = torch.softmax(shifted_logits, dim=-1)
    negative_mass = (probs * negative_mask.to(probs.dtype)).sum(dim=-1)
    negative_mass = negative_mass.clamp(min=0.0, max=1.0 - 1e-6)
    if int(valid.sum().item()) > 0:
        ul_loss = -torch.log1p(-negative_mass).masked_select(valid).mean()
    else:
        ul_loss = shifted_logits.sum() * 0.0
    total = base + weight * ul_loss
    return total, {
        **stats,
        "repetition_unlikelihood_loss": float(ul_loss.detach().cpu()),
        "repetition_unlikelihood_weight": weight,
        "repetition_unlikelihood_aggregation": "recent_probability_mass",
        "repetition_negative_count": int(negative_mask.sum().item()),
    }


def _training_objective(
    logits,
    labels,
    ids,
    *,
    eos_loss_weight: float,
    repetition_unlikelihood_weight: float,
    repetition_window: int,
    aggregation: str,
):
    if aggregation == "recent_probability_mass":
        return _mass_unlikelihood_objective(
            logits,
            labels,
            ids,
            eos_loss_weight=eos_loss_weight,
            repetition_unlikelihood_weight=repetition_unlikelihood_weight,
            repetition_window=repetition_window,
        )
    return causal_training_objective(
        logits,
        labels,
        ids,
        eos_loss_weight=eos_loss_weight,
        repetition_unlikelihood_weight=repetition_unlikelihood_weight,
        repetition_window=repetition_window,
    )


def _group_for_name(name: str) -> str:
    if ".ff." in name:
        return "ffn"
    if name.startswith("lm_head."):
        return "lm_head"
    if "token_embedding" in name or "position_embedding" in name:
        return "embeddings"
    return "attention_and_norm"


def _group_gradient_stats(model) -> dict[str, dict[str, float]]:
    import torch

    sums: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _group_for_name(name)
        grad = parameter.grad.detach().float()
        sums[group] = sums.get(group, 0.0) + float(torch.sum(grad * grad).item())
        max_abs[group] = max(max_abs.get(group, 0.0), float(grad.abs().max().item()))
        counts[group] = counts.get(group, 0) + int(grad.numel())
    return {
        group: {
            "l2_norm": math.sqrt(max(0.0, sums[group])),
            "max_abs": max_abs.get(group, 0.0),
            "coordinates": counts.get(group, 0),
        }
        for group in sorted(sums)
    }


def _group_update_stats(model, reference_model) -> dict[str, dict[str, float]]:
    import torch

    ref = dict(reference_model.named_parameters())
    update_sq: dict[str, float] = {}
    weight_sq: dict[str, float] = {}
    coordinates: dict[str, int] = {}
    max_abs: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name not in ref:
            continue
        group = _group_for_name(name)
        current = parameter.detach().float()
        base = ref[name].detach().float()
        delta = current - base
        update_sq[group] = update_sq.get(group, 0.0) + float(torch.sum(delta * delta).item())
        weight_sq[group] = weight_sq.get(group, 0.0) + float(torch.sum(base * base).item())
        coordinates[group] = coordinates.get(group, 0) + int(delta.numel())
        max_abs[group] = max(max_abs.get(group, 0.0), float(delta.abs().max().item()))
    out: dict[str, dict[str, float]] = {}
    for group in sorted(update_sq):
        update_norm = math.sqrt(max(0.0, update_sq[group]))
        weight_norm = math.sqrt(max(0.0, weight_sq[group]))
        out[group] = {
            "update_l2_norm": update_norm,
            "weight_l2_norm": weight_norm,
            "update_to_weight": update_norm / max(1e-12, weight_norm),
            "max_abs_update": max_abs.get(group, 0.0),
            "coordinates": coordinates.get(group, 0),
        }
    return out


def _activation_probe(model, ids, baseline_activations=None) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    activations: list[Any] = [None] * len(model.blocks)
    handles = []
    for index, block in enumerate(model.blocks):
        def hook(_module, _inputs, output, index=index):
            activations[index] = output.detach().float().cpu()
        handles.append(block.ff.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        logits = model(ids)["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
        top1 = probs.max(dim=-1).values.mean()
        top5 = probs.topk(min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1).mean()

    for handle in handles:
        handle.remove()

    layers = []
    for index, tensor in enumerate(activations):
        if tensor is None:
            continue
        row = {
            "layer": index,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "variance": float(torch.var(tensor, unbiased=False).item()),
        }
        if baseline_activations is not None and index < len(baseline_activations):
            baseline = baseline_activations[index]
            if baseline is not None:
                row["cosine_to_baseline"] = float(
                    F.cosine_similarity(
                        tensor.reshape(1, -1),
                        baseline.reshape(1, -1),
                        dim=-1,
                    ).item()
                )
        layers.append(row)

    return {
        "logit_entropy": float(entropy.item()),
        "mean_top1_probability": float(top1.item()),
        "mean_top5_mass": float(top5.item()),
        "ff_layers": layers,
        "_activations": activations,
    }


def _optimizer_state_summary(optimizer) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name") or "unknown")
        exp_avg_sq = 0.0
        exp_avg2_sq = 0.0
        tensor_count = 0
        coordinate_count = 0
        for parameter in group["params"]:
            state = optimizer.state.get(parameter) or {}
            exp_avg = state.get("exp_avg")
            exp_avg_sq_tensor = state.get("exp_avg_sq")
            if isinstance(exp_avg, torch.Tensor):
                value = exp_avg.detach().float()
                exp_avg_sq += float(torch.sum(value * value).item())
                coordinate_count += int(value.numel())
            if isinstance(exp_avg_sq_tensor, torch.Tensor):
                value2 = exp_avg_sq_tensor.detach().float()
                exp_avg2_sq += float(torch.sum(value2 * value2).item())
                tensor_count += 1
        out[name] = {
            "exp_avg_l2_norm": math.sqrt(max(0.0, exp_avg_sq)),
            "exp_avg_sq_l2_norm": math.sqrt(max(0.0, exp_avg2_sq)),
            "state_tensors": tensor_count,
            "state_coordinates": coordinate_count,
        }
    return out


def _set_optimizer_policy(optimizer, *, learning_rate: float, multipliers: dict[str, float]) -> None:
    for group in optimizer.param_groups:
        name = str(group.get("group_name") or "unknown")
        multiplier = float(multipliers.get(name, group.get("lr_multiplier", 1.0) or 1.0))
        group["lr_multiplier"] = multiplier
        group["lr"] = float(learning_rate) * multiplier


def _clear_group_state(optimizer, group_name: str) -> int:
    cleared = 0
    for group in optimizer.param_groups:
        if str(group.get("group_name") or "") != group_name:
            continue
        for parameter in group["params"]:
            if parameter in optimizer.state:
                optimizer.state.pop(parameter, None)
                cleared += 1
    return cleared


def _damp_optimizer_state(optimizer, factor: float) -> int:
    import torch

    touched = 0
    scale = max(0.0, min(1.0, float(factor)))
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                value.mul_(scale)
                touched += 1
    return touched


def _group_clip(optimizer, caps: dict[str, float] | None) -> dict[str, float]:
    import torch

    if not caps:
        return {}
    norms: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name") or "unknown")
        cap = caps.get(name)
        if cap is None:
            continue
        value = torch.nn.utils.clip_grad_norm_(group["params"], float(cap))
        norms[name] = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
    return norms


def _gradient_pressure(
    runtime,
    reference_model,
    blocks,
    protected_rows,
    before,
    protected_plan,
) -> dict[str, Any]:
    import torch

    model = runtime.model
    model.train()
    ids, labels = causal_batch(blocks, [0, 1], device=runtime.device)
    anti_weight, eos_weight = _anti_collapse_weights("B_short_sentence_completion", before)
    anti_weight = max(float(anti_weight), float(protected_plan["anti_repetition_weight"]))

    model.zero_grad(set_to_none=True)
    result = model(ids)
    causal_loss, _ = causal_training_objective(
        result["logits"],
        labels,
        ids,
        eos_loss_weight=eos_weight,
        repetition_unlikelihood_weight=anti_weight,
        repetition_window=16,
    )
    causal_loss.backward()
    causal_gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    causal_stats = _group_gradient_stats(model)

    model.zero_grad(set_to_none=True)
    replay_ids, replay_labels = _sft_training_batch(runtime, protected_rows, [0, 1])
    replay_result = model(replay_ids)
    replay_loss, _ = causal_training_objective(
        replay_result["logits"],
        replay_labels,
        replay_ids,
        eos_loss_weight=1.0,
        repetition_unlikelihood_weight=anti_weight,
        repetition_window=16,
    )
    with torch.no_grad():
        reference_logits = reference_model(replay_ids)["logits"]
    kl_loss, _ = reference_kl_loss(
        replay_result["logits"],
        reference_logits,
        replay_labels,
        temperature=float(protected_plan["reference_kl_temperature"]),
    )
    protected_loss = (
        float(protected_plan["replay_loss_weight"]) * replay_loss
        + float(protected_plan["reference_kl_weight"]) * kl_loss
    )
    protected_loss.backward()
    replay_stats = _group_gradient_stats(model)

    dots: dict[str, float] = {}
    c2: dict[str, float] = {}
    r2: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        replay_grad = parameter.grad
        causal_grad = causal_gradients.get(name)
        if replay_grad is None or causal_grad is None:
            continue
        group = _group_for_name(name)
        replay_value = replay_grad.detach().float()
        dots[group] = dots.get(group, 0.0) + float(torch.sum(causal_grad * replay_value).item())
        c2[group] = c2.get(group, 0.0) + float(torch.sum(causal_grad * causal_grad).item())
        r2[group] = r2.get(group, 0.0) + float(torch.sum(replay_value * replay_value).item())

    cosine = {
        group: dots[group] / max(1e-12, math.sqrt(c2[group]) * math.sqrt(r2[group]))
        for group in sorted(dots)
    }
    model.zero_grad(set_to_none=True)
    del causal_gradients
    return {
        "causal_loss": float(causal_loss.detach().cpu()),
        "protected_loss": float(protected_loss.detach().cpu()),
        "causal_group_gradients": causal_stats,
        "protected_group_gradients": replay_stats,
        "group_gradient_cosine": cosine,
    }


def run_ablation(state_dir: str | Path) -> dict[str, Any]:
    import torch

    torch.manual_seed(7)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    state = Path(state_dir).expanduser().resolve()
    bootstrap = state / "bootstrap-data"
    candidate = bootstrap / "candidate"
    optimizer_path = bootstrap / "optimizer.pt"
    progress = json.loads((bootstrap / "progress.json").read_text(encoding="utf-8"))
    guard = json.loads((bootstrap / "language-guard.json").read_text(encoding="utf-8"))
    anchor = dict(guard.get("report") or {})
    if not anchor:
        raise RuntimeError("missing durable language guard anchor")

    baseline = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    before = evaluate_phase5_language(baseline)
    replay = load_bootstrap_replay(bootstrap)
    if not replay.documents or not replay.sft_train:
        raise RuntimeError("persisted training-only replay is unavailable")

    short_documents = [row for row in replay.documents if len(row.text) <= 220] or list(replay.documents)
    blocks = pack_causal_blocks(
        short_documents,
        baseline.tokenizer,
        context_length=baseline.config.context_length,
    )
    if len(blocks) < 64:
        raise RuntimeError("insufficient persisted replay blocks for ablation")

    protected_rows, protected_counts = _protected_causal_replay_rows(
        replay.sft_train,
        heldout_sft=(),
    )
    if len(protected_rows) < 2:
        raise RuntimeError("insufficient protected replay rows")

    rejected = int(progress.get("segment_guard_consecutive_rejections", 0) or 0)
    success_streak = int(progress.get("segment_guard_success_streak", 0) or 0)
    base_scale = float(progress.get("segment_guard_lr_scale", 1.0 / 64.0) or 1.0 / 64.0)
    base_scale = max(1.0 / 64.0, min(1.0, base_scale))
    target_tokens = int(progress.get("target_tokens", 100_000_000) or 100_000_000)
    processed_tokens = int(progress.get("tokens_processed", 0) or 0)
    start_step = int(progress.get("steps", 0) or 0)
    retry_seed_offset = rejected * 1_000_003
    protected_plan = _protected_continual_plan(
        before,
        anchor,
        consecutive_rejections=rejected,
        success_streak=success_streak,
    )

    reference = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)

    probe_ids, _ = causal_batch(blocks, [0, 1], device="cpu")
    baseline_probe = _activation_probe(reference.model, probe_ids)
    baseline_activations = baseline_probe.pop("_activations")

    pressure_runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    pressure = _gradient_pressure(
        pressure_runtime,
        reference.model,
        blocks,
        protected_rows,
        before,
        protected_plan,
    )
    del pressure_runtime

    elementary_source = []
    for stage in ("R1_bilingual_foundations", "R2_simple_responses", "R3_short_dialogue"):
        stage_rows = _elementary_rehabilitation_rows()[stage]
        elementary_source.extend(stage_rows["it"])
        elementary_source.extend(stage_rows["en"])
    elementary_safe, _ = _filter_protected_replay(elementary_source, heldout_sft=())
    elementary_fingerprints = {
        _sft_row_fingerprint(row) for row in elementary_safe
    }
    elementary_indices = [
        index for index, row in enumerate(protected_rows)
        if _sft_row_fingerprint(row) in elementary_fingerprints
    ]
    general_indices = [
        index for index, row in enumerate(protected_rows)
        if _sft_row_fingerprint(row) not in elementary_fingerprints
    ]
    if not elementary_indices or not general_indices:
        raise RuntimeError("replay ablation requires elementary and general replay pools")

    seed_indices = [0, 1, 2, 3, 4, 5, 6, 7, 16, 24, 32, 38, 39, 40, 46, 63]
    validation_seed_indices = [
        max(0, rejected - 6),
        max(0, rejected - 3),
        rejected,
        rejected + 1,
        rejected + 2,
        rejected + 3,
        rejected + 4,
        rejected + 6,
        rejected + 8,
        rejected + 12,
        rejected + 16,
        rejected + 24,
    ]
    variants = [
        {
            "name": f"BASE1_SEED_{seed_index}",
            "optimizer_state": "reset",
            "steps": 1,
            "retry_rejection_index": seed_index,
            "replay_batch_size": 2,
            "replay_sampling": "uniform",
            "replay_loss_weight": 0.25,
        }
        for seed_index in validation_seed_indices
    ]
    variants.extend([
        {
            "name": f"ELEM1_W400_SEED_{seed_index}",
            "optimizer_state": "reset",
            "steps": 1,
            "retry_rejection_index": seed_index,
            "replay_batch_size": 2,
            "replay_sampling": "elementary",
            "replay_loss_weight": 4.00,
        }
        for seed_index in validation_seed_indices
    ])

    results = []
    for variant in variants:
        runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
        model = runtime.model
        model.train()

        requested_scale = base_scale * float(variant.get("lr_factor", 1.0))
        effective_scale = max(1.0 / 2048.0, requested_scale) if requested_scale > 0.0 else 0.0
        optimizer, optimizer_report = _protected_optimizer(
            model,
            learning_rate=3e-4 * effective_scale,
        )
        state_mode = str(variant["optimizer_state"])
        storage = None
        if state_mode != "reset":
            storage = _load_optimizer_checkpoint(optimizer, optimizer_path)

        if state_mode == "selective_ffn_reset":
            _clear_group_state(optimizer, "ffn")
        elif state_mode == "damped":
            _damp_optimizer_state(optimizer, 0.25)

        multipliers = {
            "embeddings": float(variant.get("embedding_multiplier", 0.25)),
            "attention_and_norm": float(variant.get("attention_multiplier", 0.35)),
            "ffn": float(variant.get("ffn_multiplier", 1.0)),
            "lm_head": float(variant.get("embedding_multiplier", 0.25)),
        }
        _set_optimizer_policy(
            optimizer,
            learning_rate=3e-4 * effective_scale,
            multipliers=multipliers,
        )
        optimizer_before = _optimizer_state_summary(optimizer)

        replay_loss_weight = float(
            variant.get("replay_loss_weight", protected_plan["replay_loss_weight"])
        )
        kl_weight = float(
            variant.get("kl_weight", protected_plan["reference_kl_weight"])
        )
        anti_floor = float(
            variant.get("anti_repetition_weight", protected_plan["anti_repetition_weight"])
        )
        group_clip_caps = variant.get("group_clip")
        grad_rows = []
        replay_supervised_total = 0
        replay_elementary_total = 0
        replay_rows_total = 0

        local_steps = max(1, int(variant.get("steps", 2)))
        variant_retry_index = int(
            variant.get("retry_rejection_index", rejected)
        )
        variant_retry_seed_offset = variant_retry_index * 1_000_003
        for local_step in range(local_steps):
            step = start_step + local_step
            rng = random.Random(
                5_000_000 + step + variant_retry_seed_offset
            )
            indices = [rng.randrange(len(blocks)) for _ in range(32)]
            optimizer.zero_grad(set_to_none=True)

            lr = _learning_rate(
                base_lr=3e-4 * effective_scale,
                processed_tokens=processed_tokens + local_step * 4064,
                target_tokens=target_tokens,
                warmup_tokens=min(100_000, max(20_000, target_tokens // 10)),
            )
            warmup_factor = min(1.0, float(local_step + 1) / 4.0)
            lr *= warmup_factor
            _set_optimizer_policy(optimizer, learning_rate=lr, multipliers=multipliers)

            anti_weight, eos_weight = _anti_collapse_weights(
                "B_short_sentence_completion",
                before,
            )
            anti_weight = max(float(anti_weight), anti_floor)
            eos_weight = float(
                variant.get("causal_eos_weight", eos_weight)
            )

            micro_batches = [indices[offset:offset + 2] for offset in range(0, 32, 2)]
            prepared = []
            supervised = 0
            for micro_indices in micro_batches:
                ids, labels = causal_batch(blocks, micro_indices, device=runtime.device)
                count = int((labels[:, 1:] != -100).sum().item())
                prepared.append((ids, labels, count))
                supervised += count

            objective = {}
            for ids, labels, count in prepared:
                result = model(ids)
                loss, objective = _training_objective(
                    result["logits"],
                    labels,
                    ids,
                    eos_loss_weight=eos_weight,
                    repetition_unlikelihood_weight=anti_weight,
                    repetition_window=16,
                    aggregation=str(variant.get("anti_repetition_aggregation", "mean_negative")),
                )
                (loss * (count / max(1, supervised))).backward()

            replay_rng = random.Random(
                9_000_000 + step * 97 + variant_retry_seed_offset
            )
            replay_batch_size = max(2, int(variant.get("replay_batch_size", 2)))
            replay_sampling = str(variant.get("replay_sampling", "uniform"))
            if replay_sampling == "balanced":
                elementary_count = replay_batch_size // 2
                general_count = replay_batch_size - elementary_count
                replay_indices = [
                    elementary_indices[replay_rng.randrange(len(elementary_indices))]
                    for _ in range(elementary_count)
                ] + [
                    general_indices[replay_rng.randrange(len(general_indices))]
                    for _ in range(general_count)
                ]
                replay_rng.shuffle(replay_indices)
            elif replay_sampling == "elementary":
                replay_indices = [
                    elementary_indices[replay_rng.randrange(len(elementary_indices))]
                    for _ in range(replay_batch_size)
                ]
            else:
                replay_indices = [
                    replay_rng.randrange(len(protected_rows))
                    for _ in range(replay_batch_size)
                ]
            replay_elementary_count = sum(
                int(index in elementary_indices) for index in replay_indices
            )
            replay_ids, replay_labels = _sft_training_batch(runtime, protected_rows, replay_indices)
            replay_count = int((replay_labels[:, 1:] != -100).sum().item())
            replay_supervised_total += replay_count
            replay_elementary_total += replay_elementary_count
            replay_rows_total += len(replay_indices)
            replay_result = model(replay_ids)
            replay_loss, _ = _training_objective(
                replay_result["logits"],
                replay_labels,
                replay_ids,
                eos_loss_weight=float(
                    variant.get("replay_eos_weight", 1.0)
                ),
                repetition_unlikelihood_weight=anti_weight,
                repetition_window=16,
                aggregation=str(variant.get("anti_repetition_aggregation", "mean_negative")),
            )
            with torch.no_grad():
                ref_logits = reference.model(replay_ids)["logits"]
            kl_loss, kl_tokens = reference_kl_loss(
                replay_result["logits"],
                ref_logits,
                replay_labels,
                temperature=float(protected_plan["reference_kl_temperature"]),
            )
            protected_loss = replay_loss_weight * replay_loss + kl_weight * kl_loss
            protected_loss.backward()

            gradient_before_clip = _group_gradient_stats(model)
            group_clip_observed = _group_clip(optimizer, group_clip_caps)
            global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            grad_rows.append({
                "step": local_step,
                "effective_lr": float(lr),
                "gradient_before_clip": gradient_before_clip,
                "group_clip_observed_norms": group_clip_observed,
                "global_gradient_norm_before_clip": float(global_norm.detach().cpu()),
                "causal_ce_loss": float(objective.get("causal_ce_loss", 0.0)),
                "repetition_unlikelihood_loss": float(
                    objective.get("repetition_unlikelihood_loss", 0.0)
                ),
                "anti_repetition_weight": anti_weight,
                "protected_replay_loss": float(replay_loss.detach().cpu()),
                "protected_reference_kl_loss": float(kl_loss.detach().cpu()),
                "protected_reference_tokens": int(kl_tokens),
            })

        after = evaluate_phase5_language(runtime)
        accepted, gate = _segment_language_gate(
            before,
            after,
            anchor,
            attempted_tokens=4064 * local_steps,
        )
        probe = _activation_probe(model, probe_ids, baseline_activations)
        probe.pop("_activations", None)
        update = _group_update_stats(model, reference.model)
        optimizer_after = _optimizer_state_summary(optimizer)

        results.append({
            "name": variant["name"],
            "accepted_by_unchanged_guard": bool(accepted),
            "optimizer_state": state_mode,
            "retry_rejection_index": variant_retry_index,
            "attempted_steps": local_steps,
            "causal_eos_weight": float(
                variant.get("causal_eos_weight", 0.0)
            ),
            "replay_eos_weight": float(
                variant.get("replay_eos_weight", 1.0)
            ),
            "replay_batch_size": int(variant.get("replay_batch_size", 2)),
            "replay_sampling": str(variant.get("replay_sampling", "uniform")),
            "optimizer_storage": storage,
            "effective_scale": effective_scale,
            "steps": local_steps,
            "attempted_tokens": 4064 * local_steps,
            "multipliers": multipliers,
            "replay_loss_weight": replay_loss_weight,
            "reference_kl_weight": kl_weight,
            "anti_repetition_floor": anti_floor,
            "anti_repetition_aggregation": str(
                variant.get("anti_repetition_aggregation", "mean_negative")
            ),
            "group_clip": group_clip_caps,
            "before": {
                "language_nll": before["language_nll"],
                "repetition_rate": before["repetition_rate"],
                "unique_token_ratio": before["unique_token_ratio"],
                "dominant_token_fraction": before["dominant_token_fraction"],
                "token_entropy": before["token_entropy"],
                "eos_probability": before["eos_probability"],
                "generation_length": before["generation_length"],
                "longest_repeated_token_run": before["longest_repeated_token_run"],
            },
            "after": {
                "language_nll": after["language_nll"],
                "repetition_rate": after["repetition_rate"],
                "unique_token_ratio": after["unique_token_ratio"],
                "dominant_token_fraction": after["dominant_token_fraction"],
                "token_entropy": after["token_entropy"],
                "eos_probability": after["eos_probability"],
                "generation_length": after["generation_length"],
                "longest_repeated_token_run": after["longest_repeated_token_run"],
            },
            "delta": {
                "language_nll": float(after["language_nll"] - before["language_nll"]),
                "repetition_rate": float(after["repetition_rate"] - before["repetition_rate"]),
                "unique_token_ratio": float(after["unique_token_ratio"] - before["unique_token_ratio"]),
                "token_entropy": float(after["token_entropy"] - before["token_entropy"]),
                "eos_probability": float(
                    after["eos_probability"] - before["eos_probability"]
                ),
                "generation_length": float(
                    after["generation_length"] - before["generation_length"]
                ),
            },
            "gate": {
                "before_quality": gate["before_quality"],
                "after_quality": gate["after_quality"],
                "local_reasons": gate["local_reasons"],
                "after_anchor_violations": gate["after_anchor_violations"],
            },
            "replay_supervised_tokens": replay_supervised_total,
            "replay_rows_sampled": replay_rows_total,
            "replay_elementary_rows_sampled": replay_elementary_total,
            "replay_elementary_fraction": (
                replay_elementary_total / max(1, replay_rows_total)
            ),
            "gradient_steps": grad_rows,
            "parameter_updates": update,
            "optimizer_before": optimizer_before,
            "optimizer_after": optimizer_after,
            "activation_probe": probe,
            "optimizer_report": optimizer_report,
        })
        del runtime, optimizer

    return {
        "schema": 1,
        "version": "phase5-live-ablation-v1",
        "read_only": True,
        "checkpoint": str(candidate),
        "lineage_id": str(progress.get("lineage_id") or ""),
        "tokens_processed": processed_tokens,
        "parameters": int(sum(p.numel() for p in reference.model.parameters())),
        "consecutive_rejections": rejected,
        "segment_lr_scale": base_scale,
        "protected_replay_rows": protected_counts,
        "baseline_language": {
            "language_nll": before["language_nll"],
            "repetition_rate": before["repetition_rate"],
            "unique_token_ratio": before["unique_token_ratio"],
            "dominant_token_fraction": before["dominant_token_fraction"],
            "token_entropy": before["token_entropy"],
            "longest_repeated_token_run": before["longest_repeated_token_run"],
        },
        "anchor_language": {
            "language_nll": anchor["language_nll"],
            "repetition_rate": anchor["repetition_rate"],
            "unique_token_ratio": anchor["unique_token_ratio"],
            "dominant_token_fraction": anchor["dominant_token_fraction"],
            "token_entropy": anchor["token_entropy"],
            "longest_repeated_token_run": anchor["longest_repeated_token_run"],
        },
        "baseline_probe": baseline_probe,
        "gradient_pressure": pressure,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only live Phase-5 optimizer ablation")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_ablation(args.state_dir)
    target = Path(args.output)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "lineage_id": report["lineage_id"],
        "tokens_processed": report["tokens_processed"],
        "parameters": report["parameters"],
        "gradient_pressure": report["gradient_pressure"],
        "variants": [
            {
                "name": row["name"],
                "accepted": row["accepted_by_unchanged_guard"],
                "nll_delta": row["delta"]["language_nll"],
                "repetition_delta": row["delta"]["repetition_rate"],
                "quality_before": row["gate"]["before_quality"],
                "quality_after": row["gate"]["after_quality"],
                "reasons": row["gate"]["local_reasons"] + row["gate"]["after_anchor_violations"],
                "ffn_update_to_weight": (
                    row["parameter_updates"].get("ffn") or {}
                ).get("update_to_weight"),
            }
            for row in report["results"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
