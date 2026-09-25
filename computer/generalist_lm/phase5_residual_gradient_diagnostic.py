# diagnostic-current-live-v2
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .bootstrap_data import load_bootstrap_replay
from .bootstrap_training import (
    _elementary_rehabilitation_rows,
    _filter_protected_replay,
    _sft_row_fingerprint,
)
from .runtime import GeneralistRuntime
from .training import _batch, causal_training_objective, reference_kl_loss


def _r2_rows() -> list[Any]:
    curriculum = _elementary_rehabilitation_rows()
    rows: list[Any] = []
    for language in ("it", "en"):
        rows.extend(curriculum["R2_simple_responses"][language])
    return sorted(rows, key=_sft_row_fingerprint)


def _residual_parameters(model, source_d_ff: int):
    params = []
    masks = []
    target_d_ff = int(model.config.d_ff)
    for block in model.blocks:
        parameter = block.ff.down.weight
        parameter.requires_grad_(True)
        mask = parameter.new_zeros(parameter.shape)
        mask[:, int(source_d_ff):target_d_ff] = 1
        params.append(parameter)
        masks.append(mask)
    return params, masks


def _masked_gradient_vector(params, masks):
    import torch
    pieces = []
    for parameter, mask in zip(params, masks):
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter).masked_select(mask.bool()).float().cpu())
        else:
            pieces.append(
                parameter.grad.detach().float().masked_select(mask.bool()).cpu()
            )
    return torch.cat(pieces) if pieces else torch.zeros(0)


def _stats(vector) -> dict[str, float | int]:
    import torch
    if vector.numel() == 0:
        return {"coordinates": 0, "l2_norm": 0.0, "max_abs": 0.0}
    return {
        "coordinates": int(vector.numel()),
        "l2_norm": float(torch.linalg.vector_norm(vector).item()),
        "max_abs": float(vector.abs().max().item()),
    }


def _cosine(left, right) -> float:
    import torch
    denom = float(torch.linalg.vector_norm(left).item()) * float(
        torch.linalg.vector_norm(right).item()
    )
    if denom <= 1e-20:
        return 0.0
    return float(torch.dot(left, right).item() / denom)


def run(state_dir: str | Path) -> dict[str, Any]:
    import torch

    state = Path(state_dir).expanduser().resolve()
    bootstrap = state / "bootstrap-data"
    candidate = bootstrap / "candidate"
    progress = json.loads((bootstrap / "progress.json").read_text(encoding="utf-8"))
    revival = dict(progress.get("dead_capacity_revival") or {})
    source_d_ff = int(revival.get("source_d_ff", 0) or 0)
    if not revival.get("completed") or source_d_ff <= 0:
        raise RuntimeError("missing revived residual source")

    runtime = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    reference = GeneralistRuntime.from_checkpoint(candidate, device="cpu")
    model = runtime.model
    teacher = reference.model
    teacher.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    params, masks = _residual_parameters(model, source_d_ff)

    examples = _r2_rows()
    ids, labels = _batch(
        examples,
        runtime.tokenizer,
        model.config.context_length,
        list(range(len(examples))),
    )
    ids = ids.to(runtime.device)
    labels = labels.to(runtime.device)

    def gradient_for(weight: float, eos_weight: float = 1.0):
        model.zero_grad(set_to_none=True)
        logits = model(ids)["logits"]
        loss, stats = causal_training_objective(
            logits,
            labels,
            ids,
            eos_loss_weight=eos_weight,
            repetition_unlikelihood_weight=weight,
            repetition_window=16,
        )
        loss.backward()
        for parameter, mask in zip(params, masks):
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
        return loss.detach(), stats, _masked_gradient_vector(params, masks)

    ce_loss, ce_stats, ce_grad = gradient_for(0.0, 1.0)
    total_loss, total_stats, total_grad = gradient_for(1.0, 1.0)
    ul_grad = total_grad - ce_grad

    eos_loss, eos_stats, eos_total_grad = gradient_for(0.0, 1.10)
    eos_aux_grad = eos_total_grad - ce_grad

    replay = load_bootstrap_replay(bootstrap)
    elementary_fps = {
        _sft_row_fingerprint(row)
        for stage in _elementary_rehabilitation_rows().values()
        for language in ("it", "en")
        for row in stage[language]
    }
    safe, _ = _filter_protected_replay(list(replay.sft_train), heldout_sft=())
    anchors = sorted(
        [row for row in safe if _sft_row_fingerprint(row) not in elementary_fps],
        key=_sft_row_fingerprint,
    )
    anchor_indices = list(range(min(12, len(anchors))))
    anchor_ids, anchor_labels = _batch(
        anchors,
        runtime.tokenizer,
        model.config.context_length,
        anchor_indices,
    )
    anchor_ids = anchor_ids.to(runtime.device)
    anchor_labels = anchor_labels.to(runtime.device)
    model.zero_grad(set_to_none=True)
    student = model(anchor_ids)["logits"]
    with torch.no_grad():
        teacher_logits = teacher(anchor_ids)["logits"]
    kl_loss, kl_tokens = reference_kl_loss(
        student,
        teacher_logits,
        anchor_labels,
        temperature=1.0,
    )
    kl_loss.backward()
    for parameter, mask in zip(params, masks):
        if parameter.grad is not None:
            parameter.grad.mul_(mask)
    kl_grad = _masked_gradient_vector(params, masks)

    ul_norm = _stats(ul_grad)["l2_norm"]
    ce_norm = _stats(ce_grad)["l2_norm"]
    eos_norm = _stats(eos_aux_grad)["l2_norm"]
    return {
        "schema": 1,
        "version": "phase5-residual-gradient-diagnostic-v1",
        "read_only": True,
        "lineage_id": str(progress.get("lineage_id") or ""),
        "tokens_processed": int(progress.get("tokens_processed", 0) or 0),
        "source_d_ff": source_d_ff,
        "target_d_ff": int(model.config.d_ff),
        "r2_rows": len(examples),
        "anchor_rows": len(anchor_indices),
        "losses": {
            "ce": float(ce_loss.cpu()),
            "total_at_ul_weight_1": float(total_loss.cpu()),
            "unlikelihood": float(total_stats["repetition_unlikelihood_loss"]),
            "eos_weighted_ce": float(eos_loss.cpu()),
            "kl_at_reference": float(kl_loss.detach().cpu()),
            "kl_tokens": int(kl_tokens),
        },
        "gradients": {
            "ce": _stats(ce_grad),
            "unlikelihood_unweighted": _stats(ul_grad),
            "eos_auxiliary_unweighted": _stats(eos_aux_grad),
            "kl_at_reference": _stats(kl_grad),
            "ce_vs_unlikelihood_cosine": _cosine(ce_grad, ul_grad),
            "ce_vs_eos_aux_cosine": _cosine(ce_grad, eos_aux_grad),
            "ul_vs_eos_aux_cosine": _cosine(ul_grad, eos_aux_grad),
            "ul_to_ce_norm_ratio": float(ul_norm / max(1e-20, ce_norm)),
            "weighted_ul_002_to_ce_norm_ratio": float(
                0.02 * ul_norm / max(1e-20, ce_norm)
            ),
            "weighted_eos_010_to_ce_norm_ratio": float(
                eos_norm / max(1e-20, ce_norm)
            ),
        },
        "objective_stats": {
            "ce": ce_stats,
            "ul_weight_1": total_stats,
            "eos_weight_110": eos_stats,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.state_dir)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
