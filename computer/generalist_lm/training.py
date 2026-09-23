from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .tokenizer import BYTE_OFFSET, EOS, PAD, ByteTokenizer


@dataclass(frozen=True)
class SFTExample:
    messages: list[dict[str, str]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SFTExample":
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("SFT example requires non-empty messages")
        cleaned: list[dict[str, str]] = []
        for row in messages:
            if not isinstance(row, dict):
                raise ValueError("each message must be an object")
            cleaned.append({"role": str(row.get("role", "")), "content": str(row.get("content", ""))})
        if cleaned[-1]["role"].lower() != "assistant":
            raise ValueError("last SFT message must be assistant")
        return cls(cleaned)


def load_sft_jsonl(path: str | Path) -> list[SFTExample]:
    out: list[SFTExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(SFTExample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid SFT row at line {line_no}: {exc}") from exc
    return out


def encode_sft_example(example: SFTExample, tokenizer: ByteTokenizer, context_length: int):
    import torch
    messages = list(example.messages)
    target = messages[-1]
    prompt = messages[:-1]
    prompt_ids = tokenizer.serialize_messages(prompt, add_generation_prompt=True)
    target_ids = tokenizer.encode(target["content"], eos=True)

    max_target = max(1, context_length - 1)
    target_ids = target_ids[:max_target]
    prompt_budget = max(1, context_length - len(target_ids))
    prompt_ids = prompt_ids[-prompt_budget:]

    ids = (prompt_ids + target_ids)[:context_length]
    labels = [-100] * min(len(prompt_ids), len(ids))
    labels.extend(ids[len(labels):])

    if len(ids) < context_length:
        pad = context_length - len(ids)
        ids.extend([PAD] * pad)
        labels.extend([-100] * pad)

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _batch(examples: list[SFTExample], tokenizer: ByteTokenizer, context_length: int, indices: Iterable[int]):
    import torch
    rows = [encode_sft_example(examples[i], tokenizer, context_length) for i in indices]
    return torch.stack([x[0] for x in rows]), torch.stack([x[1] for x in rows])


def loss_on_examples(
    model,
    tokenizer: ByteTokenizer,
    examples: list[SFTExample],
    *,
    device: str = "cpu",
    batch_size: int = 8,
) -> float:
    """Token-weighted held-out loss without materializing the full corpus batch.

    The previous implementation stacked every validation/training example at
    once. With a long autonomous replay corpus that could request tens of GB of
    activation memory even for a relatively small model. Micro-batching keeps
    peak memory bounded while preserving the same global mean over supervised
    target tokens.
    """
    import torch
    if not examples:
        raise ValueError("no SFT examples")
    model.eval()
    chunk = max(1, int(batch_size))
    total_weighted_loss = 0.0
    total_supervised_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), chunk):
            stop = min(len(examples), start + chunk)
            ids, labels = _batch(
                examples,
                tokenizer,
                model.config.context_length,
                range(start, stop),
            )
            ids, labels = ids.to(device), labels.to(device)
            supervised_tokens = int((labels != -100).sum().item())
            if supervised_tokens <= 0:
                continue
            loss = model(ids, labels=labels)["loss"]
            total_weighted_loss += float(loss.detach().cpu()) * supervised_tokens
            total_supervised_tokens += supervised_tokens
    if total_supervised_tokens <= 0:
        raise ValueError("SFT examples contain no supervised target tokens")
    return total_weighted_loss / total_supervised_tokens


def _utf8_prefix(text: str, max_bytes: int) -> str:
    """Return the longest valid UTF-8 character prefix within max_bytes."""
    raw = str(text).encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return str(text)
    return raw[:max(0, int(max_bytes))].decode("utf-8", errors="ignore")


def nll_stats_on_examples(model, tokenizer, examples: list[SFTExample], *, device: str = "cpu") -> dict[str, float | int]:
    """Return tokenizer-comparable held-out negative log-likelihood metrics.

    Cross-entropy per token is not comparable across tokenizer families because
    changing tokenization changes the number and entropy of target tokens. NLL
    per UTF-8 target byte keeps the promotion signal on a common denominator.

    The supervised assistant target is first truncated by a tokenizer-independent
    UTF-8 byte budget. This guarantees byte-v1 and merge-only bpe-v1 score the
    same target byte prefix even when the original answer exceeds context.
    """
    import torch
    from torch.nn import functional as F

    if not examples:
        raise ValueError("no SFT examples")

    # byte-v1 is the worst-case tokenization for merge-only bpe-v1: one token
    # per UTF-8 byte. Reserve one token for prompt context and one for EOS, so
    # the selected target prefix fits both tokenizer families without
    # tokenizer-dependent target truncation.
    target_byte_budget = max(1, int(model.config.context_length) - 2)
    fair_examples: list[SFTExample] = []
    for example in examples:
        messages = [dict(row) for row in example.messages]
        messages[-1]["content"] = _utf8_prefix(
            messages[-1].get("content", ""),
            target_byte_budget,
        )
        fair_examples.append(SFTExample(messages))

    model.eval()
    ids, labels = _batch(fair_examples, tokenizer, model.config.context_length, range(len(fair_examples)))
    ids, labels = ids.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(ids)["logits"][:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()
        total_nll = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

    target_tokens = int((targets != -100).sum().item())
    target_bytes = 0
    labels_cpu = labels.detach().cpu()
    for row in labels_cpu:
        supervised = [int(token) for token in row.tolist() if int(token) != -100]
        target_bytes += len(
            tokenizer.decode(supervised, skip_special=True).encode(
                "utf-8",
                errors="replace",
            )
        )
    nll = float(total_nll.detach().cpu())
    per_token = nll / max(1, target_tokens)
    per_byte = nll / max(1, target_bytes)
    return {
        "total_nll": nll,
        "target_tokens": target_tokens,
        "target_bytes": int(target_bytes),
        "loss_per_token": per_token,
        "nll_per_byte": per_byte,
        "bits_per_byte": per_byte / math.log(2.0),
    }


def causal_training_objective(
    logits,
    labels,
    input_ids,
    *,
    eos_token_id: int = EOS,
    eos_loss_weight: float = 1.0,
    repetition_unlikelihood_weight: float = 0.0,
    repetition_window: int = 16,
    special_token_floor: int = BYTE_OFFSET,
):
    """Causal LM loss with bounded, target-safe anti-collapse auxiliaries.

    Cross-entropy remains the primary objective.  EOS targets can be mildly
    reweighted so short natural sequences learn to terminate, while optional
    token-level unlikelihood penalizes recently seen tokens only when they are
    *not* the correct next token.  The helper never changes decoding.
    """
    import torch
    from torch.nn import functional as F

    if logits.ndim != 3 or labels.ndim != 2 or input_ids.ndim != 2:
        raise ValueError("unexpected causal objective tensor rank")
    if labels.shape != input_ids.shape or logits.shape[:2] != input_ids.shape:
        raise ValueError("causal objective tensors must share batch/time dimensions")

    shifted_logits = logits[:, :-1, :]
    targets = labels[:, 1:]
    valid = targets != -100

    token_loss = F.cross_entropy(
        shifted_logits.contiguous().view(-1, shifted_logits.shape[-1]),
        targets.contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)

    weights = torch.ones_like(token_loss)
    eos_weight = max(1.0, float(eos_loss_weight))
    if eos_weight > 1.0:
        weights = torch.where(
            targets == int(eos_token_id),
            torch.full_like(weights, eos_weight),
            weights,
        )
    valid_weights = weights * valid.to(weights.dtype)
    ce_loss = (token_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)

    ul_weight = max(0.0, float(repetition_unlikelihood_weight))
    ul_loss = shifted_logits.sum() * 0.0
    negative_count = 0
    if ul_weight > 0.0 and shifted_logits.shape[1] > 0:
        window = max(1, min(int(repetition_window), int(input_ids.shape[1])))
        negative_mask = torch.zeros_like(shifted_logits, dtype=torch.bool)
        prediction_steps = int(shifted_logits.shape[1])

        for position in range(prediction_steps):
            start = max(0, position - window + 1)
            recent = input_ids[:, start:position + 1]
            negative_mask[:, position, :].scatter_(1, recent, True)

        # Chat/control tokens are structural and should not be discouraged.
        floor = max(0, min(int(special_token_floor), int(shifted_logits.shape[-1])))
        if floor:
            negative_mask[:, :, :floor] = False

        # Never penalize the ground-truth target, even if it legitimately
        # repeats a recent token.
        safe_targets = targets.clamp_min(0).unsqueeze(-1)
        negative_mask.scatter_(2, safe_targets, False)
        negative_mask &= valid.unsqueeze(-1)

        negative_count = int(negative_mask.sum().item())
        if negative_count:
            probs = torch.softmax(shifted_logits, dim=-1)
            negative_probs = probs.masked_select(negative_mask).clamp(
                min=0.0,
                max=1.0 - 1e-6,
            )
            ul_loss = -torch.log1p(-negative_probs).mean()

    total = ce_loss + ul_weight * ul_loss
    stats = {
        "causal_ce_loss": float(ce_loss.detach().cpu()),
        "repetition_unlikelihood_loss": float(ul_loss.detach().cpu()),
        "repetition_unlikelihood_weight": ul_weight,
        "repetition_negative_count": int(negative_count),
        "eos_loss_weight": eos_weight,
    }
    return total, stats


def train_sft(
    model,
    tokenizer: ByteTokenizer,
    examples: list[SFTExample],
    *,
    steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = 7,
    device: str = "cpu",
    gradient_accumulation_steps: int = 1,
    precision: str = "fp32",
    repetition_unlikelihood_weight: float = 0.0,
    eos_loss_weight: float = 1.0,
    repetition_window: int = 16,
) -> dict[str, Any]:
    import torch
    if not examples:
        raise ValueError("no SFT examples")

    steps = max(1, int(steps))
    batch_size = max(1, int(batch_size))
    accumulation = max(1, min(64, int(gradient_accumulation_steps)))
    precision = str(precision).strip().lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be one of: fp32, bf16, fp16")

    device_obj = torch.device(device)
    if precision == "fp16" and device_obj.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if precision == "bf16":
        if device_obj.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 is not supported by this CUDA device")
        if device_obj.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 training requires CPU or CUDA")

    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device_obj)
    initial_loss = loss_on_examples(model, tokenizer, examples, device=str(device_obj))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(precision == "fp16" and device_obj.type == "cuda"),
    )
    losses: list[float] = []
    supervised_tokens = 0
    last_objective_stats = {
        "causal_ce_loss": float(initial_loss),
        "repetition_unlikelihood_loss": 0.0,
        "repetition_unlikelihood_weight": float(repetition_unlikelihood_weight),
        "repetition_negative_count": 0,
        "eos_loss_weight": float(eos_loss_weight),
    }

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for _micro in range(accumulation):
            indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
            ids, labels = _batch(
                examples,
                tokenizer,
                model.config.context_length,
                indices,
            )
            ids, labels = ids.to(device_obj), labels.to(device_obj)
            supervised_tokens += int((labels[:, 1:] != -100).sum().item())

            with torch.autocast(
                device_type=device_obj.type,
                dtype=autocast_dtype,
                enabled=(autocast_dtype is not None),
            ):
                result = model(ids)
                raw_loss, objective_stats = causal_training_objective(
                    result["logits"],
                    labels,
                    ids,
                    eos_loss_weight=eos_loss_weight,
                    repetition_unlikelihood_weight=repetition_unlikelihood_weight,
                    repetition_window=repetition_window,
                )
                loss = raw_loss / accumulation

            if not torch.isfinite(raw_loss):
                raise RuntimeError("non-finite language-model loss")

            scaler.scale(loss).backward()
            micro_losses.append(float(raw_loss.detach().cpu()))
            last_objective_stats = objective_stats

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(sum(micro_losses) / len(micro_losses))

    final_loss = loss_on_examples(
        model,
        tokenizer,
        examples,
        device=str(device_obj),
    )
    return {
        "ok": bool(final_loss < initial_loss),
        "steps": steps,
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "precision": precision,
        "learning_rate": float(learning_rate),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_step_loss": min(losses),
        "loss_improvement": initial_loss - final_loss,
        "supervised_tokens": int(supervised_tokens),
        "objective": last_objective_stats,
    }


def train_sft_residual_recovery(
    model,
    reference_model,
    tokenizer: ByteTokenizer,
    examples: list[SFTExample],
    *,
    anchor_examples: list[SFTExample] | None = None,
    source_d_ff: int,
    steps: int = 64,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    seed: int = 7,
    device: str = "cpu",
    repetition_unlikelihood_weight: float = 0.04,
    eos_loss_weight: float = 1.25,
    repetition_window: int = 16,
    kl_weight: float = 0.75,
    train_upstream: bool = False,
) -> dict[str, Any]:
    """Trust-region SFT for the revived FFN residual branch.

    The pre-revival language model is treated as an immutable reference. Only
    the revived SwiGLU coordinates may move; all legacy parameters and all
    legacy coordinates inside the widened FFN stay bit-stable. A token-wise
    teacher KL term limits the new residual's drift while ordinary SFT and a
    light target-safe unlikelihood term teach the missing language behavior.
    """
    import torch
    from torch.nn import functional as F

    if not examples:
        raise ValueError("no SFT examples")
    anchors = list(anchor_examples or [])
    if str(getattr(model.config, "ff_variant", "")) != "swiglu":
        raise ValueError("residual recovery requires a SwiGLU model")
    if tuple(model.state_dict().keys()) != tuple(reference_model.state_dict().keys()):
        raise ValueError("reference model architecture mismatch")

    steps = max(1, int(steps))
    batch_size = max(1, int(batch_size))
    source_d_ff = int(source_d_ff)
    target_d_ff = int(model.config.d_ff)
    if source_d_ff <= 0 or source_d_ff >= target_d_ff:
        raise ValueError("source_d_ff must identify the pre-growth FF width")
    kl_weight = max(0.0, float(kl_weight))

    device_obj = torch.device(device)
    model.to(device_obj)
    reference_model.to(device_obj)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    # Freeze every whole parameter first. We then enable only the widened FFN
    # tensors and mask their legacy coordinates after backward.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    gradient_masks: list[tuple[Any, Any]] = []
    trainable_parameters = []
    trainable_coordinate_count = 0

    for block in model.blocks:
        ff = block.ff
        up = ff.up
        down = ff.down

        down.weight.requires_grad_(True)
        down_mask = torch.zeros_like(down.weight, device=device_obj)
        down_mask[:, source_d_ff:target_d_ff] = 1
        gradient_masks.append((down.weight, down_mask))
        trainable_parameters.append(down.weight)
        trainable_coordinate_count += int(down_mask.sum().item())

        if bool(train_upstream):
            up.weight.requires_grad_(True)
            up_mask = torch.zeros_like(up.weight, device=device_obj)
            up_mask[source_d_ff:target_d_ff, :] = 1
            up_mask[target_d_ff + source_d_ff : 2 * target_d_ff, :] = 1
            gradient_masks.append((up.weight, up_mask))
            trainable_parameters.append(up.weight)
            trainable_coordinate_count += int(up_mask.sum().item())

            if up.bias is not None:
                up.bias.requires_grad_(True)
                bias_mask = torch.zeros_like(up.bias, device=device_obj)
                bias_mask[source_d_ff:target_d_ff] = 1
                bias_mask[target_d_ff + source_d_ff : 2 * target_d_ff] = 1
                gradient_masks.append((up.bias, bias_mask))
                trainable_parameters.append(up.bias)
                trainable_coordinate_count += int(bias_mask.sum().item())

    if not trainable_parameters or trainable_coordinate_count <= 0:
        raise RuntimeError("residual recovery has no trainable coordinates")

    rng = random.Random(seed)
    torch.manual_seed(seed)
    initial_loss = loss_on_examples(model, tokenizer, examples, device=str(device_obj))
    model.train()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=0.0,
    )

    losses: list[float] = []
    kl_losses: list[float] = []
    supervised_tokens = 0
    last_objective_stats: dict[str, Any] = {}

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
        ids, labels = _batch(
            examples,
            tokenizer,
            model.config.context_length,
            indices,
        )
        ids, labels = ids.to(device_obj), labels.to(device_obj)
        supervised_tokens += int((labels[:, 1:] != -100).sum().item())

        result = model(ids)
        raw_loss, objective_stats = causal_training_objective(
            result["logits"],
            labels,
            ids,
            eos_loss_weight=float(eos_loss_weight),
            repetition_unlikelihood_weight=float(repetition_unlikelihood_weight),
            repetition_window=int(repetition_window),
        )

        kl_loss = raw_loss.detach() * 0.0
        anchor_token_count = 0
        if kl_weight > 0.0:
            if not anchors:
                raise RuntimeError(
                    "residual KL recovery requires protected replay anchors"
                )
            anchor_indices = [
                rng.randrange(len(anchors))
                for _ in range(batch_size)
            ]
            anchor_ids, anchor_labels = _batch(
                anchors,
                tokenizer,
                model.config.context_length,
                anchor_indices,
            )
            anchor_ids = anchor_ids.to(device_obj)
            anchor_labels = anchor_labels.to(device_obj)
            anchor_student_logits = model(anchor_ids)["logits"][:, :-1, :].float()
            with torch.no_grad():
                anchor_reference_logits = reference_model(anchor_ids)[
                    "logits"
                ][:, :-1, :].float()

            valid_anchor = anchor_labels[:, 1:] != -100
            teacher_log_probs = F.log_softmax(anchor_reference_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            student_log_probs = F.log_softmax(anchor_student_logits, dim=-1)
            token_kl = (
                teacher_probs * (teacher_log_probs - student_log_probs)
            ).sum(dim=-1)
            anchor_token_count = int(valid_anchor.sum().item())
            if anchor_token_count:
                kl_loss = token_kl.masked_select(valid_anchor).mean()
            else:
                kl_loss = token_kl.mean() * 0.0

        total_loss = raw_loss + kl_weight * kl_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("non-finite residual recovery loss")
        total_loss.backward()

        # The optimizer owns whole tensors, but only these coordinates may
        # receive gradients. weight_decay=0 guarantees masked coordinates remain
        # unchanged even though their containing tensor is optimized.
        for parameter, mask in gradient_masks:
            if parameter.grad is not None:
                parameter.grad.mul_(mask)

        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()

        losses.append(float(total_loss.detach().cpu()))
        kl_losses.append(float(kl_loss.detach().cpu()))
        last_objective_stats = {
            **objective_stats,
            "teacher_kl_loss": float(kl_loss.detach().cpu()),
            "teacher_kl_weight": kl_weight,
            "teacher_anchor_tokens": int(anchor_token_count),
        }

    final_loss = loss_on_examples(
        model,
        tokenizer,
        examples,
        device=str(device_obj),
    )
    return {
        "ok": bool(final_loss < initial_loss),
        "mode": "residual_kl_recovery",
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "best_step_loss": min(losses),
        "loss_improvement": float(initial_loss - final_loss),
        "supervised_tokens": int(supervised_tokens),
        "source_d_ff": source_d_ff,
        "target_d_ff": target_d_ff,
        "train_upstream": bool(train_upstream),
        "trainable_coordinate_count": int(trainable_coordinate_count),
        "teacher_kl_weight": kl_weight,
        "anchor_example_count": len(anchors),
        "mean_teacher_kl_loss": (
            float(sum(kl_losses) / len(kl_losses)) if kl_losses else 0.0
        ),
        "max_teacher_kl_loss": max(kl_losses) if kl_losses else 0.0,
        "objective": last_objective_stats,
    }
