from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .tokenizer import PAD, ByteTokenizer


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


def loss_on_examples(model, tokenizer: ByteTokenizer, examples: list[SFTExample], *, device: str = "cpu") -> float:
    import torch
    if not examples:
        raise ValueError("no SFT examples")
    model.eval()
    ids, labels = _batch(examples, tokenizer, model.config.context_length, range(len(examples)))
    ids, labels = ids.to(device), labels.to(device)
    with torch.no_grad():
        loss = model(ids, labels=labels)["loss"]
    return float(loss.detach().cpu())


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

            with torch.autocast(
                device_type=device_obj.type,
                dtype=autocast_dtype,
                enabled=(autocast_dtype is not None),
            ):
                raw_loss = model(ids, labels=labels)["loss"]
                loss = raw_loss / accumulation

            if not torch.isfinite(raw_loss):
                raise RuntimeError("non-finite language-model loss")

            scaler.scale(loss).backward()
            micro_losses.append(float(raw_loss.detach().cpu()))

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
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_step_loss": min(losses),
        "loss_improvement": initial_loss - final_loss,
    }
