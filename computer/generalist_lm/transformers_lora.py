from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

from .training import SFTExample


@dataclass(frozen=True)
class LoRATrainConfig:
    steps: int = 100
    batch_size: int = 1
    learning_rate: float = 2e-4
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    max_length: int = 1024
    seed: int = 23

    def validate(self) -> "LoRATrainConfig":
        if not (1 <= self.steps <= 100_000):
            raise ValueError("steps out of bounds")
        if not (1 <= self.batch_size <= 128):
            raise ValueError("batch_size out of bounds")
        if not (1e-7 <= self.learning_rate <= 1e-2):
            raise ValueError("learning_rate out of bounds")
        if not (1 <= self.rank <= 256):
            raise ValueError("LoRA rank out of bounds")
        if not (1 <= self.alpha <= 1024):
            raise ValueError("LoRA alpha out of bounds")
        if not (0.0 <= self.dropout <= 0.5):
            raise ValueError("LoRA dropout out of bounds")
        if not (64 <= self.max_length <= 32768):
            raise ValueError("max_length out of bounds")
        return self


def _require_local_model_dir(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("local foundation model directory does not exist")
    return root


def train_local_lora(
    model_path: str | Path,
    examples: Sequence[SFTExample],
    output_dir: str | Path,
    *,
    config: LoRATrainConfig | None = None,
    device: str = "cpu",
) -> dict:
    """Fine-tune an already-downloaded causal LM with LoRA.

    No network acquisition or remote model code is permitted. This function is
    intentionally optional because transformers/peft are large dependencies.
    """
    cfg = (config or LoRATrainConfig()).validate()
    root = _require_local_model_dir(model_path)
    output = Path(output_dir).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("LoRA output must not overwrite the base model directory")
    if not examples:
        raise ValueError("no SFT examples")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("transformers and peft are required for local LoRA training") from exc

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("foundation tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, adapter_cfg)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    def encode(example: SFTExample):
        messages = example.messages
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        else:
            full = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1]) + "\nassistant:"

        encoded = tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )["input_ids"][0]
        ids = encoded["input_ids"][0]
        attention = encoded.get("attention_mask", torch.ones_like(ids))
        labels = ids.clone()
        prompt_len = min(int(prompt_ids.shape[0]), int(labels.shape[0]))
        labels[:prompt_len] = -100
        return ids, attention, labels

    encoded_rows = [encode(example) for example in examples]
    rng = random.Random(cfg.seed)
    losses: list[float] = []

    for _ in range(cfg.steps):
        batch_rows = [encoded_rows[rng.randrange(len(encoded_rows))] for _ in range(cfg.batch_size)]
        max_len = max(int(row[0].shape[0]) for row in batch_rows)
        pad = int(tokenizer.pad_token_id)

        ids_batch = []
        mask_batch = []
        labels_batch = []
        for ids, mask, labels in batch_rows:
            missing = max_len - int(ids.shape[0])
            if missing:
                ids = torch.cat([ids, torch.full((missing,), pad, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros((missing,), dtype=mask.dtype)])
                labels = torch.cat([labels, torch.full((missing,), -100, dtype=labels.dtype)])
            ids_batch.append(ids)
            mask_batch.append(mask)
            labels_batch.append(labels)

        ids = torch.stack(ids_batch).to(device)
        attention = torch.stack(mask_batch).to(device)
        labels = torch.stack(labels_batch).to(device)

        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=ids, attention_mask=attention, labels=labels)
        loss = result.loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite LoRA loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "ok": True,
        "steps": cfg.steps,
        "examples": len(examples),
        "initial_step_loss": losses[0],
        "final_step_loss": losses[-1],
        "best_step_loss": min(losses),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / max(1, total)),
        "output_dir": str(output),
        "policy": "local-files-only base model; trust_remote_code disabled",
    }
