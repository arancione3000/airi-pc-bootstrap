from __future__ import annotations

import random
from typing import Any

from .corpus import CorpusChunk
from .tokenizer import BOS, EOS, PAD, ByteTokenizer


def _encode_chunk(
    chunk: CorpusChunk,
    tokenizer: ByteTokenizer,
    context_length: int,
    *,
    offset: int = 0,
):
    import torch

    prefix = f"<file path={chunk.path}>\n"
    ids = [BOS] + tokenizer.encode(prefix + chunk.text) + [EOS]
    if len(ids) > context_length:
        start = max(0, min(int(offset), len(ids) - context_length))
        ids = ids[start : start + context_length]
    labels = list(ids)
    if len(ids) < context_length:
        pad = context_length - len(ids)
        ids.extend([PAD] * pad)
        labels.extend([-100] * pad)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _batch(
    chunks: list[CorpusChunk],
    tokenizer: ByteTokenizer,
    context_length: int,
    selections: list[tuple[int, int]],
):
    import torch

    rows = [
        _encode_chunk(chunks[index], tokenizer, context_length, offset=offset)
        for index, offset in selections
    ]
    return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])


def corpus_loss(
    model,
    tokenizer: ByteTokenizer,
    chunks: list[CorpusChunk],
    *,
    device: str = "cpu",
    max_eval_chunks: int = 16,
) -> float:
    import torch

    if not chunks:
        raise ValueError("no pretraining corpus chunks")
    selected = list(range(min(len(chunks), max(1, int(max_eval_chunks)))))
    rows = [(index, 0) for index in selected]
    ids, labels = _batch(chunks, tokenizer, model.config.context_length, rows)
    model.eval()
    with torch.no_grad():
        loss = model(ids.to(device), labels=labels.to(device))["loss"]
    return float(loss.detach().cpu())


def train_causal_pretraining(
    model,
    tokenizer: ByteTokenizer,
    chunks: list[CorpusChunk],
    *,
    steps: int = 12,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int = 19,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch

    if not chunks:
        return {
            "ok": True,
            "skipped": True,
            "reason": "empty pretraining corpus",
            "steps": 0,
        }

    steps = max(1, int(steps))
    batch_size = max(1, int(batch_size))
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    model.to(device)

    initial_loss = corpus_loss(model, tokenizer, chunks, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []

    for _ in range(steps):
        selections: list[tuple[int, int]] = []
        for _batch_index in range(batch_size):
            index = rng.randrange(len(chunks))
            encoded_length = len(tokenizer.encode(chunks[index].text)) + 2
            max_offset = max(0, encoded_length - model.config.context_length)
            offset = rng.randrange(max_offset + 1) if max_offset else 0
            selections.append((index, offset))

        ids, labels = _batch(
            chunks,
            tokenizer,
            model.config.context_length,
            selections,
        )
        ids = ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=labels)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite causal pretraining loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    final_loss = corpus_loss(model, tokenizer, chunks, device=device)
    return {
        "ok": bool(final_loss < initial_loss),
        "skipped": False,
        "steps": steps,
        "chunks": len(chunks),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_step_loss": min(losses),
        "loss_improvement": initial_loss - final_loss,
    }
