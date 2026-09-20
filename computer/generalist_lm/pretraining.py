from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Iterable, Sequence

from .tokenizer import EOS, PAD, ByteTokenizer


_ALLOWED_SUFFIXES = {
    ".txt", ".md", ".py", ".json", ".jsonl", ".csv", ".rst",
    ".html", ".htm", ".js", ".ts", ".java", ".c", ".h", ".cpp",
    ".hpp", ".rs", ".go", ".sql", ".yaml", ".yml", ".toml",
}


@dataclass(frozen=True)
class CorpusDocument:
    source: str
    text: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class CorpusLoadReport:
    documents: list[CorpusDocument]
    skipped: list[dict]
    total_bytes: int


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def load_local_corpus(
    paths: Iterable[str | Path],
    *,
    allowed_roots: Iterable[str | Path],
    max_file_bytes: int = 5_000_000,
    max_total_bytes: int = 100_000_000,
    max_documents: int = 50_000,
) -> CorpusLoadReport:
    """Load a bounded text corpus without following data outside allowed roots.

    This is deliberately local-only. Remote acquisition belongs to a separate
    read-only ingestion step whose downloaded artifact must then be reviewed and
    supplied as a local corpus.
    """
    roots = [Path(root).expanduser().resolve() for root in allowed_roots]
    if not roots:
        raise ValueError("at least one allowed corpus root is required")

    documents: list[CorpusDocument] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    total = 0

    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not _inside(path, roots):
            raise PermissionError(f"corpus path escapes allowed roots: {path}")
        if path.is_dir():
            candidates.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            candidates.append(path)

    for path in candidates:
        if len(documents) >= max(1, int(max_documents)):
            skipped.append({"path": str(path), "reason": "document_budget"})
            break
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            skipped.append({"path": str(path), "reason": "missing"})
            continue
        if not _inside(resolved, roots):
            skipped.append({"path": str(path), "reason": "symlink_escape"})
            continue
        if resolved.suffix.lower() not in _ALLOWED_SUFFIXES:
            skipped.append({"path": str(resolved), "reason": "unsupported_suffix"})
            continue
        try:
            size = resolved.stat().st_size
        except OSError:
            skipped.append({"path": str(resolved), "reason": "stat_failed"})
            continue
        if size <= 0:
            skipped.append({"path": str(resolved), "reason": "empty"})
            continue
        if size > max_file_bytes:
            skipped.append({"path": str(resolved), "reason": "file_budget"})
            continue
        if total + size > max_total_bytes:
            skipped.append({"path": str(resolved), "reason": "total_budget"})
            break

        raw = resolved.read_bytes()
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            skipped.append({"path": str(resolved), "reason": "empty_text"})
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            skipped.append({"path": str(resolved), "reason": "duplicate"})
            continue
        seen.add(digest)
        total += len(raw)
        documents.append(
            CorpusDocument(
                source=str(resolved),
                text=text,
                sha256=digest,
                bytes=len(raw),
            )
        )

    return CorpusLoadReport(documents=documents, skipped=skipped, total_bytes=total)


def pack_causal_blocks(
    documents: Sequence[CorpusDocument],
    tokenizer: ByteTokenizer,
    *,
    context_length: int,
    min_tokens: int = 8,
) -> list[list[int]]:
    """Pack documents into fixed-size next-token training blocks."""
    context_length = int(context_length)
    if context_length < 8:
        raise ValueError("context_length must be at least 8")

    stream: list[int] = []
    for document in documents:
        ids = tokenizer.encode(document.text)
        if not ids:
            continue
        stream.extend(ids)
        stream.append(EOS)

    block_size = context_length
    blocks: list[list[int]] = []
    cursor = 0
    while cursor < len(stream):
        block = stream[cursor:cursor + block_size]
        cursor += block_size
        if len(block) < max(2, int(min_tokens)):
            break
        if len(block) < block_size:
            block.extend([PAD] * (block_size - len(block)))
        blocks.append(block)
    return blocks


def _batch(blocks: Sequence[Sequence[int]], indices: Sequence[int], *, device: str):
    import torch

    ids = torch.tensor([blocks[i] for i in indices], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[labels == PAD] = -100
    return ids, labels


def corpus_loss(
    model,
    blocks: Sequence[Sequence[int]],
    *,
    device: str = "cpu",
    batch_size: int = 8,
    max_blocks: int | None = None,
    seed: int = 0,
) -> float:
    import torch

    if not blocks:
        raise ValueError("no causal-pretraining blocks")

    indices = list(range(len(blocks)))
    if max_blocks is not None and len(indices) > max(1, int(max_blocks)):
        rng = random.Random(int(seed))
        indices = sorted(rng.sample(indices, max(1, int(max_blocks))))

    model.eval()
    weighted = 0.0
    count = 0
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            ids, labels = _batch(blocks, rows, device=device)
            result = model(ids, labels=labels)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite corpus loss")
            weighted += float(loss.detach().cpu()) * len(rows)
            count += len(rows)
    return weighted / max(1, count)


def pretrain_causal(
    model,
    tokenizer: ByteTokenizer,
    documents: Sequence[CorpusDocument],
    *,
    steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = 17,
    device: str = "cpu",
    max_eval_blocks: int = 128,
) -> dict:
    """Run bounded causal next-token pretraining on a local reviewed corpus."""
    import torch

    blocks = pack_causal_blocks(
        documents,
        tokenizer,
        context_length=model.config.context_length,
    )
    if not blocks:
        raise ValueError("corpus produced no training blocks")

    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device)
    eval_blocks = max(1, min(int(max_eval_blocks), len(blocks)))
    initial_loss = corpus_loss(
        model,
        blocks,
        device=device,
        batch_size=batch_size,
        max_blocks=eval_blocks,
        seed=seed + 1,
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    for _ in range(max(1, int(steps))):
        rows = [rng.randrange(len(blocks)) for _ in range(max(1, int(batch_size)))]
        ids, labels = _batch(blocks, rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=labels)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite pretraining loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    final_loss = corpus_loss(
        model,
        blocks,
        device=device,
        batch_size=batch_size,
        max_blocks=eval_blocks,
        seed=seed + 1,
    )
    return {
        "ok": bool(final_loss < initial_loss),
        "documents": len(documents),
        "blocks": len(blocks),
        "eval_blocks": eval_blocks,
        "steps": max(1, int(steps)),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improvement": initial_loss - final_loss,
        "best_step_loss": min(losses),
        "corpus_bytes": sum(row.bytes for row in documents),
        "corpus_sha256": hashlib.sha256(
            "\n".join(row.sha256 for row in documents).encode("ascii")
        ).hexdigest(),
    }
