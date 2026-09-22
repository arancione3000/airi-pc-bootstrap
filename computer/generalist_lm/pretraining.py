from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from .tokenizer import EOS, PAD, ByteTokenizer
from .training import causal_training_objective


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
    domain: str = "general"


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



class PackedBlockArray(Sequence[Sequence[int]]):
    """Compact fixed-width causal blocks backed by one contiguous integer array."""

    def __init__(self, flat: array, block_size: int):
        self._flat = flat
        self.block_size = int(block_size)
        if self.block_size <= 0 or len(self._flat) % self.block_size:
            raise ValueError("invalid packed block array shape")

    def __len__(self) -> int:
        return len(self._flat) // self.block_size

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(index)
        start = idx * self.block_size
        return list(self._flat[start:start + self.block_size])


def save_packed_block_cache(
    path: str | Path,
    blocks: Sequence[Sequence[int]],
    *,
    block_size: int,
    identity: dict,
) -> dict:
    """Atomically persist deterministic fixed-width token blocks.

    The cache is an acceleration artifact only.  Its sidecar records a SHA-256
    of the binary payload plus the corpus/tokenizer identity supplied by the
    caller, so stale or corrupt cache files fail closed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    max_token = 0
    for block in blocks:
        if len(block) != block_size:
            raise ValueError("all cached blocks must have the fixed block_size")
        if block:
            max_token = max(max_token, max(int(value) for value in block))
    typecode = "H" if max_token <= 0xFFFF else "I"
    flat = array(typecode)
    for block in blocks:
        flat.extend(int(value) for value in block)

    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        flat.tofile(handle)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(target)

    meta = {
        "schema": 1,
        "version": "packed-causal-blocks-v1",
        "block_size": block_size,
        "block_count": len(blocks),
        "typecode": typecode,
        "binary_sha256": digest,
        "identity": dict(identity),
    }
    meta_path = target.with_suffix(target.suffix + ".json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    meta_tmp.replace(meta_path)
    return meta


def load_packed_block_cache(
    path: str | Path,
    *,
    expected_identity: dict,
) -> PackedBlockArray | None:
    """Load a verified packed-block cache or return None when it is stale."""
    target = Path(path)
    meta_path = target.with_suffix(target.suffix + ".json")
    if not target.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("version") != "packed-causal-blocks-v1":
        return None
    if dict(meta.get("identity") or {}) != dict(expected_identity):
        return None
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != str(meta.get("binary_sha256") or ""):
        return None

    typecode = str(meta.get("typecode") or "")
    if typecode not in {"H", "I"}:
        return None
    block_size = int(meta.get("block_size", 0) or 0)
    block_count = int(meta.get("block_count", 0) or 0)
    if block_size <= 0 or block_count <= 0:
        return None

    flat = array(typecode)
    with target.open("rb") as handle:
        flat.fromfile(handle, block_size * block_count)
    if len(flat) != block_size * block_count:
        return None
    return PackedBlockArray(flat, block_size)


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
    domain_weights: dict[str, float] | None = None,
    repetition_unlikelihood_weight: float = 0.0,
    eos_loss_weight: float = 1.0,
    repetition_window: int = 16,
) -> dict:
    """Run bounded causal next-token pretraining on a local reviewed corpus."""
    import torch

    grouped_documents: dict[str, list[CorpusDocument]] = {}
    for document in documents:
        domain = str(getattr(document, "domain", "general") or "general")
        grouped_documents.setdefault(domain, []).append(document)

    blocks: list[list[int]] = []
    domain_indices: dict[str, list[int]] = {}
    domain_block_counts: dict[str, int] = {}
    for domain in sorted(grouped_documents):
        packed = pack_causal_blocks(
            grouped_documents[domain],
            tokenizer,
            context_length=model.config.context_length,
        )
        if not packed:
            continue
        start = len(blocks)
        blocks.extend(packed)
        domain_indices[domain] = list(range(start, len(blocks)))
        domain_block_counts[domain] = len(packed)

    if not blocks:
        raise ValueError("corpus produced no training blocks")

    clean_domain_weights = {
        domain: max(
            0.05,
            float((domain_weights or {}).get(domain, 1.0)),
        )
        for domain in sorted(domain_indices)
    }
    weight_total = sum(clean_domain_weights.values()) or 1.0
    normalized_domain_weights = {
        domain: value / weight_total
        for domain, value in clean_domain_weights.items()
    }
    sampling_domains = sorted(domain_indices)
    sampling_weights = [clean_domain_weights[domain] for domain in sampling_domains]

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
    last_objective_stats = {
        "causal_ce_loss": float(initial_loss),
        "repetition_unlikelihood_loss": 0.0,
        "repetition_unlikelihood_weight": float(repetition_unlikelihood_weight),
        "repetition_negative_count": 0,
        "eos_loss_weight": float(eos_loss_weight),
    }
    for _ in range(max(1, int(steps))):
        chosen_domains = rng.choices(
            sampling_domains,
            weights=sampling_weights,
            k=max(1, int(batch_size)),
        )
        rows = [
            rng.choice(domain_indices[domain])
            for domain in chosen_domains
        ]
        ids, labels = _batch(blocks, rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        result = model(ids)
        loss, objective_stats = causal_training_objective(
            result["logits"],
            labels,
            ids,
            eos_loss_weight=eos_loss_weight,
            repetition_unlikelihood_weight=repetition_unlikelihood_weight,
            repetition_window=repetition_window,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite pretraining loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        last_objective_stats = objective_stats

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
        "domain_blocks": domain_block_counts,
        "sampling_domain_weights": normalized_domain_weights,
        "eval_blocks": eval_blocks,
        "steps": max(1, int(steps)),
        "learning_rate": float(learning_rate),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improvement": initial_loss - final_loss,
        "best_step_loss": min(losses),
        "objective": last_objective_stats,
        "corpus_bytes": sum(row.bytes for row in documents),
        "corpus_sha256": hashlib.sha256(
            "\n".join(row.sha256 for row in documents).encode("ascii")
        ).hexdigest(),
    }
