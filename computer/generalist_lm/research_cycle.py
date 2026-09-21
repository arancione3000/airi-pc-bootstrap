from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from .curriculum import ResearchRow, train_rows, validation_rows
from .curriculum_memory import CurriculumMemory, canary_rows
from .corpus import repository_corpus
from .bpe_tokenizer import BPETokenizer, train_bpe
from .evolution import GeneralistGenome, generate_challengers
from .mathesis_bridge import mathesis_signals
from .model import CausalTransformerLM, estimate_parameter_count, parameter_count
from .runtime import GeneralistRuntime
from .pretraining import CorpusDocument, pretrain_causal
from .tokenizer import BYTE_OFFSET, VOCAB_SIZE as BYTE_VOCAB_SIZE, ByteTokenizer
from .training import encode_sft_example, loss_on_examples, nll_stats_on_examples, train_sft


def research_seed() -> GeneralistGenome:
    return GeneralistGenome(
        generation=0,
        parent_id=None,
        genome_id="generalist-research-0-seed",
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        learning_rate=3e-3,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _teacher_forced_accuracy(model, tokenizer: ByteTokenizer, rows: list[ResearchRow], *, device: str) -> dict[str, Any]:
    import hashlib
    import torch

    encoded = [encode_sft_example(row.sft(), tokenizer, model.config.context_length) for row in rows]
    ids = torch.stack([item[0] for item in encoded]).to(device)
    labels = torch.stack([item[1] for item in encoded]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(ids)["logits"]
    predicted = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    mask = targets != -100
    correct = (predicted == targets) & mask

    total = int(mask.sum().item())
    token_accuracy = float(correct.sum().item() / total) if total else 0.0
    solved: list[str] = []
    domain_correct: dict[str, int] = {}
    domain_total: dict[str, int] = {}
    for index, row in enumerate(rows):
        row_mask = mask[index]
        row_correct = correct[index]
        count = int(row_mask.sum().item())
        good = int(row_correct.sum().item())
        domain_correct[row.domain] = domain_correct.get(row.domain, 0) + good
        domain_total[row.domain] = domain_total.get(row.domain, 0) + count
        if count and bool(torch.all(row_correct[row_mask]).item()):
            prompt = row.messages[0]["content"]
            digest = hashlib.sha256(f"{row.domain}\0{prompt}".encode("utf-8")).hexdigest()[:16]
            solved.append(f"{row.domain}:{digest}")

    return {
        "target_token_accuracy": token_accuracy,
        "domain_token_accuracy": {
            domain: (domain_correct[domain] / domain_total[domain] if domain_total[domain] else 0.0)
            for domain in sorted(domain_total)
        },
        "solved_items": sorted(solved),
        "target_tokens": total,
    }


def _generation_probe(
    model,
    tokenizer: ByteTokenizer,
    rows: list[ResearchRow],
    *,
    device: str,
) -> dict[str, Any]:
    """Probe real autoregressive decoding on held-out examples.

    One deterministic row per domain keeps the research loop bounded while
    ensuring teacher-forced loss cannot be the only promotion signal.
    """
    selected: list[ResearchRow] = []
    seen_domains: set[str] = set()
    for row in rows:
        if row.domain in seen_domains:
            continue
        seen_domains.add(row.domain)
        selected.append(row)

    runtime = GeneralistRuntime(model, model.config, tokenizer, device=device)
    solved: list[str] = []
    domain_accuracy: dict[str, float] = {}
    outputs: list[dict[str, Any]] = []

    for row in selected:
        target = str(row.messages[-1]["content"]).strip()
        prompt_messages = row.messages[:-1]
        max_new_tokens = min(
            96,
            max(4, len(tokenizer.encode(target, eos=True)) + 2),
        )
        try:
            output = runtime.chat(
                prompt_messages,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            ).strip()
            ok = output == target
        except Exception as exc:
            output = f"<generation-error:{type(exc).__name__}>"
            ok = False

        prompt = str(row.messages[0]["content"])
        digest = hashlib.sha256(
            f"{row.domain}\0{prompt}".encode("utf-8")
        ).hexdigest()[:16]
        if ok:
            solved.append(f"{row.domain}:{digest}")
        domain_accuracy[row.domain] = 1.0 if ok else 0.0
        outputs.append({
            "domain": row.domain,
            "item": f"{row.domain}:{digest}",
            "ok": ok,
            "target": target[:500],
            "output": output[:500],
        })

    accuracy = (
        sum(1 for row in outputs if row["ok"]) / len(outputs)
        if outputs else 0.0
    )
    return {
        "generation_exact_accuracy": float(accuracy),
        "domain_generation_accuracy": domain_accuracy,
        "generated_solved_items": sorted(solved),
        "generation_probe": outputs,
    }


def _grouped_validation(model, tokenizer, rows: list[ResearchRow], *, device: str = "cpu") -> dict[str, Any]:
    all_examples = [row.sft() for row in rows]
    overall_stats = nll_stats_on_examples(
        model,
        tokenizer,
        all_examples,
        device=device,
    )
    domains: dict[str, float] = {}
    domain_nll_per_byte: dict[str, float] = {}
    domain_bits_per_byte: dict[str, float] = {}
    for domain in sorted({row.domain for row in rows}):
        examples = [row.sft() for row in rows if row.domain == domain]
        stats = nll_stats_on_examples(
            model,
            tokenizer,
            examples,
            device=device,
        )
        domains[domain] = float(stats["loss_per_token"])
        domain_nll_per_byte[domain] = float(stats["nll_per_byte"])
        domain_bits_per_byte[domain] = float(stats["bits_per_byte"])
    accuracy = _teacher_forced_accuracy(model, tokenizer, rows, device=device)
    generation = _generation_probe(model, tokenizer, rows, device=device)
    finite_values = [
        float(overall_stats["loss_per_token"]),
        float(overall_stats["nll_per_byte"]),
        float(overall_stats["bits_per_byte"]),
        *domains.values(),
        *domain_nll_per_byte.values(),
    ]
    return {
        "tokenizer_version": str(getattr(tokenizer, "version", "unknown")),
        "loss": float(overall_stats["loss_per_token"]),
        "nll_per_byte": float(overall_stats["nll_per_byte"]),
        "bits_per_byte": float(overall_stats["bits_per_byte"]),
        "validation_target_tokens": int(overall_stats["target_tokens"]),
        "validation_target_bytes": int(overall_stats["target_bytes"]),
        "domain_loss": domains,
        "domain_nll_per_byte": domain_nll_per_byte,
        "domain_bits_per_byte": domain_bits_per_byte,
        **accuracy,
        **generation,
        "finite": bool(all(math.isfinite(v) for v in finite_values)),
    }

def _research_score(report: dict[str, Any], params: int) -> float:
    quality = float(report.get("nll_per_byte", report["loss"]))
    efficiency_penalty = min(1.0, params / 10_000_000.0) * 0.01
    return float(100.0 / (1.0 + quality) - efficiency_penalty)


def _genome_training_seed(genome: GeneralistGenome, *, namespace: str = "candidate") -> int:
    payload = genome.to_dict()
    for key in ("genome_id", "parent_id", "generation"):
        payload.pop(key, None)
    raw = (namespace + "\0" + json.dumps(payload, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    return 1 + (value % 2_000_000_000)


def _research_eligible(
    champion: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_loss_gain: float,
    max_domain_regression: float,
) -> tuple[bool, str]:
    if not candidate.get("finite"):
        return False, "candidate validation is non-finite"

    old_quality = float(champion.get("nll_per_byte", champion["loss"]))
    new_quality = float(candidate.get("nll_per_byte", candidate["loss"]))
    if old_quality - new_quality < float(minimum_loss_gain):
        return False, "candidate did not reduce held-out NLL per byte by the research margin"

    old_domain_quality = champion.get("domain_nll_per_byte") or champion.get("domain_loss") or {}
    new_domain_quality = candidate.get("domain_nll_per_byte") or candidate.get("domain_loss") or {}
    for domain, old_value in old_domain_quality.items():
        if domain not in new_domain_quality:
            return False, f"candidate lost validation domain: {domain}"
        if float(new_domain_quality[domain]) > float(old_value) + float(max_domain_regression):
            return False, f"candidate regressed in byte-normalized validation domain: {domain}"

    same_tokenizer = (
        str(champion.get("tokenizer_version", "byte-v1"))
        == str(candidate.get("tokenizer_version", "byte-v1"))
    )
    # Token accuracy is meaningful within one tokenizer family, but is not a
    # fair cross-tokenizer comparison because segmentation changes.
    if same_tokenizer:
        old_accuracy = float(champion.get("target_token_accuracy", 0.0))
        new_accuracy = float(candidate.get("target_token_accuracy", 0.0))
        if new_accuracy + 0.01 < old_accuracy:
            return False, "candidate regressed in held-out target-token accuracy"

        old_domains = champion.get("domain_token_accuracy") or {}
        new_domains = candidate.get("domain_token_accuracy") or {}
        for domain, old_value in old_domains.items():
            if domain not in new_domains:
                return False, f"candidate lost token-accuracy domain: {domain}"
            if float(new_domains[domain]) + 0.02 < float(old_value):
                return False, f"candidate regressed in target-token domain: {domain}"

    remembered = set(champion.get("solved_items") or [])
    retained = set(candidate.get("solved_items") or [])
    forgotten = sorted(remembered - retained)
    if forgotten:
        return False, f"candidate forgot {len(forgotten)} previously solved held-out items"

    old_generation = float(champion.get("generation_exact_accuracy", 0.0))
    new_generation = float(candidate.get("generation_exact_accuracy", 0.0))
    if new_generation + 1e-12 < old_generation:
        return False, "candidate regressed in held-out autoregressive generation"

    old_generation_domains = champion.get("domain_generation_accuracy") or {}
    new_generation_domains = candidate.get("domain_generation_accuracy") or {}
    for domain, old_value in old_generation_domains.items():
        if domain not in new_generation_domains:
            return False, f"candidate lost generation domain: {domain}"
        if float(new_generation_domains[domain]) + 1e-12 < float(old_value):
            return False, f"candidate regressed in autoregressive generation domain: {domain}"

    generated_remembered = set(champion.get("generated_solved_items") or [])
    generated_retained = set(candidate.get("generated_solved_items") or [])
    generated_forgotten = sorted(generated_remembered - generated_retained)
    if generated_forgotten:
        return False, f"candidate forgot {len(generated_forgotten)} generated held-out items"

    old_canary = champion.get("canary")
    new_canary = candidate.get("canary")
    if isinstance(old_canary, dict):
        if not isinstance(new_canary, dict):
            return False, "candidate is missing the rotating held-out canary"
        if not new_canary.get("finite"):
            return False, "candidate rotating canary validation is non-finite"

        old_canary_quality = float(old_canary.get("nll_per_byte", old_canary.get("loss", 0.0)))
        new_canary_quality = float(new_canary.get("nll_per_byte", new_canary.get("loss", float("inf"))))
        if new_canary_quality > old_canary_quality + float(max_domain_regression):
            return False, "candidate regressed on rotating canary NLL per byte"

        old_canary_domains = old_canary.get("domain_nll_per_byte") or old_canary.get("domain_loss") or {}
        new_canary_domains = new_canary.get("domain_nll_per_byte") or new_canary.get("domain_loss") or {}
        for domain, old_value in old_canary_domains.items():
            if domain not in new_canary_domains:
                return False, f"candidate lost rotating canary domain: {domain}"
            if float(new_canary_domains[domain]) > float(old_value) + float(max_domain_regression):
                return False, f"candidate regressed on rotating canary domain: {domain}"

        old_canary_generation = float(old_canary.get("generation_exact_accuracy", 0.0))
        new_canary_generation = float(new_canary.get("generation_exact_accuracy", 0.0))
        if new_canary_generation + 1e-12 < old_canary_generation:
            return False, "candidate regressed on rotating canary generation"
        old_canary_solved = set(old_canary.get("generated_solved_items") or [])
        new_canary_solved = set(new_canary.get("generated_solved_items") or [])
        if old_canary_solved - new_canary_solved:
            return False, "candidate forgot a solved rotating canary item"

    return True, "held-out NLL/byte, rotating canary, generation, and anti-forgetting gates passed"

def _weaknesses(report: dict[str, Any]) -> list[str]:
    domains = report.get("domain_nll_per_byte") or report.get("domain_loss") or {}
    if not domains:
        return []
    ordered = sorted(domains.items(), key=lambda item: float(item[1]), reverse=True)
    signals: list[str] = []
    for domain, _loss in ordered[:3]:
        if domain == "coding":
            signals.append("coding_gap")
        elif domain == "data":
            signals.append("data_gap")
        elif domain == "tools":
            signals.append("tool_gap")
        elif domain == "reasoning":
            signals.append("reasoning_gap")
    return signals


def _champion_paths(root: Path) -> tuple[Path, Path]:
    return root / "champion", root / "champion-genome.json"


def _load_champion(root: Path, *, device: str) -> tuple[GeneralistGenome, GeneralistRuntime] | None:
    checkpoint, genome_path = _champion_paths(root)
    try:
        genome_raw = json.loads(genome_path.read_text(encoding="utf-8"))
        genome = GeneralistGenome(**genome_raw).validate()
        runtime = GeneralistRuntime.from_checkpoint(checkpoint, device=device)
        return genome, runtime
    except Exception:
        return None


def _save_champion(root: Path, genome: GeneralistGenome, runtime: GeneralistRuntime, metrics: dict[str, Any]) -> None:
    checkpoint, genome_path = _champion_paths(root)
    tmp = root / ".champion-next"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    runtime.save_checkpoint(tmp, metadata={
        "role": "research_champion",
        "genome_id": genome.genome_id,
        "generation": genome.generation,
        "production_qualified": False,
    })
    _atomic_json(tmp / "research-metrics.json", metrics)
    backup = root / ".champion-old"
    shutil.rmtree(backup, ignore_errors=True)
    if checkpoint.exists():
        checkpoint.replace(backup)
    tmp.replace(checkpoint)
    shutil.rmtree(backup, ignore_errors=True)
    _atomic_json(genome_path, genome.to_dict())


def _domain_balanced_rows(rows: list[ResearchRow]) -> list[ResearchRow]:
    """Equalize replay mass across observed generalist domains deterministically.

    Persistent curriculum growth is signal-directed and therefore intentionally
    uneven. Before SFT, equalizing the base replay prevents accidental domain
    frequency from becoming a hidden forgetting pressure. Explicit genome focus
    genes are applied only after this neutral baseline is constructed.
    """
    grouped: dict[str, list[ResearchRow]] = {}
    for row in rows:
        grouped.setdefault(row.domain, []).append(row)
    if not grouped:
        return []
    target = max(len(items) for items in grouped.values())
    balanced: list[ResearchRow] = []
    for domain in sorted(grouped):
        items = grouped[domain]
        balanced.extend(items[index % len(items)] for index in range(target))
    return balanced


def adaptive_domain_weights(report: dict[str, Any] | None) -> dict[str, float]:
    """Turn held-out weakness into bounded replay weights.

    The best domain remains at weight 1.0; weaker domains receive up to 3x
    replay. This is deliberately derived from held-out loss only and cannot
    lower any validation/promotion gate.
    """
    if not isinstance(report, dict):
        return {}
    domains = report.get("domain_nll_per_byte") or report.get("domain_loss") or {}
    clean = {
        str(domain): float(value)
        for domain, value in domains.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    if not clean:
        return {}
    best = min(clean.values())
    worst = max(clean.values())
    span = max(1e-9, worst - best)
    return {
        domain: 1.0 + 2.0 * ((value - best) / span)
        for domain, value in clean.items()
    }


def _training_rows_for_genome(
    genome: GeneralistGenome,
    replay_rows: list[ResearchRow] | None = None,
    *,
    domain_weights: dict[str, float] | None = None,
) -> list[ResearchRow]:
    """Turn strategy genes and measured weakness into replay weighting."""
    base = _domain_balanced_rows(list(train_rows()) + list(replay_rows or []))
    extra: list[ResearchRow] = []
    if genome.code_adapter:
        extra.extend(row for row in base if row.domain == "coding")
    if genome.data_adapter:
        extra.extend(row for row in base if row.domain == "data")
    if genome.retrieval_adapter:
        extra.extend(row for row in base if row.domain in {"tools", "structured"})
    if genome.symbolic_adapter:
        extra.extend(row for row in base if row.domain == "reasoning")
    for _ in range(max(0, int(genome.reasoning_depth) - 1)):
        extra.extend(row for row in base if row.domain == "reasoning")

    weighted: list[ResearchRow] = []
    for domain, weight in sorted((domain_weights or {}).items()):
        repeats = max(0, min(2, int(math.floor(max(1.0, float(weight)))) - 1))
        if repeats:
            rows = [row for row in base if row.domain == domain]
            for _ in range(repeats):
                weighted.extend(rows)
    return base + extra + weighted


def _tokenizer_training_texts(
    replay_rows: list[ResearchRow] | None,
    pretrain_documents: list[CorpusDocument] | None,
) -> list[str]:
    """Build BPE text only from allowed training/repository sources.

    Protected validation, qualification and rotating canary rows are never
    passed here.
    """
    texts: list[str] = []
    for row in list(train_rows()) + list(replay_rows or []):
        for message in row.messages:
            texts.append(str(message.get("content", "")))
    for document in list(pretrain_documents or []):
        texts.append(str(document.text))
    return [text for text in texts if text]


def _tokenizer_for_genome(
    genome: GeneralistGenome,
    *,
    source_runtime: GeneralistRuntime | None = None,
    replay_rows: list[ResearchRow] | None = None,
    pretrain_documents: list[CorpusDocument] | None = None,
    bpe_vocab_size: int = 384,
    bpe_max_bytes: int = 256_000,
):
    if source_runtime is not None and source_runtime.tokenizer.version == genome.tokenizer_version:
        return source_runtime.tokenizer
    if genome.tokenizer_version == "byte-v1":
        return ByteTokenizer()
    if genome.tokenizer_version == "bpe-v1":
        return train_bpe(
            _tokenizer_training_texts(replay_rows, pretrain_documents),
            vocab_size=max(BYTE_VOCAB_SIZE, int(bpe_vocab_size)),
            min_frequency=2,
            max_bytes=max(8_192, int(bpe_max_bytes)),
        )
    raise ValueError(f"unsupported research tokenizer: {genome.tokenizer_version}")


def _research_budget_reason(
    genome: GeneralistGenome,
    *,
    max_params: int,
    max_context: int,
    max_width: int,
    max_layers: int,
    vocab_size: int = BYTE_VOCAB_SIZE,
) -> str | None:
    if genome.context_length > max_context:
        return f"context_length {genome.context_length} exceeds research max {max_context}"
    if genome.d_model > max_width:
        return f"d_model {genome.d_model} exceeds research max {max_width}"
    if genome.n_layers > max_layers:
        return f"n_layers {genome.n_layers} exceeds research max {max_layers}"
    estimated = estimate_parameter_count(genome.model_config(vocab_size))
    if estimated > max_params:
        return f"estimated parameters {estimated} exceed research max {max_params}"
    return None


def _transfer_compatible_weights(
    source_model,
    target_model,
    *,
    source_tokenizer=None,
    target_tokenizer=None,
) -> dict[str, Any]:
    """Transfer learned structure into a larger compatible Generalist model.

    Exact tensors are copied directly. For same-width progressive growth,
    SwiGLU/GELU FFN expansion is layout-aware and new residual blocks are
    initialized as identities by zeroing their output projections. This avoids
    the semantic corruption caused by naive prefix copies of concatenated
    QKV/SwiGLU tensors.
    """
    source = source_model.state_dict()
    target = target_model.state_dict()
    copied: dict[str, Any] = {}
    copied_params = 0
    total_params = sum(int(tensor.numel()) for tensor in target.values())

    source_version = str(getattr(source_tokenizer, "version", "unknown"))
    target_version = str(getattr(target_tokenizer, "version", "unknown"))
    source_digest = getattr(source_tokenizer, "digest", None)
    target_digest = getattr(target_tokenizer, "digest", None)
    tokenizer_identical = (
        source_version == target_version
        and (source_version != "bpe-v1" or source_digest == target_digest)
    )
    vocab_names = {"token_embedding.weight", "lm_head.weight"}

    source_cfg = getattr(source_model, "config", None)
    target_cfg = getattr(target_model, "config", None)
    same_width = bool(
        source_cfg is not None
        and target_cfg is not None
        and int(source_cfg.d_model) == int(target_cfg.d_model)
    )
    same_ff_variant = bool(
        source_cfg is not None
        and target_cfg is not None
        and str(source_cfg.ff_variant) == str(target_cfg.ff_variant)
    )

    structured_growth_tensors: list[str] = []
    identity_initialized_tensors: list[str] = []

    for name, tensor in target.items():
        old = source.get(name)
        if old is None:
            continue
        if name in vocab_names and not tokenizer_identical:
            continue

        if tuple(old.shape) == tuple(tensor.shape):
            copied[name] = old.detach().to(
                device=tensor.device,
                dtype=tensor.dtype,
            ).clone()
            copied_params += int(tensor.numel())
            continue

        # Context growth is semantically aligned by position row. New positions
        # start neutral while every learned old position is retained exactly.
        if (
            name == "position_embedding.weight"
            and old.ndim == tensor.ndim == 2
            and int(old.shape[1]) == int(tensor.shape[1])
            and int(tensor.shape[0]) > int(old.shape[0])
        ):
            migrated = tensor.detach().new_zeros(tensor.shape)
            migrated[: int(old.shape[0]), :] = old.detach().to(
                device=migrated.device,
                dtype=migrated.dtype,
            )
            copied[name] = migrated
            copied_params += int(old.numel())
            structured_growth_tensors.append(name)
            continue

        # Same-width FFN growth can preserve the existing residual function.
        # SwiGLU stores [gate ; value] in one tensor, so each half must be
        # copied into its corresponding half in the larger tensor rather than
        # using a flat prefix.
        if (
            same_width
            and same_ff_variant
            and name.endswith(".ff.up.weight")
            and old.ndim == tensor.ndim == 2
            and int(old.shape[1]) == int(tensor.shape[1])
            and int(tensor.shape[0]) > int(old.shape[0])
        ):
            migrated = tensor.detach().new_zeros(tensor.shape)
            if str(target_cfg.ff_variant) == "swiglu":
                if int(old.shape[0]) % 2 or int(tensor.shape[0]) % 2:
                    continue
                old_ff = int(old.shape[0]) // 2
                new_ff = int(tensor.shape[0]) // 2
                migrated[:old_ff, :] = old[:old_ff, :].detach().to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                )
                migrated[new_ff:new_ff + old_ff, :] = old[old_ff:, :].detach().to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                )
            else:
                migrated[: int(old.shape[0]), :] = old.detach().to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                )
            copied[name] = migrated
            copied_params += int(old.numel())
            structured_growth_tensors.append(name)
            continue

        if (
            same_width
            and same_ff_variant
            and name.endswith(".ff.up.bias")
            and old.ndim == tensor.ndim == 1
            and int(tensor.shape[0]) > int(old.shape[0])
        ):
            migrated = tensor.detach().new_zeros(tensor.shape)
            if str(target_cfg.ff_variant) == "swiglu":
                if int(old.shape[0]) % 2 or int(tensor.shape[0]) % 2:
                    continue
                old_ff = int(old.shape[0]) // 2
                new_ff = int(tensor.shape[0]) // 2
                migrated[:old_ff] = old[:old_ff].detach().to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                )
                migrated[new_ff:new_ff + old_ff] = old[old_ff:].detach().to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                )
            else:
                migrated[: int(old.shape[0])] = old.detach().to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                )
            copied[name] = migrated
            copied_params += int(old.numel())
            structured_growth_tensors.append(name)
            continue

        if (
            same_width
            and same_ff_variant
            and name.endswith(".ff.down.weight")
            and old.ndim == tensor.ndim == 2
            and int(old.shape[0]) == int(tensor.shape[0])
            and int(tensor.shape[1]) > int(old.shape[1])
        ):
            migrated = tensor.detach().new_zeros(tensor.shape)
            migrated[:, : int(old.shape[1])] = old.detach().to(
                device=migrated.device,
                dtype=migrated.dtype,
            )
            copied[name] = migrated
            copied_params += int(old.numel())
            structured_growth_tensors.append(name)
            continue

    # A newly inserted Transformer block is made an exact residual identity at
    # initialization: attention and FFN branches may compute internal values,
    # but their output projections are zero so x -> x. This lets added depth
    # learn gradually instead of destroying the champion before training.
    source_layers = int(getattr(source_cfg, "n_layers", 0) or 0)
    target_layers = int(getattr(target_cfg, "n_layers", 0) or 0)
    if same_width and target_layers > source_layers:
        for layer in range(source_layers, target_layers):
            for suffix in (
                "attn.out.weight",
                "attn.out.bias",
                "ff.down.weight",
                "ff.down.bias",
            ):
                name = f"blocks.{layer}.{suffix}"
                tensor = target.get(name)
                if tensor is None:
                    continue
                copied[name] = tensor.detach().new_zeros(tensor.shape)
                identity_initialized_tensors.append(name)

    shared_token_rows = 0
    derived_bpe_rows = 0
    embedding_width_migrated = False
    source_embed = source.get("token_embedding.weight")
    target_embed = target.get("token_embedding.weight")

    # Width growth is not claimed function-preserving because normalization
    # spans the model width, but retaining old embedding coordinates is still
    # preferable to randomizing known tokens. New coordinates start at zero.
    if (
        tokenizer_identical
        and source_embed is not None
        and target_embed is not None
        and source_embed.ndim == 2
        and target_embed.ndim == 2
        and int(target_embed.shape[0]) == int(source_embed.shape[0])
        and int(target_embed.shape[1]) > int(source_embed.shape[1])
    ):
        migrated = target_embed.detach().new_zeros(target_embed.shape)
        migrated[:, : int(source_embed.shape[1])] = source_embed.detach().to(
            device=migrated.device,
            dtype=migrated.dtype,
        )
        copied["token_embedding.weight"] = migrated
        if (
            "lm_head.weight" in target
            and tuple(target["lm_head.weight"].shape) == tuple(migrated.shape)
        ):
            copied["lm_head.weight"] = migrated.clone()
        copied_params += int(source_embed.numel())
        embedding_width_migrated = True

    if (
        source_embed is not None
        and target_embed is not None
        and source_embed.ndim == 2
        and target_embed.ndim == 2
        and source_embed.shape[1] == target_embed.shape[1]
        and not tokenizer_identical
    ):
        migrated = target_embed.detach().clone()
        shared_token_rows = min(
            BYTE_VOCAB_SIZE,
            int(source_embed.shape[0]),
            int(target_embed.shape[0]),
        )
        if shared_token_rows:
            migrated[:shared_token_rows] = source_embed[:shared_token_rows].to(
                device=migrated.device,
                dtype=migrated.dtype,
            )

        if isinstance(target_tokenizer, BPETokenizer):
            for token_id in range(BYTE_VOCAB_SIZE, int(target_embed.shape[0])):
                raw = target_tokenizer.token_bytes(token_id)
                byte_rows = [
                    BYTE_OFFSET + value
                    for value in raw
                    if BYTE_OFFSET + value < int(source_embed.shape[0])
                ]
                if not byte_rows:
                    continue
                migrated[token_id] = source_embed[byte_rows].to(
                    device=migrated.device,
                    dtype=migrated.dtype,
                ).mean(dim=0)
                derived_bpe_rows += 1

        copied["token_embedding.weight"] = migrated
        if (
            "lm_head.weight" in target
            and tuple(target["lm_head.weight"].shape) == tuple(migrated.shape)
        ):
            copied["lm_head.weight"] = migrated.clone()

        migrated_params = int(migrated.numel())
        copied_params += migrated_params
        if "lm_head.weight" in copied:
            copied_params += migrated_params

    target_model.load_state_dict(copied, strict=False)
    source_unmatched = sorted(
        name
        for name, tensor in source.items()
        if name not in target
        or (
            tuple(target[name].shape) != tuple(tensor.shape)
            and name not in copied
        )
    )
    function_preserving_growth = bool(
        tokenizer_identical
        and same_width
        and same_ff_variant
        and source_cfg is not None
        and target_cfg is not None
        and int(target_cfg.n_layers) >= int(source_cfg.n_layers)
        and int(target_cfg.d_ff) >= int(source_cfg.d_ff)
        and int(target_cfg.context_length) >= int(source_cfg.context_length)
        and int(target_cfg.n_heads) == int(source_cfg.n_heads)
        and str(target_cfg.norm_type) == str(source_cfg.norm_type)
        and str(target_cfg.position_encoding) == str(source_cfg.position_encoding)
    )
    return {
        "copied_tensors": len(copied),
        "source_tensors": len(source),
        "target_tensors": len(target),
        "source_unmatched_tensors": source_unmatched,
        "copied_parameters": copied_params,
        "target_parameters": total_params,
        "parameter_fraction": (
            float(min(copied_params, total_params) / total_params)
            if total_params else 0.0
        ),
        "source_tokenizer": source_version,
        "target_tokenizer": target_version,
        "tokenizer_identical": tokenizer_identical,
        "shared_token_rows": shared_token_rows,
        "derived_bpe_rows": derived_bpe_rows,
        "embedding_width_migrated": embedding_width_migrated,
        "vocabulary_migrated": bool(
            shared_token_rows or derived_bpe_rows or embedding_width_migrated
        ),
        "partial_prefix_tensors": sorted(structured_growth_tensors),
        "partial_prefix_parameters": sum(
            int(source[name].numel())
            for name in structured_growth_tensors
            if name in source
        ),
        "identity_initialized_tensors": sorted(identity_initialized_tensors),
        "function_preserving_growth": function_preserving_growth,
        "policy": (
            "layout-aware Net2Grow transfer with identity residual depth expansion"
            if tokenizer_identical
            else "exact compatible tensors plus deterministic byte-compatible vocabulary migration"
        ),
    }

def _train_genome(
    genome: GeneralistGenome,
    *,
    steps: int,
    seed: int,
    device: str,
    replay_rows: list[ResearchRow] | None = None,
    domain_weights: dict[str, float] | None = None,
    source_model=None,
    source_tokenizer=None,
    tokenizer=None,
    gradient_accumulation_steps: int = 1,
    precision: str = "fp32",
    pretrain_documents: list[CorpusDocument] | None = None,
    pretrain_steps: int = 0,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    tokenizer = tokenizer or ByteTokenizer()
    model = CausalTransformerLM(genome.model_config(tokenizer.vocab_size))
    transfer = (
        _transfer_compatible_weights(
            source_model,
            model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=tokenizer,
        )
        if source_model is not None
        else {
            "copied_tensors": 0,
            "source_tensors": 0,
            "target_tensors": len(model.state_dict()),
            "source_unmatched_tensors": [],
            "copied_parameters": 0,
            "target_parameters": parameter_count(model),
            "parameter_fraction": 0.0,
            "policy": "fresh initialization",
        }
    )
    if pretrain_documents and int(pretrain_steps) > 0:
        pretraining = pretrain_causal(
            model,
            tokenizer,
            pretrain_documents,
            steps=int(pretrain_steps),
            batch_size=4,
            learning_rate=min(1e-3, max(1e-5, genome.learning_rate * 0.5)),
            weight_decay=0.01,
            seed=seed + 101,
            device=device,
        )
    else:
        pretraining = {
            "ok": True,
            "skipped": True,
            "reason": "grounded pretraining disabled or corpus empty",
            "steps": 0,
        }
    report = train_sft(
        model,
        tokenizer,
        [
            row.sft()
            for row in _training_rows_for_genome(
                genome,
                replay_rows,
                domain_weights=domain_weights,
            )
        ],
        steps=steps,
        batch_size=4,
        learning_rate=genome.learning_rate,
        weight_decay=0.0,
        seed=seed,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
    )
    runtime = GeneralistRuntime(model, genome.model_config(tokenizer.vocab_size), tokenizer, device=device)
    validation = _grouped_validation(runtime.model, tokenizer, validation_rows(), device=device)
    validation["parameters"] = parameter_count(runtime.model)
    validation["score"] = _research_score(validation, validation["parameters"])
    validation["training"] = report
    validation["pretraining"] = pretraining
    validation["weight_transfer"] = transfer
    return runtime, validation


def _continual_candidate_genome(champion: GeneralistGenome, cycle: int) -> GeneralistGenome:
    tag = hashlib.sha256(
        f"{champion.genome_id}\0{int(cycle)}\0continual".encode("utf-8")
    ).hexdigest()[:8]
    return replace(
        champion,
        generation=champion.generation + 1,
        parent_id=champion.genome_id,
        genome_id=f"generalist-{champion.generation + 1}-continual-{tag}",
    ).validate()


def _continue_champion(
    champion_genome: GeneralistGenome,
    champion_runtime: GeneralistRuntime,
    replay_rows: list[ResearchRow],
    *,
    steps: int,
    seed: int,
    device: str,
    cycle: int,
    gradient_accumulation_steps: int = 1,
    precision: str = "fp32",
    pretrain_documents: list[CorpusDocument] | None = None,
    pretrain_steps: int = 0,
) -> tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]]:
    """Fine-tune a copy of the current champion with full replay.

    This is the cumulative-learning path. Architecture challengers still train
    independently, but knowledge can now improve without forcing a topology
    change every generation.
    """
    genome = _continual_candidate_genome(champion_genome, cycle)
    tokenizer = champion_runtime.tokenizer
    model = copy.deepcopy(champion_runtime.model)
    if pretrain_documents and int(pretrain_steps) > 0:
        pretraining = pretrain_causal(
            model,
            tokenizer,
            pretrain_documents,
            steps=int(pretrain_steps),
            batch_size=4,
            learning_rate=min(1e-3, max(1e-5, genome.learning_rate * 0.5)),
            weight_decay=0.01,
            seed=seed + 101,
            device=device,
        )
    else:
        pretraining = {
            "ok": True,
            "skipped": True,
            "reason": "grounded pretraining disabled or corpus empty",
            "steps": 0,
        }
    report = train_sft(
        model,
        tokenizer,
        [row.sft() for row in _training_rows_for_genome(genome, replay_rows)],
        steps=steps,
        batch_size=4,
        learning_rate=genome.learning_rate,
        weight_decay=0.0,
        seed=seed,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
    )
    runtime = GeneralistRuntime(model, genome.model_config(tokenizer.vocab_size), tokenizer, device=device)
    validation = _grouped_validation(runtime.model, tokenizer, validation_rows(), device=device)
    validation["parameters"] = parameter_count(runtime.model)
    validation["score"] = _research_score(validation, validation["parameters"])
    validation["training"] = report
    validation["pretraining"] = pretraining
    validation["continual_learning"] = True
    return genome, runtime, validation


def run_research_cycle(state_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(state_dir or os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("AIRI_GENERALIST_RESEARCH_DEVICE", "cpu")
    precision = os.environ.get("AIRI_GENERALIST_RESEARCH_PRECISION", "fp32").strip().lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("AIRI_GENERALIST_RESEARCH_PRECISION must be fp32, bf16, or fp16")
    gradient_accumulation_steps = max(
        1,
        min(
            int(os.environ.get("AIRI_GENERALIST_RESEARCH_GRADIENT_ACCUMULATION", "1")),
            64,
        ),
    )
    steps = max(2, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_STEPS", "20")), 500))
    bootstrap_steps = max(2, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "30")), 500))
    challenger_count = max(1, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "2")), 6))
    minimum_loss_gain = max(0.0, float(os.environ.get("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "0.02")))
    max_domain_regression = max(0.0, float(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_DOMAIN_REGRESSION", "0.10")))
    max_params = max(100_000, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_PARAMS", "5000000")))
    max_context = max(64, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_CONTEXT", "512")))
    max_width = max(32, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_WIDTH", "256")))
    max_layers = max(1, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_LAYERS", "6")))
    curriculum_max_rows = max(
        60,
        min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CURRICULUM_MAX_ROWS", "1200")), 20_000),
    )

    bpe_vocab_size = max(
        BYTE_VOCAB_SIZE,
        min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_BPE_VOCAB", "384")), 2048),
    )
    bpe_max_bytes = max(
        8_192,
        min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_BPE_MAX_BYTES", "256000")), 5_000_000),
    )

    pretrain_steps = max(
        0,
        min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_PRETRAIN_STEPS", "0")), 100),
    )
    corpus_root_raw = os.environ.get("AIRI_GENERALIST_RESEARCH_CORPUS_ROOT", "").strip()
    corpus_documents: list[CorpusDocument] = []
    corpus_manifest: dict[str, Any] = {
        "enabled": False,
        "files": 0,
        "documents": 0,
        "digest": None,
    }
    if corpus_root_raw:
        corpus_documents, built_manifest = repository_corpus(
            corpus_root_raw,
            max_files=max(1, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CORPUS_MAX_FILES", "192")), 2000)),
            max_bytes=max(32_768, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CORPUS_MAX_BYTES", "1500000")), 50_000_000)),
            max_file_bytes=max(4_096, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CORPUS_MAX_FILE_BYTES", "192000")), 5_000_000)),
            chunk_chars=max(256, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CORPUS_CHUNK_CHARS", "4000")), 32_000)),
            max_documents=max(8, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CORPUS_MAX_DOCUMENTS", "800")), 10_000)),
        )
        corpus_manifest = {
            "enabled": bool(corpus_documents and pretrain_steps > 0),
            **built_manifest.to_dict(),
        }

    previous = {}
    try:
        previous = json.loads((root / "status.json").read_text(encoding="utf-8"))
    except Exception:
        previous = {}
    cycle = int(previous.get("cycle", 0) or 0) + 1

    loaded = _load_champion(root, device=device)
    if loaded is None:
        champion_genome = research_seed()
        champion_runtime, champion_report = _train_genome(
            champion_genome,
            steps=max(steps, bootstrap_steps),
            seed=_genome_training_seed(champion_genome, namespace="bootstrap"),
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            precision=precision,
            pretrain_documents=corpus_documents,
            pretrain_steps=pretrain_steps,
        )
        _save_champion(root, champion_genome, champion_runtime, champion_report)
        bootstrapped = True
    else:
        champion_genome, champion_runtime = loaded
        champion_report = _grouped_validation(
            champion_runtime.model, champion_runtime.tokenizer, validation_rows(), device=device
        )
        champion_report["parameters"] = parameter_count(champion_runtime.model)
        champion_report["score"] = _research_score(champion_report, champion_report["parameters"])
        bootstrapped = False

    rotating_canary = canary_rows(cycle)
    champion_report["canary_cycle"] = cycle
    champion_report["canary"] = _grouped_validation(
        champion_runtime.model,
        champion_runtime.tokenizer,
        rotating_canary,
        device=device,
    )

    signals = _weaknesses(champion_report)
    mathesis = None
    mathesis_dir = os.environ.get("MATHESIS_STATE_DIR")
    if mathesis_dir and Path(mathesis_dir).exists():
        mathesis = mathesis_signals(mathesis_dir)
        signals.extend(mathesis.get("signals") or [])
    if (
        champion_genome.tokenizer_version == "byte-v1"
        and float(champion_report.get("generation_exact_accuracy", 0.0)) < 1.0
    ):
        signals.append("tokenizer_efficiency_gap")
    signals = list(dict.fromkeys(signals))

    curriculum_memory = CurriculumMemory(root, max_rows=curriculum_max_rows)
    curriculum_report = curriculum_memory.expand(cycle, signals=signals)
    replay_rows = curriculum_memory.rows()

    challengers = generate_challengers(
        champion_genome,
        signals=signals,
        count=challenger_count,
        exploration_offset=max(0, cycle - 1),
    )
    trials: list[dict[str, Any]] = []
    winner: tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]] | None = None

    continual_genome, continual_runtime, continual_report = _continue_champion(
        champion_genome,
        champion_runtime,
        replay_rows,
        steps=steps,
        seed=_genome_training_seed(champion_genome, namespace=f"continual-{cycle}"),
        device=device,
        cycle=cycle,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
        pretrain_documents=corpus_documents,
        pretrain_steps=pretrain_steps,
    )
    continual_report["canary_cycle"] = cycle
    continual_report["canary"] = _grouped_validation(
        continual_runtime.model,
        continual_runtime.tokenizer,
        rotating_canary,
        device=device,
    )
    continual_eligible, continual_reason = _research_eligible(
        champion_report,
        continual_report,
        minimum_loss_gain=minimum_loss_gain,
        max_domain_regression=max_domain_regression,
    )
    trials.append({
        "kind": "continual_learning",
        "genome": continual_genome.to_dict(),
        "report": continual_report,
        "eligible": continual_eligible,
        "reason": continual_reason,
    })
    if continual_eligible:
        winner = (continual_genome, continual_runtime, continual_report)

    for index, genome in enumerate(challengers):
        candidate_tokenizer = _tokenizer_for_genome(
            genome,
            source_runtime=champion_runtime,
            replay_rows=replay_rows,
            pretrain_documents=corpus_documents,
            bpe_vocab_size=bpe_vocab_size,
            bpe_max_bytes=bpe_max_bytes,
        )
        budget_reason = _research_budget_reason(
            genome,
            max_params=max_params,
            max_context=max_context,
            max_width=max_width,
            max_layers=max_layers,
            vocab_size=candidate_tokenizer.vocab_size,
        )
        if budget_reason:
            trials.append({
                "kind": "architecture",
                "genome": genome.to_dict(),
                "report": None,
                "eligible": False,
                "reason": f"research_resource_budget:{budget_reason}",
            })
            continue
        runtime, report = _train_genome(
            genome,
            steps=steps,
            seed=_genome_training_seed(genome),
            device=device,
            replay_rows=replay_rows,
            source_model=champion_runtime.model,
            source_tokenizer=champion_runtime.tokenizer,
            tokenizer=candidate_tokenizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            precision=precision,
            pretrain_documents=corpus_documents,
            pretrain_steps=pretrain_steps,
        )
        report["canary_cycle"] = cycle
        report["canary"] = _grouped_validation(
            runtime.model,
            runtime.tokenizer,
            rotating_canary,
            device=device,
        )
        eligible, reason = _research_eligible(
            champion_report,
            report,
            minimum_loss_gain=minimum_loss_gain,
            max_domain_regression=max_domain_regression,
        )
        trials.append({
            "kind": "architecture",
            "genome": genome.to_dict(),
            "report": report,
            "eligible": eligible,
            "reason": reason,
        })
        if eligible and (
            winner is None
            or float(report.get("nll_per_byte", report["loss"]))
            < float(winner[2].get("nll_per_byte", winner[2]["loss"]))
        ):
            winner = (genome, runtime, report)

    promoted = winner is not None
    if winner is not None:
        champion_genome, champion_runtime, champion_report = winner
        _save_champion(root, champion_genome, champion_runtime, champion_report)

    result = {
        "ok": True,
        "cycle": cycle,
        "bootstrapped": bootstrapped,
        "promoted": promoted,
        "champion": champion_genome.to_dict(),
        "champion_report": champion_report,
        "signals": signals,
        "mathesis": mathesis,
        "curriculum_memory": curriculum_report,
        "grounded_pretraining": {
            "steps": pretrain_steps,
            "corpus": corpus_manifest,
        },
        "rotating_canary": {
            "cycle": cycle,
            "domains": [row.domain for row in rotating_canary],
            "training_overlap": sorted(
                {row.messages[0]["content"] for row in rotating_canary}
                & {row.messages[0]["content"] for row in replay_rows}
            ),
        },
        "trials": trials,
        "policy": {
            "research_only": True,
            "production_qualification_separate": True,
            "minimum_loss_gain": minimum_loss_gain,
            "max_domain_regression": max_domain_regression,
            "research_resource_budget": {
                "max_params": max_params,
                "max_context": max_context,
                "max_width": max_width,
                "max_layers": max_layers,
            },
            "continual_learning": {
                "enabled": True,
                "full_replay": True,
                "domain_balanced_base_replay": True,
                "curriculum_max_rows": curriculum_max_rows,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "precision": precision,
            },
            "grounded_pretraining": {
                "enabled": bool(corpus_documents and pretrain_steps > 0),
                "steps_per_candidate": pretrain_steps,
                "corpus_digest": corpus_manifest.get("digest"),
                "protected_exam_sources_excluded": True,
            },
            "rotating_canary": {
                "enabled": True,
                "training_excluded": True,
                "domains": 6,
            },
            "architecture_weight_transfer": {
                "enabled": True,
                "exact_name_and_shape_only": True,
                "byte_to_bpe_embedding_migration": True,
            },
            "tokenizer_research": {
                "allowed": ["byte-v1", "bpe-v1"],
                "comparison_metric": "nll_per_byte",
                "bpe_vocab_size": bpe_vocab_size,
                "bpe_max_training_bytes": bpe_max_bytes,
                "protected_eval_rows_excluded_from_tokenizer_training": True,
            },
        },
        "updated_at": time.time(),
    }
    _atomic_json(root / "status.json", result)
    with (root / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> int:
    result = run_research_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
