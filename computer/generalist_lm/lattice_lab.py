from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Any, Sequence

from .native_data import (
    NativeTokenBlock,
    load_native_corpus,
    pack_native_blocks,
    split_native_corpus,
)
from .native_foundation import (
    NativeFoundationConfig,
    NativeFoundationLM,
    native_parameter_count,
)
from .native_lattice import (
    AiriLatticeConfig,
    AiriLatticeLM,
    lattice_active_parameter_estimate,
    lattice_parameter_count,
    lattice_state_bytes,
)
from .tokenizer import ByteTokenizer, PAD


LATTICE_LAB_VERSION = "airi-lattice-lab-v0"


@dataclass
class LatticeLabConfig:
    steps: int = 8
    batch_size: int = 1
    learning_rate: float = 2e-3
    min_learning_rate: float = 2e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    validation_fraction: float = 0.25
    max_eval_blocks: int = 12
    seed: int = 1701
    device: str = "cpu"
    minimum_loss_gain: float = 0.002
    max_domain_regression: float = 0.05
    max_active_parameter_ratio: float = 1.15

    def validate(self) -> "LatticeLabConfig":
        self.steps = max(1, int(self.steps))
        self.batch_size = max(1, int(self.batch_size))
        self.learning_rate = float(self.learning_rate)
        self.min_learning_rate = float(self.min_learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip = float(self.grad_clip)
        self.validation_fraction = float(self.validation_fraction)
        self.max_eval_blocks = max(1, int(self.max_eval_blocks))
        self.seed = int(self.seed)
        self.device = str(self.device)
        self.minimum_loss_gain = max(0.0, float(self.minimum_loss_gain))
        self.max_domain_regression = max(0.0, float(self.max_domain_regression))
        self.max_active_parameter_ratio = max(
            0.25,
            float(self.max_active_parameter_ratio),
        )
        if not (0.0 < self.learning_rate <= 0.1):
            raise ValueError("invalid Lattice Lab learning_rate")
        if not (0.0 <= self.min_learning_rate <= self.learning_rate):
            raise ValueError("invalid Lattice Lab min_learning_rate")
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError("invalid Lattice Lab weight_decay")
        if not (0.0 < self.grad_clip <= 100.0):
            raise ValueError("invalid Lattice Lab grad_clip")
        if not (0.0 < self.validation_fraction < 0.5):
            raise ValueError("invalid Lattice Lab validation_fraction")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("unsupported Lattice Lab device")
        return self


def _cosine_lr(
    step: int,
    total_steps: int,
    high: float,
    low: float,
) -> float:
    if total_steps <= 1:
        return float(low)
    ratio = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return float(low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * ratio)))


def _select_blocks(
    blocks: Sequence[NativeTokenBlock],
    *,
    batch_size: int,
    step: int,
    order: Sequence[int],
) -> list[NativeTokenBlock]:
    if not blocks:
        raise ValueError("Lattice Lab received no training blocks")
    start = (step * batch_size) % len(order)
    indices = [
        order[(start + offset) % len(order)]
        for offset in range(batch_size)
    ]
    return [blocks[index] for index in indices]


def _tensor_batch(torch, rows, *, device):
    ids = torch.tensor(
        [row.ids for row in rows],
        dtype=torch.long,
        device=device,
    )
    labels = ids.clone()
    labels[labels == PAD] = -100
    return ids, labels


def _balanced_eval_blocks(
    blocks: Sequence[NativeTokenBlock],
    *,
    max_blocks: int,
) -> list[NativeTokenBlock]:
    """Round-robin validation domains before taking extra blocks.

    The first real Lattice swarms revealed that prefix slicing could evaluate
    only one domain when one document produced many early blocks. Architecture
    promotion must see as many held-out domains as the budget permits.
    """
    limit = max(1, int(max_blocks))
    grouped: dict[str, list[NativeTokenBlock]] = {}
    domain_order: list[str] = []
    for row in blocks:
        if row.domain not in grouped:
            grouped[row.domain] = []
            domain_order.append(row.domain)
        grouped[row.domain].append(row)

    selected: list[NativeTokenBlock] = []
    cursor = 0
    while len(selected) < limit:
        added = False
        for domain in domain_order:
            rows = grouped[domain]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _evaluate(
    model,
    blocks: Sequence[NativeTokenBlock],
    *,
    torch,
    device,
    batch_size: int,
    max_blocks: int,
) -> dict[str, Any]:
    selected = _balanced_eval_blocks(
        blocks,
        max_blocks=max_blocks,
    )
    if not selected:
        raise ValueError("Lattice Lab received no validation blocks")

    model.eval()
    total = 0.0
    rows_seen = 0
    stats_accum: dict[str, float] = {}
    stats_count = 0

    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            rows = selected[start:start + batch_size]
            ids, labels = _tensor_batch(torch, rows, device=device)
            output = model(ids, labels=labels)
            loss = output["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite Lattice Lab validation loss")
            value = float(loss.detach().float().cpu())
            total += value * len(rows)
            rows_seen += len(rows)
            stats = output.get("stats")
            if isinstance(stats, dict):
                for key in ("mean_surprise", "mean_reasoning_steps"):
                    if key in stats:
                        stats_accum[key] = stats_accum.get(key, 0.0) + float(stats[key])
                stats_count += 1

        # Recompute domain losses on domain-pure mini-batches. This prevents a
        # mixed batch mean from hiding a regression in one protected domain.
        domain_loss: dict[str, float] = {}
        for domain in sorted({row.domain for row in selected}):
            rows_for_domain = [row for row in selected if row.domain == domain]
            domain_total = 0.0
            domain_seen = 0
            for start in range(0, len(rows_for_domain), batch_size):
                rows = rows_for_domain[start:start + batch_size]
                ids, labels = _tensor_batch(torch, rows, device=device)
                output = model(ids, labels=labels)
                loss = output["loss"]
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite Lattice Lab domain loss: {domain}"
                    )
                value = float(loss.detach().float().cpu())
                domain_total += value * len(rows)
                domain_seen += len(rows)
            domain_loss[domain] = domain_total / max(1, domain_seen)

    result = {
        "loss": total / max(1, rows_seen),
        "blocks": rows_seen,
        "domain_loss": domain_loss,
    }
    if stats_count:
        result["model_stats"] = {
            key: value / stats_count
            for key, value in sorted(stats_accum.items())
        }
    return result


def _train_fixed_budget(
    model,
    train_blocks: Sequence[NativeTokenBlock],
    validation_blocks: Sequence[NativeTokenBlock],
    *,
    lab: LatticeLabConfig,
    order: Sequence[int],
) -> dict[str, Any]:
    import torch

    torch.manual_seed(lab.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(lab.seed)
    device = torch.device(lab.device)
    model.to(device)

    initial = _evaluate(
        model,
        validation_blocks,
        torch=torch,
        device=device,
        batch_size=lab.batch_size,
        max_blocks=lab.max_eval_blocks,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lab.learning_rate,
        weight_decay=lab.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    model.train()
    best_train_loss = float("inf")
    tokens_processed = 0
    started = time.perf_counter()
    for step in range(lab.steps):
        rows = _select_blocks(
            train_blocks,
            batch_size=lab.batch_size,
            step=step,
            order=order,
        )
        ids, labels = _tensor_batch(torch, rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(ids, labels=labels)
        loss = output["loss"]
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite Lattice Lab training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), lab.grad_clip)
        lr = _cosine_lr(
            step,
            lab.steps,
            lab.learning_rate,
            lab.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        best_train_loss = min(
            best_train_loss,
            float(loss.detach().float().cpu()),
        )
        tokens_processed += int(ids.numel())

    elapsed = max(1e-9, time.perf_counter() - started)
    final = _evaluate(
        model,
        validation_blocks,
        torch=torch,
        device=device,
        batch_size=lab.batch_size,
        max_blocks=lab.max_eval_blocks,
    )
    return {
        "initial": initial,
        "final": final,
        "best_train_loss": best_train_loss,
        "steps": lab.steps,
        "tokens_processed": tokens_processed,
        "train_seconds": elapsed,
        "tokens_per_second": tokens_processed / elapsed,
    }


def _closest_transformer_baseline(
    lattice: AiriLatticeConfig,
) -> NativeFoundationConfig:
    """Choose a scratch Transformer near Lattice active parameter count.

    The search is deliberately small/deterministic. Active Lattice parameters
    are the relevant compute proxy because inactive experts are not executed.
    """
    target = lattice_active_parameter_estimate(lattice)
    head_options = [
        value for value in (1, 2, 4, 8, 16)
        if value <= lattice.d_model and lattice.d_model % value == 0
        and (lattice.d_model // value) % 2 == 0
    ]
    if not head_options:
        head_options = [1]
    best = None
    for layers in range(1, min(8, lattice.n_cells * 3 + 2) + 1):
        for heads in head_options:
            kv_options = [
                value for value in (1, 2, 4, 8)
                if value <= heads and heads % value == 0
            ]
            for kv_heads in kv_options:
                for multiplier in (1.0, 1.5, 2.0, 3.0, 4.0):
                    d_ff = max(
                        lattice.d_model,
                        int(round(lattice.d_model * multiplier / 16.0)) * 16,
                    )
                    try:
                        cfg = NativeFoundationConfig(
                            vocab_size=lattice.vocab_size,
                            context_length=lattice.context_length,
                            d_model=lattice.d_model,
                            n_heads=heads,
                            n_kv_heads=kv_heads,
                            n_layers=layers,
                            d_ff=d_ff,
                            dropout=lattice.dropout,
                            tokenizer_version=lattice.tokenizer_version,
                        ).validate()
                    except Exception:
                        continue
                    params = native_parameter_count(cfg)
                    distance = abs(params - target)
                    candidate = (distance, params, cfg)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
    if best is None:
        raise RuntimeError("unable to construct matched Transformer baseline")
    return best[2]


def _transformer_kv_bytes(
    config: NativeFoundationConfig,
    *,
    batch_size: int = 1,
    bytes_per_element: int = 4,
) -> int:
    head_dim = config.d_model // config.n_heads
    return (
        2
        * config.n_layers
        * config.n_kv_heads
        * head_dim
        * config.context_length
        * int(batch_size)
        * int(bytes_per_element)
    )


def lattice_promotion_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    lab: LatticeLabConfig,
) -> tuple[bool, str]:
    old_loss = float(baseline["training"]["final"]["loss"])
    new_loss = float(candidate["training"]["final"]["loss"])
    if old_loss - new_loss < lab.minimum_loss_gain:
        return False, "Lattice did not beat the matched Transformer loss margin"

    baseline_domains = baseline["training"]["final"]["domain_loss"]
    candidate_domains = candidate["training"]["final"]["domain_loss"]
    for domain, old_value in baseline_domains.items():
        if domain not in candidate_domains:
            return False, f"Lattice lost held-out domain: {domain}"
        if float(candidate_domains[domain]) > (
            float(old_value) + lab.max_domain_regression
        ):
            return False, f"Lattice regressed in held-out domain: {domain}"

    active_ratio = (
        float(candidate["active_parameters"])
        / max(1.0, float(baseline["parameters"]))
    )
    if active_ratio > lab.max_active_parameter_ratio:
        return False, "Lattice exceeds active-parameter compute budget"

    return True, "Lattice beat matched Transformer under the research gate"


def benchmark_lattice_against_transformer(
    corpus_manifest: str,
    *,
    allowed_roots: Sequence[str],
    lattice_config: AiriLatticeConfig | None = None,
    lab_config: LatticeLabConfig | None = None,
) -> dict[str, Any]:
    """Scratch-train Lattice and a parameter-matched Transformer fairly.

    This is deliberately a small architecture-research benchmark. It does not
    claim general intelligence from a tiny corpus; it answers the narrower and
    useful question: which architecture extracts more held-out language-model
    signal from the same tokens and optimizer budget?
    """
    import torch

    lattice = (lattice_config or AiriLatticeConfig()).validate()
    lab = (lab_config or LatticeLabConfig()).validate()
    if lattice.tokenizer_version != "byte-v1":
        raise ValueError(
            "Lattice Lab v0 requires byte-v1 so both families see identical tokens"
        )

    report = load_native_corpus(
        corpus_manifest,
        allowed_roots=allowed_roots,
    )
    train_docs, validation_docs = split_native_corpus(
        report.documents,
        validation_fraction=lab.validation_fraction,
        seed=lab.seed,
    )
    tokenizer = ByteTokenizer()
    train_blocks = pack_native_blocks(
        train_docs,
        tokenizer,
        context_length=lattice.context_length,
    )
    validation_blocks = pack_native_blocks(
        validation_docs,
        tokenizer,
        context_length=lattice.context_length,
    )
    if not train_blocks or not validation_blocks:
        raise ValueError("Lattice Lab corpus produced no token blocks")

    order = list(range(len(train_blocks)))
    random.Random(lab.seed).shuffle(order)

    # Re-seed immediately before each construction so initialization is stable
    # across repeated laboratory runs.
    torch.manual_seed(lab.seed)
    lattice_model = AiriLatticeLM(lattice)
    lattice_total = lattice_parameter_count(lattice)
    lattice_active = lattice_active_parameter_estimate(lattice)
    lattice_training = _train_fixed_budget(
        lattice_model,
        train_blocks,
        validation_blocks,
        lab=lab,
        order=order,
    )

    transformer_cfg = _closest_transformer_baseline(lattice)
    torch.manual_seed(lab.seed)
    transformer_model = NativeFoundationLM(transformer_cfg)
    transformer_params = native_parameter_count(transformer_cfg)
    transformer_training = _train_fixed_budget(
        transformer_model,
        train_blocks,
        validation_blocks,
        lab=lab,
        order=order,
    )

    baseline = {
        "family": "airi-native-foundation",
        "config": transformer_cfg.to_dict(),
        "parameters": transformer_params,
        "active_parameters": transformer_params,
        "state_bytes_at_context": _transformer_kv_bytes(
            transformer_cfg,
            batch_size=lab.batch_size,
        ),
        "training": transformer_training,
    }
    candidate = {
        "family": "airi-native-lattice",
        "config": lattice.to_dict(),
        "parameters": lattice_total,
        "active_parameters": lattice_active,
        "state_bytes_at_context": lattice_state_bytes(
            lattice,
            batch_size=lab.batch_size,
        ),
        "training": lattice_training,
    }
    promoted, reason = lattice_promotion_gate(
        baseline,
        candidate,
        lab=lab,
    )

    return {
        "ok": True,
        "version": LATTICE_LAB_VERSION,
        "corpus_digest": report.corpus_digest,
        "train_documents": len(train_docs),
        "validation_documents": len(validation_docs),
        "train_blocks": len(train_blocks),
        "validation_blocks": len(validation_blocks),
        "lab": asdict(lab),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_wins": promoted,
        "decision": reason,
        "research_only": True,
        "external_pretrained": False,
    }
