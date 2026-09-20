from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .native_foundation import (
    NativeFoundationConfig,
    native_checkpoint_status,
    native_parameter_count,
)
from .native_training import NativeTrainConfig, native_training_status


NATIVE_EVOLUTION_VERSION = "native-evolution-v1"


def _digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _rounded_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


@dataclass
class NativeEvolutionGenome:
    generation: int
    genome_id: str
    parent_id: str | None
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    rope_theta: float
    rms_eps: float
    init_std: float
    tokenizer_version: str
    tie_embeddings: bool
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip: float
    domain_weights: dict[str, float] = field(default_factory=dict)

    def validate(self) -> "NativeEvolutionGenome":
        cfg = self.foundation_config().validate()
        train = self.training_config(max_steps=max(2, self.warmup_steps + 1)).validate()
        self.generation = max(0, int(self.generation))
        self.vocab_size = cfg.vocab_size
        self.context_length = cfg.context_length
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_layers = cfg.n_layers
        self.d_ff = cfg.d_ff
        self.dropout = cfg.dropout
        self.rope_theta = cfg.rope_theta
        self.rms_eps = cfg.rms_eps
        self.init_std = cfg.init_std
        self.tokenizer_version = cfg.tokenizer_version
        self.tie_embeddings = cfg.tie_embeddings
        self.learning_rate = train.learning_rate
        self.min_learning_rate = train.min_learning_rate
        self.warmup_steps = int(self.warmup_steps)
        self.weight_decay = train.weight_decay
        self.adam_beta1 = train.adam_beta1
        self.adam_beta2 = train.adam_beta2
        self.adam_eps = train.adam_eps
        self.grad_clip = train.grad_clip
        clean: dict[str, float] = {}
        for key, value in dict(self.domain_weights or {}).items():
            key = str(key).strip().lower()
            value = float(value)
            if key and 0.01 <= value <= 100.0:
                clean[key] = value
        self.domain_weights = clean
        if not str(self.genome_id).strip():
            raise ValueError("native evolution genome_id is required")
        return self

    def foundation_config(self) -> NativeFoundationConfig:
        return NativeFoundationConfig(
            vocab_size=int(self.vocab_size),
            context_length=int(self.context_length),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_kv_heads=int(self.n_kv_heads),
            n_layers=int(self.n_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
            rope_theta=float(self.rope_theta),
            rms_eps=float(self.rms_eps),
            init_std=float(self.init_std),
            tokenizer_version=str(self.tokenizer_version),
            tie_embeddings=bool(self.tie_embeddings),
        )

    def training_config(
        self,
        *,
        max_steps: int,
        seed: int = 17,
        device: str = "cpu",
        precision: str = "fp32",
        micro_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_eval_blocks: int = 32,
        validation_fraction: float = 0.2,
        save_optimizer_state: bool = True,
    ) -> NativeTrainConfig:
        max_steps = max(1, int(max_steps))
        warmup = min(max_steps - 1, max(0, int(self.warmup_steps)))
        return NativeTrainConfig(
            max_steps=max_steps,
            micro_batch_size=max(1, int(micro_batch_size)),
            gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
            learning_rate=float(self.learning_rate),
            min_learning_rate=min(
                float(self.learning_rate),
                max(0.0, float(self.min_learning_rate)),
            ),
            warmup_steps=warmup,
            weight_decay=float(self.weight_decay),
            adam_beta1=float(self.adam_beta1),
            adam_beta2=float(self.adam_beta2),
            adam_eps=float(self.adam_eps),
            grad_clip=float(self.grad_clip),
            validation_fraction=float(validation_fraction),
            max_eval_blocks=max(1, int(max_eval_blocks)),
            seed=int(seed),
            device=str(device),
            precision=str(precision),
            save_optimizer_state=bool(save_optimizer_state),
            domain_weights=dict(self.domain_weights),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def parameters(self) -> int:
        return native_parameter_count(self.foundation_config())


@dataclass(frozen=True)
class NativeMutation:
    name: str
    kind: str
    changes: dict[str, Any]
    requires_reinit: bool = False
    requires_retokenization: bool = False
    evidence_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "changes": dict(self.changes),
            "requires_reinit": self.requires_reinit,
            "requires_retokenization": self.requires_retokenization,
            "evidence_tags": list(self.evidence_tags),
        }


def genome_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    generation: int = 0,
    genome_id: str | None = None,
    parent_id: str | None = None,
) -> NativeEvolutionGenome:
    checkpoint = native_checkpoint_status(checkpoint_dir)
    if not checkpoint.get("ok"):
        raise ValueError(str(checkpoint.get("reason") or "invalid Native checkpoint"))
    status = native_training_status(checkpoint_dir)
    training = status.get("training") if status.get("ok") else None
    training_config = dict((training or {}).get("config") or {})

    cfg = NativeFoundationConfig.from_dict(checkpoint["config"])
    defaults = NativeTrainConfig(max_steps=1000).validate()
    seed_payload = {
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "generation": int(generation),
    }
    identifier = genome_id or f"native-{int(generation)}-{_digest(seed_payload)[:12]}"
    return NativeEvolutionGenome(
        generation=int(generation),
        genome_id=identifier,
        parent_id=parent_id,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        rope_theta=cfg.rope_theta,
        rms_eps=cfg.rms_eps,
        init_std=cfg.init_std,
        tokenizer_version=cfg.tokenizer_version,
        tie_embeddings=cfg.tie_embeddings,
        learning_rate=float(training_config.get("learning_rate", defaults.learning_rate)),
        min_learning_rate=float(training_config.get("min_learning_rate", defaults.min_learning_rate)),
        warmup_steps=int(training_config.get("warmup_steps", defaults.warmup_steps)),
        weight_decay=float(training_config.get("weight_decay", defaults.weight_decay)),
        adam_beta1=float(training_config.get("adam_beta1", defaults.adam_beta1)),
        adam_beta2=float(training_config.get("adam_beta2", defaults.adam_beta2)),
        adam_eps=float(training_config.get("adam_eps", defaults.adam_eps)),
        grad_clip=float(training_config.get("grad_clip", defaults.grad_clip)),
        domain_weights=dict(training_config.get("domain_weights") or {}),
    ).validate()


def _mutation_id(parent: NativeEvolutionGenome, mutation: NativeMutation) -> str:
    payload = {
        "parent": parent.genome_id,
        "generation": parent.generation + 1,
        "mutation": mutation.to_dict(),
    }
    return f"native-{parent.generation + 1}-{_digest(payload)[:12]}"


def apply_mutation(
    champion: NativeEvolutionGenome,
    mutation: NativeMutation,
) -> NativeEvolutionGenome:
    payload = champion.to_dict()
    for key, value in mutation.changes.items():
        if key == "domain_weights":
            merged = dict(payload.get("domain_weights") or {})
            merged.update({str(k): float(v) for k, v in dict(value).items()})
            payload["domain_weights"] = merged
        else:
            if key not in payload:
                raise ValueError(f"mutation targets unauthorized genome field: {key}")
            payload[key] = value
    payload["generation"] = champion.generation + 1
    payload["parent_id"] = champion.genome_id
    candidate = NativeEvolutionGenome(**payload)
    candidate.genome_id = _mutation_id(champion, mutation)
    return candidate.validate()


def _research_tags(research: dict[str, Any] | None) -> set[str]:
    if not isinstance(research, dict):
        return set()
    counts = research.get("tag_counts")
    if not isinstance(counts, dict):
        return set()
    return {
        str(key)
        for key, value in counts.items()
        if int(value or 0) > 0
    }


def _weak_domain_mutations(signals: Iterable[str]) -> list[NativeMutation]:
    mappings = {
        "coding_gap": "code",
        "data_gap": "data-analysis",
        "tool_gap": "tool-use",
        "reasoning_gap": "reasoning",
        "symbolic_reasoning_signal": "math",
        "deep_symbolic_signal": "math",
        "research_curriculum_signal": "reasoning",
    }
    out: list[NativeMutation] = []
    for signal in signals:
        domain = mappings.get(str(signal))
        if domain:
            out.append(NativeMutation(
                name=f"curriculum-{domain}",
                kind="curriculum",
                changes={"domain_weights": {domain: 1.35}},
                evidence_tags=("curriculum",),
            ))
    return out


def mutation_library(
    champion: NativeEvolutionGenome,
    *,
    research: dict[str, Any] | None = None,
    mathesis_signals: Iterable[str] | None = None,
) -> list[NativeMutation]:
    tags = _research_tags(research)
    signals = tuple(dict.fromkeys(str(row) for row in (mathesis_signals or ())))

    mutations: list[NativeMutation] = []
    mutations.extend(_weak_domain_mutations(signals))

    if "optimizer" in tags or not tags:
        mutations.extend([
            NativeMutation(
                "optimizer-lr-down",
                "optimizer",
                {
                    "learning_rate": max(1e-6, champion.learning_rate * 0.75),
                    "min_learning_rate": max(0.0, champion.min_learning_rate * 0.75),
                },
                evidence_tags=("optimizer",),
            ),
            NativeMutation(
                "optimizer-lr-up",
                "optimizer",
                {
                    "learning_rate": min(1e-2, champion.learning_rate * 1.20),
                    "min_learning_rate": min(1e-2, champion.min_learning_rate * 1.10),
                },
                evidence_tags=("optimizer",),
            ),
            NativeMutation(
                "optimizer-beta2-lower",
                "optimizer",
                {"adam_beta2": max(0.90, champion.adam_beta2 - 0.02)},
                evidence_tags=("optimizer",),
            ),
            NativeMutation(
                "optimizer-weight-decay-lower",
                "optimizer",
                {"weight_decay": max(0.0, champion.weight_decay * 0.5)},
                evidence_tags=("optimizer",),
            ),
        ])

    if "gqa" in tags or "efficiency" in tags:
        if champion.n_kv_heads > 1:
            divisors = [
                value for value in range(1, champion.n_kv_heads)
                if champion.n_heads % value == 0
            ]
            if divisors:
                mutations.append(NativeMutation(
                    "architecture-more-gqa",
                    "architecture",
                    {"n_kv_heads": max(divisors)},
                    requires_reinit=True,
                    evidence_tags=("gqa", "efficiency"),
                ))
        elif champion.n_heads > 1:
            candidates = [
                value for value in range(2, champion.n_heads + 1)
                if champion.n_heads % value == 0
            ]
            if candidates:
                mutations.append(NativeMutation(
                    "architecture-less-gqa",
                    "architecture",
                    {"n_kv_heads": min(candidates)},
                    requires_reinit=True,
                    evidence_tags=("gqa",),
                ))

    if "efficiency" in tags:
        mutations.append(NativeMutation(
            "architecture-ff-compact",
            "architecture",
            {"d_ff": max(champion.d_model, _rounded_multiple(champion.d_ff * 0.875))},
            requires_reinit=True,
            evidence_tags=("efficiency",),
        ))

    if "long-context" in tags:
        mutations.extend([
            NativeMutation(
                "architecture-context-grow",
                "architecture",
                {"context_length": min(131072, champion.context_length * 2)},
                requires_reinit=True,
                evidence_tags=("long-context",),
            ),
            NativeMutation(
                "architecture-rope-theta-grow",
                "architecture",
                {"rope_theta": min(10_000_000.0, champion.rope_theta * 2.0)},
                requires_reinit=True,
                evidence_tags=("long-context",),
            ),
        ])

    if "tokenizer" in tags or "tokenizer_efficiency_gap" in signals:
        if champion.tokenizer_version == "byte-v1":
            target_vocab = 512
            mutations.append(NativeMutation(
                "tokenizer-byte-to-bpe",
                "tokenizer",
                {
                    "tokenizer_version": "bpe-v1",
                    "vocab_size": target_vocab,
                },
                requires_reinit=True,
                requires_retokenization=True,
                evidence_tags=("tokenizer",),
            ))
        else:
            target_vocab = min(65536, max(320, _rounded_multiple(champion.vocab_size * 1.25, 64)))
            if target_vocab != champion.vocab_size:
                mutations.append(NativeMutation(
                    "tokenizer-bpe-grow",
                    "tokenizer",
                    {"vocab_size": target_vocab},
                    requires_reinit=True,
                    requires_retokenization=True,
                    evidence_tags=("tokenizer",),
                ))

    if "curriculum" in tags:
        for domain in ("reasoning", "code", "math", "data-analysis", "tool-use"):
            mutations.append(NativeMutation(
                f"curriculum-{domain}-research",
                "curriculum",
                {"domain_weights": {domain: 1.20}},
                evidence_tags=("curriculum",),
            ))

    # Safe local exploration remains available when online evidence is sparse.
    mutations.extend([
        NativeMutation(
            "training-warmup-shorter",
            "training",
            {"warmup_steps": max(0, int(champion.warmup_steps * 0.75))},
        ),
        NativeMutation(
            "architecture-ff-grow",
            "architecture",
            {"d_ff": min(65536, _rounded_multiple(champion.d_ff * 1.125))},
            requires_reinit=True,
        ),
    ])

    unique: list[NativeMutation] = []
    seen: set[str] = set()
    for mutation in mutations:
        signature = _digest(mutation.to_dict())
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mutation)
    return unique


def generate_native_challengers(
    champion: NativeEvolutionGenome,
    *,
    research: dict[str, Any] | None = None,
    mathesis_signals: Iterable[str] | None = None,
    count: int = 4,
    exploration_offset: int = 0,
    max_parameters: int | None = None,
    max_parameter_ratio: float = 1.5,
) -> list[tuple[NativeMutation, NativeEvolutionGenome]]:
    champion.validate()
    variants = mutation_library(
        champion,
        research=research,
        mathesis_signals=mathesis_signals,
    )
    if variants:
        shift = max(0, int(exploration_offset)) % len(variants)
        variants = variants[shift:] + variants[:shift]

    absolute_limit = (
        int(max_parameters)
        if max_parameters is not None
        else max(champion.parameters, int(math.ceil(champion.parameters * float(max_parameter_ratio))))
    )
    ratio_limit = max(
        champion.parameters,
        int(math.ceil(champion.parameters * max(1.0, float(max_parameter_ratio)))),
    )
    parameter_limit = min(absolute_limit, ratio_limit)

    out: list[tuple[NativeMutation, NativeEvolutionGenome]] = []
    seen_genomes: set[str] = set()
    for mutation in variants:
        try:
            candidate = apply_mutation(champion, mutation)
        except Exception:
            continue
        if candidate.parameters > parameter_limit:
            continue
        signature = _digest({
            key: value
            for key, value in candidate.to_dict().items()
            if key not in {"generation", "genome_id", "parent_id"}
        })
        if signature in seen_genomes:
            continue
        seen_genomes.add(signature)
        out.append((mutation, candidate))
        if len(out) >= max(1, int(count)):
            break
    return out


def current_training_step(checkpoint_dir: str | Path) -> int:
    status = native_training_status(checkpoint_dir)
    training = status.get("training") if status.get("ok") else None
    if not isinstance(training, dict):
        return 0
    return max(0, int(training.get("global_step", 0) or 0))
