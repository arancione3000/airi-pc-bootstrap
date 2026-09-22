from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .evolution import GeneralistGenome
from .model import GeneralistLMConfig, estimate_parameter_count


ARCHITECTURE_IR_VERSION = "airi-architecture-ir-v1"
SUPPORTED_FAMILIES = {"decoder_transformer_v1", "gated_recurrent_v1"}


@dataclass(frozen=True)
class ArchitectureSpec:
    """Immutable, bounded description of an AIRI neural architecture.

    The IR deliberately contains data only.  It cannot embed Python, shell
    commands, imports, URLs or executable callbacks.  New neural primitives
    must first be implemented and verifier-tested in the model factory before
    this schema can reference them.
    """

    architecture_id: str
    generation: int
    parent_id: str | None
    family: str = "decoder_transformer_v1"

    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.0

    tokenizer_version: str = "bpe-v1"
    target_vocab_size: int = 1024

    norm_type: str = "rmsnorm"
    norm_placement: str = "pre"
    position_encoding: str = "rope"
    ff_variant: str = "swiglu"

    attention_type: str = "mha"
    n_kv_heads: int | None = None
    local_attention_window: int = 0
    local_attention_every: int = 0
    tie_embeddings: bool = True

    learning_rate: float = 3e-4
    reasoning_depth: int = 2

    retrieval_adapter: bool = True
    symbolic_adapter: bool = True
    code_adapter: bool = True
    data_adapter: bool = True

    tool_protocol: str = "tool-json-v1"

    def validate(self) -> "ArchitectureSpec":
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError("unsupported architecture family")
        if not self.architecture_id or len(self.architecture_id) > 160:
            raise ValueError("invalid architecture_id")
        if self.parent_id is not None and len(self.parent_id) > 160:
            raise ValueError("invalid parent_id")
        if not (0 <= int(self.generation) <= 1_000_000):
            raise ValueError("invalid architecture generation")
        if not (264 <= int(self.target_vocab_size) <= 65_536):
            raise ValueError("target vocab outside bounded Architecture IR")

        # GeneralistGenome and GeneralistLMConfig remain the authoritative
        # implementation validators.  Reconstructing both here makes the IR
        # fail closed if their constraints diverge.
        genome = self.to_genome(validate=False)
        genome.validate()
        self.to_model_config(
            vocab_size=max(264, min(int(self.target_vocab_size), 65_536)),
            validate=False,
        ).validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_payload(self) -> dict[str, Any]:
        raw = self.to_dict()
        for key in ("architecture_id", "generation", "parent_id"):
            raw.pop(key, None)
        # Normalize semantically equivalent encodings before hashing.
        if raw.get("family") == "gated_recurrent_v1":
            raw["attention_type"] = "none"
            raw["n_kv_heads"] = 0
            raw["local_attention_window"] = 0
            raw["local_attention_every"] = 0
            raw["position_encoding"] = "none"
            raw["n_heads"] = 1
        else:
            # MHA has one KV head set per query head; GQA without an explicit
            # value means the bounded factory policy of half the query heads.
            if raw.get("attention_type") == "mha":
                raw["n_kv_heads"] = int(raw["n_heads"])
            elif raw.get("n_kv_heads") is None:
                raw["n_kv_heads"] = max(1, int(raw["n_heads"]) // 2)
            if int(raw.get("local_attention_window", 0) or 0) == 0:
                raw["local_attention_every"] = 0
        return raw

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def parameter_estimate(self, *, vocab_size: int | None = None) -> int:
        cfg = self.to_model_config(
            vocab_size=int(vocab_size or self.target_vocab_size),
        )
        return int(estimate_parameter_count(cfg))

    def to_model_config(
        self,
        *,
        vocab_size: int | None = None,
        validate: bool = True,
    ) -> GeneralistLMConfig:
        cfg = GeneralistLMConfig(
            architecture_family=str(self.family),
            vocab_size=int(vocab_size or self.target_vocab_size),
            context_length=int(self.context_length),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
            tokenizer_version=str(self.tokenizer_version),
            norm_type=str(self.norm_type),
            position_encoding=str(self.position_encoding),
            ff_variant=str(self.ff_variant),
            attention_type=str(self.attention_type),
            n_kv_heads=self.n_kv_heads,
            local_attention_window=int(self.local_attention_window),
            local_attention_every=int(self.local_attention_every),
            norm_placement=str(self.norm_placement),
            tie_embeddings=bool(self.tie_embeddings),
        )
        return cfg.validate() if validate else cfg

    def to_genome(self, *, validate: bool = True) -> GeneralistGenome:
        genome = GeneralistGenome(
            architecture_family=str(self.family),
            generation=int(self.generation),
            parent_id=self.parent_id,
            genome_id=str(self.architecture_id),
            context_length=int(self.context_length),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
            learning_rate=float(self.learning_rate),
            tokenizer_version=str(self.tokenizer_version),
            tool_protocol=str(self.tool_protocol),
            retrieval_adapter=bool(self.retrieval_adapter),
            symbolic_adapter=bool(self.symbolic_adapter),
            code_adapter=bool(self.code_adapter),
            data_adapter=bool(self.data_adapter),
            reasoning_depth=int(self.reasoning_depth),
            norm_type=str(self.norm_type),
            position_encoding=str(self.position_encoding),
            ff_variant=str(self.ff_variant),
            attention_type=str(self.attention_type),
            n_kv_heads=self.n_kv_heads,
            local_attention_window=int(self.local_attention_window),
            local_attention_every=int(self.local_attention_every),
            norm_placement=str(self.norm_placement),
            tie_embeddings=bool(self.tie_embeddings),
        )
        return genome.validate() if validate else genome

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArchitectureSpec":
        return cls(**dict(raw)).validate()

    @classmethod
    def from_genome(
        cls,
        genome: GeneralistGenome,
        *,
        target_vocab_size: int = 1024,
        family: str | None = None,
    ) -> "ArchitectureSpec":
        genome.validate()
        resolved_family = str(family or genome.architecture_family)
        return cls(
            architecture_id=genome.genome_id,
            generation=genome.generation,
            parent_id=genome.parent_id,
            family=resolved_family,
            context_length=genome.context_length,
            d_model=genome.d_model,
            n_heads=genome.n_heads,
            n_layers=genome.n_layers,
            d_ff=genome.d_ff,
            dropout=genome.dropout,
            tokenizer_version=genome.tokenizer_version,
            target_vocab_size=int(target_vocab_size),
            norm_type=genome.norm_type,
            norm_placement=genome.norm_placement,
            position_encoding=genome.position_encoding,
            ff_variant=genome.ff_variant,
            attention_type=genome.attention_type,
            n_kv_heads=genome.n_kv_heads,
            local_attention_window=genome.local_attention_window,
            local_attention_every=genome.local_attention_every,
            tie_embeddings=genome.tie_embeddings,
            learning_rate=genome.learning_rate,
            reasoning_depth=genome.reasoning_depth,
            retrieval_adapter=genome.retrieval_adapter,
            symbolic_adapter=genome.symbolic_adapter,
            code_adapter=genome.code_adapter,
            data_adapter=genome.data_adapter,
            tool_protocol=genome.tool_protocol,
        ).validate()


def architecture_id(
    parent_id: str | None,
    generation: int,
    payload: dict[str, Any],
) -> str:
    material = {
        "version": ARCHITECTURE_IR_VERSION,
        "parent": parent_id,
        "generation": int(generation),
        "payload": dict(payload),
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"airi-arch-{int(generation)}-{digest}"


def architecture_manifest(spec: ArchitectureSpec) -> dict[str, Any]:
    spec.validate()
    return {
        "schema": 1,
        "version": ARCHITECTURE_IR_VERSION,
        "architecture": spec.to_dict(),
        "fingerprint": spec.fingerprint(),
        "parameter_estimate": spec.parameter_estimate(),
        "executable_payload_allowed": False,
        "external_pretrained_weights": False,
    }
