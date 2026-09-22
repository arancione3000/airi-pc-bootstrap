from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .architecture_ir import ArchitectureSpec
from .model import (
    estimate_flops_per_token,
    parameter_count,
)
from .model_registry import build_causal_lm


@dataclass(frozen=True)
class BuiltArchitecture:
    spec: ArchitectureSpec
    model: Any
    parameter_count: int
    flops_per_token_estimate: int

    def report(self) -> dict[str, Any]:
        return {
            "architecture_id": self.spec.architecture_id,
            "fingerprint": self.spec.fingerprint(),
            "family": self.spec.family,
            "parameters": int(self.parameter_count),
            "flops_per_token_estimate": int(self.flops_per_token_estimate),
            "context_length": int(self.spec.context_length),
            "d_model": int(self.spec.d_model),
            "n_layers": int(self.spec.n_layers),
            "n_heads": int(self.spec.n_heads),
            "attention_type": self.spec.attention_type,
            "n_kv_heads": self.spec.n_kv_heads,
            "norm_type": self.spec.norm_type,
            "norm_placement": self.spec.norm_placement,
            "position_encoding": self.spec.position_encoding,
            "ff_variant": self.spec.ff_variant,
            "local_attention_window": int(self.spec.local_attention_window),
            "local_attention_every": int(self.spec.local_attention_every),
            "tie_embeddings": bool(self.spec.tie_embeddings),
            "external_pretrained_weights": False,
        }


def build_architecture(
    spec: ArchitectureSpec,
    *,
    vocab_size: int | None = None,
) -> BuiltArchitecture:
    spec.validate()
    config = spec.to_model_config(vocab_size=vocab_size)
    model = build_causal_lm(config)
    return BuiltArchitecture(
        spec=spec,
        model=model,
        parameter_count=parameter_count(model),
        flops_per_token_estimate=estimate_flops_per_token(config),
    )
