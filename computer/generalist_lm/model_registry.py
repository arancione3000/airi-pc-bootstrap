from __future__ import annotations

from .model import CausalTransformerLM, GeneralistLMConfig
from .recurrent_model import CausalGatedRecurrentLM


def build_causal_lm(config: GeneralistLMConfig):
    """Instantiate one trusted AIRI architecture family from config.

    Architecture Search can select a registered family, but it cannot execute
    arbitrary model code. New families must be implemented here and pass the
    same verifier/training/ONNX gates before they become selectable.
    """
    cfg = config.validate()
    if cfg.architecture_family == "decoder_transformer_v1":
        return CausalTransformerLM(cfg)
    if cfg.architecture_family == "gated_recurrent_v1":
        return CausalGatedRecurrentLM(cfg)
    raise ValueError(
        f"unregistered AIRI architecture family: {cfg.architecture_family}"
    )
