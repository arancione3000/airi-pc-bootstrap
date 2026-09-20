from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def _torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    return torch, nn, F


_ALLOWED_NORMS = {"layernorm", "rmsnorm"}
_ALLOWED_POSITIONS = {"learned", "sinusoidal"}
_ALLOWED_FF = {"swiglu", "gelu"}


@dataclass
class GeneralistLMConfig:
    vocab_size: int = 264
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.0
    bias: bool = False
    tokenizer_version: str = "byte-v1"
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    ff_variant: str = "swiglu"

    def validate(self) -> "GeneralistLMConfig":
        self.vocab_size = max(16, int(self.vocab_size))
        self.context_length = max(32, min(8192, int(self.context_length)))
        self.d_model = max(32, min(4096, int(self.d_model)))
        self.n_heads = max(1, min(64, int(self.n_heads)))
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_layers = max(1, min(96, int(self.n_layers)))
        self.d_ff = max(self.d_model, min(16384, int(self.d_ff)))
        self.dropout = min(0.5, max(0.0, float(self.dropout)))
        if self.norm_type not in _ALLOWED_NORMS:
            raise ValueError("unsupported norm_type")
        if self.position_encoding not in _ALLOWED_POSITIONS:
            raise ValueError("unsupported position_encoding")
        if self.ff_variant not in _ALLOWED_FF:
            raise ValueError("unsupported ff_variant")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GeneralistLMConfig":
        return cls(**dict(raw)).validate()


class CausalTransformerLM:
    """Factory wrapper returning a real decoder-only causal language model."""

    def __new__(cls, config: GeneralistLMConfig):
        torch, nn, F = _torch()
        config = config.validate()

        class RMSNorm(nn.Module):
            def __init__(self, width: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(width))
                self.eps = eps

            def forward(self, x):
                scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return x * scale * self.weight

        def norm():
            if config.norm_type == "rmsnorm":
                return RMSNorm(config.d_model)
            return nn.LayerNorm(config.d_model)

        class CausalSelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = config.n_heads
                self.head_dim = config.d_model // config.n_heads
                self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
                self.out = nn.Linear(config.d_model, config.d_model, bias=config.bias)
                self.dropout = config.dropout

            def forward(self, x):
                bsz, seqlen, width = x.shape
                qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
                q, k, v = qkv.unbind(dim=2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
                return self.out(y)

        class SwiGLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(config.d_model, 2 * config.d_ff, bias=config.bias)
                self.down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

            def forward(self, x):
                gate, value = self.up(x).chunk(2, dim=-1)
                return self.down(F.silu(gate) * value)

        class GeluFF(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
                self.down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

            def forward(self, x):
                return self.down(F.gelu(self.up(x)))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_norm = norm()
                self.ff_norm = norm()
                self.attn = CausalSelfAttention()
                self.ff = SwiGLU() if config.ff_variant == "swiglu" else GeluFF()
                self.drop = nn.Dropout(config.dropout)

            def forward(self, x):
                x = x + self.drop(self.attn(self.attn_norm(x)))
                x = x + self.drop(self.ff(self.ff_norm(x)))
                return x

        class DecoderOnlyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
                if config.position_encoding == "learned":
                    self.position_embedding = nn.Embedding(config.context_length, config.d_model)
                    self.register_buffer("fixed_position_encoding", torch.empty(0), persistent=False)
                else:
                    self.position_embedding = None
                    position = torch.arange(config.context_length, dtype=torch.float32).unsqueeze(1)
                    div = torch.exp(
                        torch.arange(0, config.d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / config.d_model)
                    )
                    pe = torch.zeros(config.context_length, config.d_model)
                    pe[:, 0::2] = torch.sin(position * div)
                    odd = pe[:, 1::2].shape[1]
                    pe[:, 1::2] = torch.cos(position * div[:odd])
                    self.register_buffer("fixed_position_encoding", pe, persistent=True)
                self.blocks = nn.ModuleList([Block() for _ in range(config.n_layers)])
                self.final_norm = norm()
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                self.lm_head.weight = self.token_embedding.weight
                self.apply(self._init_weights)

            @staticmethod
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

            def _position_values(self, positions):
                if self.position_embedding is not None:
                    return self.position_embedding(positions)
                return self.fixed_position_encoding.index_select(0, positions)

            def forward(self, input_ids, labels=None):
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, time]")
                _bsz, seqlen = input_ids.shape
                if seqlen > config.context_length:
                    raise ValueError("sequence exceeds model context_length")
                positions = torch.arange(seqlen, device=input_ids.device)
                x = self.token_embedding(input_ids) + self._position_values(positions)[None, :, :]
                for block in self.blocks:
                    x = block(x)
                logits = self.lm_head(self.final_norm(x))
                loss = None
                if labels is not None:
                    if labels.shape != input_ids.shape:
                        raise ValueError("labels must match input_ids shape")
                    loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, config.vocab_size),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )
                return {"logits": logits, "loss": loss}

            @torch.no_grad()
            def generate(
                self,
                input_ids,
                *,
                max_new_tokens: int = 64,
                eos_token_id: int | None = 2,
                temperature: float = 0.0,
                top_k: int | None = None,
            ):
                self.eval()
                out = input_ids
                for _ in range(max(0, int(max_new_tokens))):
                    window = out[:, -config.context_length:]
                    logits = self(window)["logits"][:, -1, :]
                    if temperature is None or float(temperature) <= 0.0:
                        next_token = logits.argmax(dim=-1, keepdim=True)
                    else:
                        scaled = logits / max(1e-5, float(temperature))
                        if top_k is not None and 0 < int(top_k) < scaled.shape[-1]:
                            values, _ = torch.topk(scaled, int(top_k))
                            cutoff = values[:, -1].unsqueeze(-1)
                            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                        probs = torch.softmax(scaled, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    out = torch.cat([out, next_token], dim=1)
                    if eos_token_id is not None and bool(torch.all(next_token == int(eos_token_id))):
                        break
                return out

        return DecoderOnlyLM()


def parameter_count(model) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def estimate_flops_per_token(config: GeneralistLMConfig) -> int:
    cfg = config.validate()
    ff_multiplier = 3 if cfg.ff_variant == "swiglu" else 2
    return int(cfg.n_layers * (4 * cfg.d_model * cfg.d_model + ff_multiplier * cfg.d_model * cfg.d_ff))


def estimate_parameter_count(config: GeneralistLMConfig) -> int:
    cfg = config.validate()
    d = cfg.d_model
    ff = cfg.d_ff
    position_params = cfg.context_length * d if cfg.position_encoding == "learned" else 0
    embeddings = cfg.vocab_size * d + position_params
    ff_multiplier = 3 if cfg.ff_variant == "swiglu" else 2
    norm_params = 4 * d if cfg.norm_type == "layernorm" else 2 * d
    per_layer = 4 * d * d + ff_multiplier * d * ff + norm_params
    if cfg.bias:
        qkv_out_bias = 4 * d
        ff_bias = (2 * ff + d) if cfg.ff_variant == "swiglu" else (ff + d)
        per_layer += qkv_out_bias + ff_bias
    final_norm = 2 * d if cfg.norm_type == "layernorm" else d
    return int(embeddings + cfg.n_layers * per_layer + final_norm)
