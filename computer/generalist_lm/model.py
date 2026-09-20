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
_ALLOWED_POSITIONS = {"learned", "sinusoidal", "rope"}
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
        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
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
                if config.position_encoding == "rope":
                    inv_freq = 1.0 / (
                        10000.0
                        ** (
                            torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                            / self.head_dim
                        )
                    )
                    self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
                else:
                    self.register_buffer("rope_inv_freq", torch.empty(0), persistent=False)

            @staticmethod
            def _rotate_half(x):
                even = x[..., 0::2]
                odd = x[..., 1::2]
                return torch.stack((-odd, even), dim=-1).flatten(-2)

            def _apply_rope(self, q, k, *, position_offset: int = 0):
                seqlen = q.shape[-2]
                positions = torch.arange(
                    int(position_offset),
                    int(position_offset) + seqlen,
                    device=q.device,
                    dtype=self.rope_inv_freq.dtype,
                )
                freqs = torch.outer(positions, self.rope_inv_freq.to(q.device))
                angles = torch.repeat_interleave(freqs, 2, dim=-1)
                cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
                sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]
                q_rot = q * cos + self._rotate_half(q) * sin
                k_rot = k * cos + self._rotate_half(k) * sin
                return q_rot, k_rot

            def forward(self, x, *, past_key_value=None, use_cache: bool = False):
                bsz, seqlen, width = x.shape
                qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
                q, k, v = qkv.unbind(dim=2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                past_len = 0
                if past_key_value is not None:
                    past_k, past_v = past_key_value
                    if past_k.shape[:2] != (bsz, self.n_heads) or past_v.shape != past_k.shape:
                        raise ValueError("invalid attention KV cache shape")
                    if past_k.shape[-1] != self.head_dim:
                        raise ValueError("invalid attention KV cache head dimension")
                    past_len = int(past_k.shape[-2])
                else:
                    past_k = past_v = None

                if config.position_encoding == "rope":
                    q, k = self._apply_rope(q, k, position_offset=past_len)

                if past_k is not None:
                    k = torch.cat([past_k, k], dim=-2)
                    v = torch.cat([past_v, v], dim=-2)

                total_len = int(k.shape[-2])
                if past_len == 0:
                    attn_mask = None
                    is_causal = True
                else:
                    query_positions = torch.arange(
                        past_len,
                        past_len + seqlen,
                        device=x.device,
                    )[:, None]
                    key_positions = torch.arange(total_len, device=x.device)[None, :]
                    attn_mask = key_positions <= query_positions
                    is_causal = False

                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
                present = (k, v) if use_cache else None
                return self.out(y), present

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

            def forward(self, x, *, past_key_value=None, use_cache: bool = False):
                attn_out, present = self.attn(
                    self.attn_norm(x),
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                x = x + self.drop(attn_out)
                x = x + self.drop(self.ff(self.ff_norm(x)))
                return x, present

        class DecoderOnlyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
                if config.position_encoding == "learned":
                    self.position_embedding = nn.Embedding(config.context_length, config.d_model)
                    self.register_buffer("fixed_position_encoding", torch.empty(0), persistent=False)
                elif config.position_encoding == "sinusoidal":
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
                else:
                    self.position_embedding = None
                    self.register_buffer(
                        "fixed_position_encoding",
                        torch.zeros(config.context_length, config.d_model),
                        persistent=False,
                    )
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

            def forward(self, input_ids, labels=None, *, past_key_values=None, use_cache: bool = False):
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, time]")
                _bsz, seqlen = input_ids.shape

                if past_key_values is not None:
                    if labels is not None:
                        raise ValueError("labels are not supported with past_key_values")
                    if len(past_key_values) != len(self.blocks):
                        raise ValueError("past_key_values length must match transformer layers")
                    past_len = int(past_key_values[0][0].shape[-2]) if past_key_values else 0
                    if any(int(item[0].shape[-2]) != past_len for item in past_key_values):
                        raise ValueError("all KV cache layers must have the same sequence length")
                else:
                    past_len = 0
                    past_key_values = [None] * len(self.blocks)

                total_len = past_len + seqlen
                if total_len > config.context_length:
                    raise ValueError("sequence plus KV cache exceeds model context_length")

                positions = torch.arange(
                    past_len,
                    total_len,
                    device=input_ids.device,
                )
                x = self.token_embedding(input_ids) + self._position_values(positions)[None, :, :]
                presents = []
                for block, past in zip(self.blocks, past_key_values):
                    x, present = block(x, past_key_value=past, use_cache=use_cache)
                    if use_cache:
                        presents.append(present)
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
                return {
                    "logits": logits,
                    "loss": loss,
                    "past_key_values": tuple(presents) if use_cache else None,
                }

            @torch.no_grad()
            def generate(
                self,
                input_ids,
                *,
                max_new_tokens: int = 64,
                eos_token_id: int | None = 2,
                temperature: float = 0.0,
                top_k: int | None = None,
                use_cache: bool = True,
            ):
                self.eval()
                out = input_ids
                past_key_values = None
                next_input = out[:, -config.context_length:]

                for _ in range(max(0, int(max_new_tokens))):
                    if use_cache and past_key_values is not None:
                        result = self(
                            next_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    else:
                        window = out[:, -config.context_length:]
                        result = self(window, use_cache=bool(use_cache))
                    logits = result["logits"][:, -1, :]
                    past_key_values = result["past_key_values"] if use_cache else None

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

                    if use_cache and past_key_values is not None:
                        cache_len = int(past_key_values[0][0].shape[-2]) if past_key_values else 0
                        if cache_len >= config.context_length:
                            # Sliding-window semantics reset position indices,
                            # so recompute the bounded window rather than using
                            # stale absolute positions beyond the context limit.
                            past_key_values = None
                            next_input = out[:, -config.context_length:]
                        else:
                            next_input = next_token
                    else:
                        next_input = out[:, -config.context_length:]
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
