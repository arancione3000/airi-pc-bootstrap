from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def _torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    return torch, nn, F


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

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_norm = nn.LayerNorm(config.d_model)
                self.ff_norm = nn.LayerNorm(config.d_model)
                self.attn = CausalSelfAttention()
                self.ff = SwiGLU()
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
                self.position_embedding = nn.Embedding(config.context_length, config.d_model)
                self.blocks = nn.ModuleList([Block() for _ in range(config.n_layers)])
                self.final_norm = nn.LayerNorm(config.d_model)
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

            def forward(self, input_ids, labels=None):
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, time]")
                bsz, seqlen = input_ids.shape
                if seqlen > config.context_length:
                    raise ValueError("sequence exceeds model context_length")
                positions = torch.arange(seqlen, device=input_ids.device)
                x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
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
    # Coarse dense-transformer estimate used only as an evolution cost signal.
    return int(cfg.n_layers * (4 * cfg.d_model * cfg.d_model + 3 * cfg.d_model * cfg.d_ff))


def estimate_parameter_count(config: GeneralistLMConfig) -> int:
    cfg = config.validate()
    d = cfg.d_model
    ff = cfg.d_ff
    embeddings = cfg.vocab_size * d + cfg.context_length * d
    per_layer = 4 * d * d + 3 * d * ff + 4 * d
    if cfg.bias:
        per_layer += 5 * d + 2 * ff
    final_norm = 2 * d
    return int(embeddings + cfg.n_layers * per_layer + final_norm)
