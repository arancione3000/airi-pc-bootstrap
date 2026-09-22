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
_ALLOWED_FF = {"swiglu", "gelu", "moe_swiglu"}
_ALLOWED_TOKENIZERS = {"byte-v1", "bpe-v1"}
_ALLOWED_ATTENTION = {"mha", "gqa"}
_ALLOWED_NORM_PLACEMENT = {"pre", "post"}


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
    attention_type: str = "mha"
    n_kv_heads: int | None = None
    local_attention_window: int = 0
    local_attention_every: int = 0
    norm_placement: str = "pre"
    tie_embeddings: bool = True
    recurrent_depth: int = 1
    moe_experts: int = 1
    moe_top_k: int = 1

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
        if self.tokenizer_version not in _ALLOWED_TOKENIZERS:
            raise ValueError("unsupported tokenizer_version")
        if self.norm_type not in _ALLOWED_NORMS:
            raise ValueError("unsupported norm_type")
        if self.position_encoding not in _ALLOWED_POSITIONS:
            raise ValueError("unsupported position_encoding")
        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.ff_variant not in _ALLOWED_FF:
            raise ValueError("unsupported ff_variant")
        if self.attention_type not in _ALLOWED_ATTENTION:
            raise ValueError("unsupported attention_type")
        if self.norm_placement not in _ALLOWED_NORM_PLACEMENT:
            raise ValueError("unsupported norm_placement")
        if self.attention_type == "mha":
            self.n_kv_heads = self.n_heads
        else:
            requested = self.n_kv_heads
            if requested is None:
                requested = max(1, self.n_heads // 2)
            self.n_kv_heads = max(1, min(self.n_heads, int(requested)))
            if self.n_heads % self.n_kv_heads:
                raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        self.local_attention_window = max(
            0,
            min(self.context_length, int(self.local_attention_window)),
        )
        self.local_attention_every = max(
            0,
            min(self.n_layers, int(self.local_attention_every)),
        )
        self.tie_embeddings = bool(self.tie_embeddings)
        self.recurrent_depth = max(1, min(4, int(self.recurrent_depth)))
        self.moe_experts = max(1, min(8, int(self.moe_experts)))
        self.moe_top_k = max(1, min(self.moe_experts, int(self.moe_top_k)))
        if self.ff_variant != "moe_swiglu":
            self.moe_experts = 1
            self.moe_top_k = 1
        elif self.moe_experts < 2:
            raise ValueError("moe_swiglu requires at least two experts")
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
            def __init__(self, layer_index: int):
                super().__init__()
                self.n_heads = config.n_heads
                self.n_kv_heads = int(config.n_kv_heads or config.n_heads)
                self.head_dim = config.d_model // config.n_heads
                self.group_size = self.n_heads // self.n_kv_heads
                self.layer_index = int(layer_index)
                if config.attention_type == "mha":
                    # Keep the legacy parameter names exactly so old checkpoints load.
                    self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
                    self.q_proj = self.k_proj = self.v_proj = None
                else:
                    self.qkv = None
                    self.q_proj = nn.Linear(
                        config.d_model,
                        self.n_heads * self.head_dim,
                        bias=config.bias,
                    )
                    self.k_proj = nn.Linear(
                        config.d_model,
                        self.n_kv_heads * self.head_dim,
                        bias=config.bias,
                    )
                    self.v_proj = nn.Linear(
                        config.d_model,
                        self.n_kv_heads * self.head_dim,
                        bias=config.bias,
                    )
                self.out = nn.Linear(config.d_model, config.d_model, bias=config.bias)
                self.dropout = config.dropout
                every = int(config.local_attention_every)
                self.local_window = 0
                if int(config.local_attention_window) > 0:
                    if every <= 0 or ((self.layer_index + 1) % every) != 0:
                        self.local_window = int(config.local_attention_window)
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
                if self.qkv is not None:
                    qkv = self.qkv(x).view(
                        bsz,
                        seqlen,
                        3,
                        self.n_heads,
                        self.head_dim,
                    )
                    q, k, v = qkv.unbind(dim=2)
                else:
                    q = self.q_proj(x).view(
                        bsz,
                        seqlen,
                        self.n_heads,
                        self.head_dim,
                    )
                    k = self.k_proj(x).view(
                        bsz,
                        seqlen,
                        self.n_kv_heads,
                        self.head_dim,
                    )
                    v = self.v_proj(x).view(
                        bsz,
                        seqlen,
                        self.n_kv_heads,
                        self.head_dim,
                    )
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                past_len = 0
                if past_key_value is not None:
                    past_k, past_v = past_key_value
                    if past_k.shape[:2] != (bsz, self.n_kv_heads) or past_v.shape != past_k.shape:
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

                present = (k, v) if use_cache else None
                if self.n_kv_heads != self.n_heads:
                    k_attn = k.repeat_interleave(self.group_size, dim=1)
                    v_attn = v.repeat_interleave(self.group_size, dim=1)
                else:
                    k_attn = k
                    v_attn = v

                total_len = int(k_attn.shape[-2])
                if past_len == 0 and self.local_window <= 0:
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
                    if self.local_window > 0:
                        earliest = query_positions - self.local_window + 1
                        attn_mask = attn_mask & (key_positions >= earliest)
                    is_causal = False

                y = F.scaled_dot_product_attention(
                    q, k_attn, v_attn,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
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

        class SparseMoE(nn.Module):
            """Top-k routed SwiGLU experts with an ONNX-safe reference kernel.

            Routing is sparse: only top_k expert outputs contribute to a token.
            The reference kernel evaluates all experts so export stays portable;
            measured latency and FLOP accounting therefore charge the real dense
            execution cost instead of pretending the sparse kernel already exists.
            """
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(
                    config.d_model,
                    config.moe_experts,
                    bias=False,
                )
                self.experts = nn.ModuleList([
                    SwiGLU() for _ in range(config.moe_experts)
                ])
                self.top_k = int(config.moe_top_k)

            def forward(self, x):
                router_logits = self.router(x)
                top_values, top_indices = torch.topk(
                    router_logits,
                    k=self.top_k,
                    dim=-1,
                )
                top_weights = torch.softmax(top_values, dim=-1)
                gates = torch.zeros_like(router_logits).scatter(
                    -1,
                    top_indices,
                    top_weights,
                )
                expert_outputs = torch.stack(
                    [expert(x) for expert in self.experts],
                    dim=-2,
                )
                return torch.sum(
                    expert_outputs * gates.unsqueeze(-1),
                    dim=-2,
                )

        class Block(nn.Module):
            def __init__(self, layer_index: int):
                super().__init__()
                self.attn_norm = norm()
                self.ff_norm = norm()
                self.attn = CausalSelfAttention(layer_index)
                if config.ff_variant == "swiglu":
                    self.ff = SwiGLU()
                elif config.ff_variant == "moe_swiglu":
                    self.ff = SparseMoE()
                else:
                    self.ff = GeluFF()
                self.drop = nn.Dropout(config.dropout)

            def forward(self, x, *, past_key_value=None, use_cache: bool = False):
                if config.norm_placement == "pre":
                    attn_out, present = self.attn(
                        self.attn_norm(x),
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                    )
                    x = x + self.drop(attn_out)
                    x = x + self.drop(self.ff(self.ff_norm(x)))
                    return x, present

                attn_out, present = self.attn(
                    x,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                x = self.attn_norm(x + self.drop(attn_out))
                x = self.ff_norm(x + self.drop(self.ff(x)))
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
                self.blocks = nn.ModuleList([
                    Block(layer_index)
                    for layer_index in range(config.n_layers)
                ])
                self.final_norm = (
                    norm() if config.norm_placement == "pre" else nn.Identity()
                )
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                if config.tie_embeddings:
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

                recurrent_layers = len(self.blocks) * int(config.recurrent_depth)
                if past_key_values is not None:
                    if labels is not None:
                        raise ValueError("labels are not supported with past_key_values")
                    if len(past_key_values) != recurrent_layers:
                        raise ValueError(
                            "past_key_values length must match physical layers times recurrent_depth"
                        )
                    past_len = int(past_key_values[0][0].shape[-2]) if past_key_values else 0
                    if any(int(item[0].shape[-2]) != past_len for item in past_key_values):
                        raise ValueError("all KV cache layers must have the same sequence length")
                else:
                    past_len = 0
                    past_key_values = [None] * recurrent_layers

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
                cache_index = 0
                for block in self.blocks:
                    for _recurrent_step in range(int(config.recurrent_depth)):
                        past = past_key_values[cache_index]
                        x, present = block(
                            x,
                            past_key_value=past,
                            use_cache=use_cache,
                        )
                        if use_cache:
                            presents.append(present)
                        cache_index += 1
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
                top_p: float | None = None,
                repetition_penalty: float = 1.0,
                use_cache: bool = True,
            ):
                self.eval()
                out = input_ids
                prompt_length = int(input_ids.shape[1])
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
                        # Canonical RAW GREEDY: sampling controls intentionally
                        # cannot alter deterministic evaluation output.
                        next_token = logits.argmax(dim=-1, keepdim=True)
                    else:
                        penalty = max(1.0, float(repetition_penalty or 1.0))
                        if penalty > 1.0 and int(out.shape[1]) > prompt_length:
                            logits = logits.clone()
                            for batch_index in range(int(out.shape[0])):
                                repeated = torch.unique(out[batch_index, prompt_length:])
                                values = logits[batch_index, repeated]
                                logits[batch_index, repeated] = torch.where(
                                    values < 0,
                                    values * penalty,
                                    values / penalty,
                                )
                        scaled = logits / max(1e-5, float(temperature))
                        if top_k is not None and 0 < int(top_k) < scaled.shape[-1]:
                            values, _ = torch.topk(scaled, int(top_k))
                            cutoff = values[:, -1].unsqueeze(-1)
                            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                        if top_p is not None and 0.0 < float(top_p) < 1.0:
                            sorted_logits, sorted_indices = torch.sort(
                                scaled,
                                descending=True,
                                dim=-1,
                            )
                            sorted_probs = torch.softmax(sorted_logits, dim=-1)
                            cumulative = torch.cumsum(sorted_probs, dim=-1)
                            remove = cumulative > float(top_p)
                            remove[..., 1:] = remove[..., :-1].clone()
                            remove[..., 0] = False
                            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                            filtered = torch.full_like(scaled, float("-inf"))
                            filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
                            scaled = filtered
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


def _attention_projection_terms(config: GeneralistLMConfig) -> tuple[int, int]:
    cfg = config.validate()
    d = int(cfg.d_model)
    if cfg.attention_type == "mha":
        # QKV + output projection.
        return 4 * d * d, 4 * d
    kv_dim = int(cfg.n_kv_heads or cfg.n_heads) * (d // int(cfg.n_heads))
    # Q + K + V + output.
    weights = 2 * d * d + 2 * d * kv_dim
    biases = 2 * d + 2 * kv_dim
    return int(weights), int(biases)


def estimate_flops_per_token(config: GeneralistLMConfig) -> int:
    cfg = config.validate()
    attention_weights, _attention_biases = _attention_projection_terms(cfg)
    if cfg.ff_variant == "gelu":
        active_ff = 2 * cfg.d_model * cfg.d_ff
    elif cfg.ff_variant == "moe_swiglu":
        # The portable reference kernel evaluates every expert. Top-k routing
        # is structurally sparse, but compute accounting stays honest.
        active_ff = (
            cfg.moe_experts * cfg.d_model
            + cfg.moe_experts * 3 * cfg.d_model * cfg.d_ff
        )
    else:
        active_ff = 3 * cfg.d_model * cfg.d_ff
    return int(
        cfg.n_layers
        * cfg.recurrent_depth
        * (attention_weights + active_ff)
    )


def estimate_parameter_count(config: GeneralistLMConfig) -> int:
    cfg = config.validate()
    d = cfg.d_model
    ff = cfg.d_ff
    position_params = cfg.context_length * d if cfg.position_encoding == "learned" else 0
    embeddings = cfg.vocab_size * d + position_params
    if not cfg.tie_embeddings:
        embeddings += cfg.vocab_size * d
    norm_params = 4 * d if cfg.norm_type == "layernorm" else 2 * d
    attention_weights, attention_bias = _attention_projection_terms(cfg)
    if cfg.ff_variant == "gelu":
        ff_weights = 2 * d * ff
        ff_bias = ff + d
    elif cfg.ff_variant == "moe_swiglu":
        ff_weights = cfg.moe_experts * (3 * d * ff) + d * cfg.moe_experts
        ff_bias = cfg.moe_experts * (2 * ff + d)
    else:
        ff_weights = 3 * d * ff
        ff_bias = 2 * ff + d
    per_layer = attention_weights + ff_weights + norm_params
    if cfg.bias:
        per_layer += attention_bias + ff_bias
    if cfg.norm_placement == "post":
        final_norm = 0
    else:
        final_norm = 2 * d if cfg.norm_type == "layernorm" else d
    # recurrent_depth reuses the same physical block parameters.
    return int(embeddings + cfg.n_layers * per_layer + final_norm)
