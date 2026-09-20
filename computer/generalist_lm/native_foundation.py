from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


NATIVE_FAMILY = "airi-native-foundation"
NATIVE_ARCHITECTURE_VERSION = "native-foundation-v1"
NATIVE_CHECKPOINT_VERSION = 1
NATIVE_MANIFEST_FILENAME = "airi-native-foundation.json"
NATIVE_CONFIG_FILENAME = "native-config.json"
NATIVE_MODEL_FILENAME = "model.pt"
NATIVE_TOKENIZER_FILENAME = "tokenizer.json"

_ALLOWED_TOKENIZERS = {"byte-v1", "bpe-v1"}
_ALLOWED_WEIGHT_ORIGINS = {"random-init", "airi-native-descendant"}


def _torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    return torch, nn, F


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class NativeFoundationConfig:
    vocab_size: int = 264
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    init_std: float = 0.02
    tokenizer_version: str = "byte-v1"
    tie_embeddings: bool = True
    architecture_version: str = NATIVE_ARCHITECTURE_VERSION

    def validate(self) -> "NativeFoundationConfig":
        self.vocab_size = int(self.vocab_size)
        self.context_length = int(self.context_length)
        self.d_model = int(self.d_model)
        self.n_heads = int(self.n_heads)
        self.n_kv_heads = int(self.n_kv_heads)
        self.n_layers = int(self.n_layers)
        self.d_ff = int(self.d_ff)
        self.dropout = float(self.dropout)
        self.rope_theta = float(self.rope_theta)
        self.rms_eps = float(self.rms_eps)
        self.init_std = float(self.init_std)

        if not (16 <= self.vocab_size <= 262144):
            raise ValueError("vocab_size out of native foundation bounds")
        if not (32 <= self.context_length <= 131072):
            raise ValueError("context_length out of native foundation bounds")
        if not (32 <= self.d_model <= 16384):
            raise ValueError("d_model out of native foundation bounds")
        if not (1 <= self.n_heads <= 128):
            raise ValueError("n_heads out of native foundation bounds")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not (1 <= self.n_kv_heads <= self.n_heads):
            raise ValueError("n_kv_heads out of native foundation bounds")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if not (1 <= self.n_layers <= 192):
            raise ValueError("n_layers out of native foundation bounds")
        if not (self.d_model <= self.d_ff <= 65536):
            raise ValueError("d_ff out of native foundation bounds")
        if not (0.0 <= self.dropout <= 0.5):
            raise ValueError("dropout out of native foundation bounds")
        if not (1000.0 <= self.rope_theta <= 10_000_000.0):
            raise ValueError("rope_theta out of native foundation bounds")
        if not (1e-8 <= self.rms_eps <= 1e-3):
            raise ValueError("rms_eps out of native foundation bounds")
        if not (1e-5 <= self.init_std <= 0.2):
            raise ValueError("init_std out of native foundation bounds")
        if self.tokenizer_version not in _ALLOWED_TOKENIZERS:
            raise ValueError("unsupported native tokenizer_version")
        if self.architecture_version != NATIVE_ARCHITECTURE_VERSION:
            raise ValueError("unsupported native architecture_version")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NativeFoundationConfig":
        if not isinstance(raw, dict):
            raise ValueError("native foundation config must be an object")
        return cls(**dict(raw)).validate()


@dataclass(frozen=True)
class NativeFoundationProvenance:
    family: str
    architecture_version: str
    checkpoint_version: int
    weights_origin: str
    root_seed: int
    tokenizer_version: str
    tokenizer_digest: str | None
    config_digest: str
    model_digest: str
    parent_checkpoint_digest: str | None = None
    external_pretrained: bool = False

    def validate(self) -> "NativeFoundationProvenance":
        if self.family != NATIVE_FAMILY:
            raise ValueError("checkpoint is not AIRI Native Foundation")
        if self.architecture_version != NATIVE_ARCHITECTURE_VERSION:
            raise ValueError("unsupported native architecture version")
        if int(self.checkpoint_version) != NATIVE_CHECKPOINT_VERSION:
            raise ValueError("unsupported native checkpoint version")
        if self.weights_origin not in _ALLOWED_WEIGHT_ORIGINS:
            raise ValueError("unauthorized native weight origin")
        if self.external_pretrained:
            raise ValueError("external pretrained weights are forbidden in AIRI Native Foundation")
        if self.weights_origin == "random-init" and self.parent_checkpoint_digest is not None:
            raise ValueError("random-init root must not declare a parent checkpoint")
        if self.weights_origin == "airi-native-descendant" and not self.parent_checkpoint_digest:
            raise ValueError("native descendant must declare a parent checkpoint digest")
        if self.tokenizer_version not in _ALLOWED_TOKENIZERS:
            raise ValueError("unsupported tokenizer in native provenance")
        if self.tokenizer_version == "bpe-v1" and not self.tokenizer_digest:
            raise ValueError("bpe-v1 native checkpoint requires tokenizer digest")
        if self.tokenizer_version == "byte-v1" and self.tokenizer_digest is not None:
            raise ValueError("byte-v1 native checkpoint must not declare tokenizer digest")
        for label, value in {
            "config_digest": self.config_digest,
            "model_digest": self.model_digest,
        }.items():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid {label}")
        if self.tokenizer_digest is not None and len(self.tokenizer_digest) != 64:
            raise ValueError("invalid tokenizer_digest")
        if self.parent_checkpoint_digest is not None and len(self.parent_checkpoint_digest) != 64:
            raise ValueError("invalid parent_checkpoint_digest")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NativeFoundationProvenance":
        if not isinstance(raw, dict):
            raise ValueError("native provenance must be an object")
        return cls(**dict(raw)).validate()


class NativeFoundationLM:
    """Factory for AIRI's scratch-built decoder-only Foundation family.

    Architecture v1 uses:
      - pre-norm RMSNorm
      - rotary positional embeddings
      - grouped-query attention
      - SwiGLU feed-forward blocks
      - optional tied token/output embeddings
      - KV-cache generation
    """

    def __new__(cls, config: NativeFoundationConfig):
        torch, nn, F = _torch()
        config = config.validate()

        class RMSNorm(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(width))

            def forward(self, x):
                scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + config.rms_eps)
                return (x * scale.to(dtype=x.dtype)) * self.weight

        class GroupedQueryAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = config.n_heads
                self.n_kv_heads = config.n_kv_heads
                self.head_dim = config.d_model // config.n_heads
                self.kv_repeat = config.n_heads // config.n_kv_heads
                self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
                self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
                self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
                self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

                inv_freq = 1.0 / (
                    config.rope_theta
                    ** (
                        torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                        / self.head_dim
                    )
                )
                self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

            @staticmethod
            def _rotate_half(x):
                even = x[..., 0::2]
                odd = x[..., 1::2]
                return torch.stack((-odd, even), dim=-1).flatten(-2)

            def _rope(self, q, k, *, position_offset: int):
                seqlen = q.shape[-2]
                positions = torch.arange(
                    position_offset,
                    position_offset + seqlen,
                    device=q.device,
                    dtype=self.rope_inv_freq.dtype,
                )
                freqs = torch.outer(positions, self.rope_inv_freq.to(q.device))
                angles = torch.repeat_interleave(freqs, 2, dim=-1)
                cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
                sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]
                return (
                    q * cos + self._rotate_half(q) * sin,
                    k * cos + self._rotate_half(k) * sin,
                )

            def forward(self, x, *, past_key_value=None, use_cache: bool = False):
                bsz, seqlen, _ = x.shape
                q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

                past_len = 0
                if past_key_value is not None:
                    past_k, past_v = past_key_value
                    if past_k.shape != past_v.shape:
                        raise ValueError("invalid native KV cache")
                    if past_k.ndim != 4 or past_k.shape[0] != bsz:
                        raise ValueError("invalid native KV cache shape")
                    if past_k.shape[1] != self.n_kv_heads or past_k.shape[-1] != self.head_dim:
                        raise ValueError("invalid native KV cache dimensions")
                    past_len = int(past_k.shape[-2])
                else:
                    past_k = past_v = None

                q, k = self._rope(q, k, position_offset=past_len)
                if past_k is not None:
                    k = torch.cat([past_k, k], dim=-2)
                    v = torch.cat([past_v, v], dim=-2)

                present = (k, v) if use_cache else None
                if self.kv_repeat > 1:
                    attn_k = k.repeat_interleave(self.kv_repeat, dim=1)
                    attn_v = v.repeat_interleave(self.kv_repeat, dim=1)
                else:
                    attn_k, attn_v = k, v

                total_len = int(attn_k.shape[-2])
                if past_len == 0:
                    attn_mask = None
                    is_causal = True
                else:
                    qpos = torch.arange(past_len, past_len + seqlen, device=x.device)[:, None]
                    kpos = torch.arange(total_len, device=x.device)[None, :]
                    attn_mask = kpos <= qpos
                    is_causal = False

                y = F.scaled_dot_product_attention(
                    q,
                    attn_k,
                    attn_v,
                    attn_mask=attn_mask,
                    dropout_p=config.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
                y = y.transpose(1, 2).contiguous().view(bsz, seqlen, config.d_model)
                return self.out_proj(y), present

        class SwiGLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
                self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
                self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

            def forward(self, x):
                return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_norm = RMSNorm(config.d_model)
                self.ff_norm = RMSNorm(config.d_model)
                self.attn = GroupedQueryAttention()
                self.ff = SwiGLU()
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, x, *, past_key_value=None, use_cache: bool = False):
                a, present = self.attn(
                    self.attn_norm(x),
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                x = x + self.dropout(a)
                x = x + self.dropout(self.ff(self.ff_norm(x)))
                return x, present

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.blocks = nn.ModuleList([Block() for _ in range(config.n_layers)])
                self.final_norm = RMSNorm(config.d_model)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                if config.tie_embeddings:
                    self.lm_head.weight = self.token_embedding.weight
                self.apply(self._init_weights)
                self._scale_residual_projections()

            @staticmethod
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=config.init_std)

            def _scale_residual_projections(self):
                std = config.init_std / math.sqrt(2.0 * config.n_layers)
                for block in self.blocks:
                    nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
                    nn.init.normal_(block.ff.down_proj.weight, mean=0.0, std=std)

            def forward(self, input_ids, labels=None, *, past_key_values=None, use_cache: bool = False):
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, time]")
                _bsz, seqlen = input_ids.shape
                if past_key_values is None:
                    past_key_values = [None] * len(self.blocks)
                    past_len = 0
                else:
                    if labels is not None:
                        raise ValueError("labels are not supported with past_key_values")
                    if len(past_key_values) != len(self.blocks):
                        raise ValueError("past_key_values length mismatch")
                    lengths = [int(item[0].shape[-2]) for item in past_key_values]
                    if len(set(lengths)) > 1:
                        raise ValueError("all native KV cache layers must have same length")
                    past_len = lengths[0] if lengths else 0

                if past_len + seqlen > config.context_length:
                    raise ValueError("sequence plus KV cache exceeds native context_length")

                x = self.token_embedding(input_ids)
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
                past = None
                next_input = out[:, -config.context_length:]
                for _ in range(max(0, int(max_new_tokens))):
                    if use_cache and past is not None:
                        result = self(next_input, past_key_values=past, use_cache=True)
                    else:
                        result = self(out[:, -config.context_length:], use_cache=use_cache)
                    logits = result["logits"][:, -1, :]
                    past = result["past_key_values"] if use_cache else None

                    if temperature is None or float(temperature) <= 0:
                        token = logits.argmax(dim=-1, keepdim=True)
                    else:
                        scaled = logits / max(1e-5, float(temperature))
                        if top_k is not None and 0 < int(top_k) < scaled.shape[-1]:
                            values, _ = torch.topk(scaled, int(top_k))
                            cutoff = values[:, -1].unsqueeze(-1)
                            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                        token = torch.multinomial(torch.softmax(scaled, dim=-1), 1)

                    out = torch.cat([out, token], dim=1)
                    if eos_token_id is not None and bool(torch.all(token == int(eos_token_id))):
                        break

                    if use_cache and past is not None:
                        cache_len = int(past[0][0].shape[-2]) if past else 0
                        if cache_len >= config.context_length:
                            past = None
                            next_input = out[:, -config.context_length:]
                        else:
                            next_input = token
                    else:
                        next_input = out[:, -config.context_length:]
                return out

        return Decoder()


def native_parameter_count(config: NativeFoundationConfig) -> int:
    cfg = config.validate()
    head_dim = cfg.d_model // cfg.n_heads
    q = cfg.d_model * (cfg.n_heads * head_dim)
    kv = 2 * cfg.d_model * (cfg.n_kv_heads * head_dim)
    out = cfg.d_model * cfg.d_model
    ff = 3 * cfg.d_model * cfg.d_ff
    norms = 2 * cfg.d_model
    per_layer = q + kv + out + ff + norms
    if cfg.bias:
        kv_width = cfg.n_kv_heads * head_dim
        per_layer += (
            (2 * cfg.d_model + 2 * kv_width)
            + (2 * cfg.d_ff + cfg.d_model)
        )
    embeddings = cfg.vocab_size * cfg.d_model
    output = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    final_norm = cfg.d_model
    return int(embeddings + output + cfg.n_layers * per_layer + final_norm)


def native_scale_profile(name: str, *, vocab_size: int = 32768) -> NativeFoundationConfig:
    profiles = {
        "micro": dict(context_length=2048, d_model=256, n_heads=8, n_kv_heads=2, n_layers=8, d_ff=768),
        "1b": dict(context_length=8192, d_model=2048, n_heads=16, n_kv_heads=4, n_layers=24, d_ff=5504),
        "3b": dict(context_length=16384, d_model=3072, n_heads=24, n_kv_heads=8, n_layers=32, d_ff=8192),
        "7b": dict(context_length=32768, d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32, d_ff=14336),
    }
    key = str(name).strip().lower()
    if key not in profiles:
        raise ValueError(f"unknown native scale profile: {name}")
    return NativeFoundationConfig(
        vocab_size=int(vocab_size),
        tokenizer_version="bpe-v1",
        **profiles[key],
    ).validate()


def create_native_root_checkpoint(
    state_dir: str | Path,
    config: NativeFoundationConfig,
    *,
    root_seed: int,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a root AIRI Native checkpoint from random initialization only.

    This API intentionally accepts no source-model or pretrained-weight path.
    """
    import torch

    cfg = config.validate()
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("native root checkpoint directory must be empty")

    tokenizer_digest = None
    if cfg.tokenizer_version == "bpe-v1":
        if tokenizer_path is None:
            raise ValueError("bpe-v1 native root requires an AIRI tokenizer artifact")
        source = Path(tokenizer_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("native tokenizer artifact does not exist")
        from .bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(source)
        if int(tokenizer.vocab_size) != int(cfg.vocab_size):
            raise ValueError("native config vocab_size must match AIRI BPE tokenizer")
        target = root / NATIVE_TOKENIZER_FILENAME
        tokenizer.save(target)
        tokenizer_digest = _sha256_file(target)
    elif tokenizer_path is not None:
        raise ValueError("byte-v1 native root does not accept tokenizer artifact")

    torch.manual_seed(int(root_seed))
    model = NativeFoundationLM(cfg)

    config_path = root / NATIVE_CONFIG_FILENAME
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    model_path = root / NATIVE_MODEL_FILENAME
    torch.save(model.state_dict(), model_path)

    config_digest = _sha256_json(cfg.to_dict())
    model_digest = _sha256_file(model_path)
    provenance = NativeFoundationProvenance(
        family=NATIVE_FAMILY,
        architecture_version=NATIVE_ARCHITECTURE_VERSION,
        checkpoint_version=NATIVE_CHECKPOINT_VERSION,
        weights_origin="random-init",
        root_seed=int(root_seed),
        tokenizer_version=cfg.tokenizer_version,
        tokenizer_digest=tokenizer_digest,
        config_digest=config_digest,
        model_digest=model_digest,
        parent_checkpoint_digest=None,
        external_pretrained=False,
    ).validate()
    (root / NATIVE_MANIFEST_FILENAME).write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return native_checkpoint_status(root)


def native_checkpoint_status(state_dir: str | Path) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    try:
        config_path = root / NATIVE_CONFIG_FILENAME
        model_path = root / NATIVE_MODEL_FILENAME
        manifest_path = root / NATIVE_MANIFEST_FILENAME
        for path in (config_path, model_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path.name)

        cfg = NativeFoundationConfig.from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        provenance = NativeFoundationProvenance.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        tokenizer_path = root / NATIVE_TOKENIZER_FILENAME
        current_tokenizer_digest = (
            _sha256_file(tokenizer_path)
            if cfg.tokenizer_version == "bpe-v1" and tokenizer_path.is_file()
            else None
        )
        if cfg.tokenizer_version == "bpe-v1" and current_tokenizer_digest is None:
            raise FileNotFoundError(NATIVE_TOKENIZER_FILENAME)

        current_config_digest = _sha256_json(cfg.to_dict())
        current_model_digest = _sha256_file(model_path)
        integrity_ok = bool(
            provenance.family == NATIVE_FAMILY
            and provenance.architecture_version == cfg.architecture_version
            and provenance.tokenizer_version == cfg.tokenizer_version
            and provenance.config_digest == current_config_digest
            and provenance.model_digest == current_model_digest
            and provenance.tokenizer_digest == current_tokenizer_digest
            and not provenance.external_pretrained
        )
        checkpoint_digest = _sha256_json({
            "config": current_config_digest,
            "model": current_model_digest,
            "tokenizer": current_tokenizer_digest,
            "provenance": provenance.to_dict(),
        })
        return {
            "ok": integrity_ok,
            "integrity_ok": integrity_ok,
            "family": NATIVE_FAMILY,
            "architecture_version": cfg.architecture_version,
            "weights_origin": provenance.weights_origin,
            "external_pretrained": provenance.external_pretrained,
            "root_seed": provenance.root_seed,
            "tokenizer_version": cfg.tokenizer_version,
            "tokenizer_digest": current_tokenizer_digest,
            "config_digest": current_config_digest,
            "model_digest": current_model_digest,
            "checkpoint_digest": checkpoint_digest,
            "parameters": native_parameter_count(cfg),
            "config": cfg.to_dict(),
            "reason": (
                "verified AIRI Native Foundation checkpoint with no external pretrained weights"
                if integrity_ok
                else "native checkpoint digest/provenance mismatch"
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "integrity_ok": False,
            "family": NATIVE_FAMILY,
            "reason": f"invalid_native_checkpoint:{type(exc).__name__}:{exc}",
        }


def load_native_checkpoint(state_dir: str | Path, *, device: str = "cpu"):
    import torch

    root = Path(state_dir).expanduser().resolve()
    status = native_checkpoint_status(root)
    if not status.get("ok"):
        raise RuntimeError(str(status.get("reason") or "invalid AIRI Native checkpoint"))
    cfg = NativeFoundationConfig.from_dict(status["config"])
    model = NativeFoundationLM(cfg)
    state = torch.load(root / NATIVE_MODEL_FILENAME, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg, status
