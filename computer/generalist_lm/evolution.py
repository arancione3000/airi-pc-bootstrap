from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .model import GeneralistLMConfig, estimate_flops_per_token


_ALLOWED_TOKENIZERS = {"byte-v1"}
_ALLOWED_TOOL_PROTOCOLS = {"tool-json-v1"}


@dataclass(frozen=True)
class GeneralistGenome:
    generation: int = 0
    parent_id: str | None = None
    genome_id: str = "generalist-0-seed"
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.0
    learning_rate: float = 3e-4
    tokenizer_version: str = "byte-v1"
    tool_protocol: str = "tool-json-v1"
    retrieval_adapter: bool = True
    symbolic_adapter: bool = True
    code_adapter: bool = True
    data_adapter: bool = True
    reasoning_depth: int = 2
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    ff_variant: str = "swiglu"

    def validate(self) -> "GeneralistGenome":
        if self.tokenizer_version not in _ALLOWED_TOKENIZERS:
            raise ValueError("unauthorized tokenizer")
        if self.tool_protocol not in _ALLOWED_TOOL_PROTOCOLS:
            raise ValueError("unauthorized tool protocol")
        if not (64 <= self.context_length <= 8192):
            raise ValueError("context_length out of bounded DSL")
        if not (32 <= self.d_model <= 4096):
            raise ValueError("d_model out of bounded DSL")
        if not (1 <= self.n_heads <= 64 and self.d_model % self.n_heads == 0):
            raise ValueError("invalid attention heads")
        if not (1 <= self.n_layers <= 96):
            raise ValueError("n_layers out of bounded DSL")
        if not (self.d_model <= self.d_ff <= 16384):
            raise ValueError("d_ff out of bounded DSL")
        if not (0.0 <= self.dropout <= 0.5):
            raise ValueError("dropout out of bounded DSL")
        if not (1e-6 <= self.learning_rate <= 1e-2):
            raise ValueError("learning_rate out of bounded DSL")
        if not (1 <= self.reasoning_depth <= 16):
            raise ValueError("reasoning_depth out of bounded DSL")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("unauthorized norm_type")
        if self.position_encoding not in {"learned", "sinusoidal"}:
            raise ValueError("unauthorized position_encoding")
        if self.ff_variant not in {"swiglu", "gelu"}:
            raise ValueError("unauthorized ff_variant")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_config(self, vocab_size: int = 264) -> GeneralistLMConfig:
        self.validate()
        return GeneralistLMConfig(
            vocab_size=vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            tokenizer_version=self.tokenizer_version,
            norm_type=self.norm_type,
            position_encoding=self.position_encoding,
            ff_variant=self.ff_variant,
        ).validate()


def _id(parent: GeneralistGenome, index: int, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    tag = hashlib.sha256(raw).hexdigest()[:8]
    return f"generalist-{parent.generation + 1}-{index}-{tag}"


def generate_challengers(
    champion: GeneralistGenome,
    *,
    signals: list[str] | None = None,
    count: int = 4,
    exploration_offset: int = 0,
) -> list[GeneralistGenome]:
    champion.validate()
    signals = list(signals or [])
    variants = [
        {"norm_type": "rmsnorm" if champion.norm_type == "layernorm" else "layernorm"},
        {"position_encoding": "sinusoidal" if champion.position_encoding == "learned" else "learned"},
        {"ff_variant": "gelu" if champion.ff_variant == "swiglu" else "swiglu"},
        {"d_model": min(4096, champion.d_model + 32), "d_ff": min(16384, champion.d_ff + 96)},
        {"n_layers": min(96, champion.n_layers + 1)},
        {"context_length": min(8192, champion.context_length * 2)},
        {"reasoning_depth": min(16, champion.reasoning_depth + 1)},
        {"d_model": max(32, champion.d_model - 32), "d_ff": max(champion.d_model, champion.d_ff - 64)},
        {"dropout": min(0.5, champion.dropout + 0.05)},
    ]
    if variants:
        shift = int(exploration_offset) % len(variants)
        variants = variants[shift:] + variants[:shift]

    directed: list[dict[str, Any]] = []
    if "coding_gap" in signals:
        directed.append({"code_adapter": True, "reasoning_depth": min(16, champion.reasoning_depth + 1)})
    if "data_gap" in signals:
        directed.append({"data_adapter": True, "context_length": min(8192, champion.context_length * 2)})
    if "tool_gap" in signals:
        directed.append({"retrieval_adapter": True})
    if "symbolic_reasoning_signal" in signals:
        directed.append({"symbolic_adapter": True, "reasoning_depth": min(16, champion.reasoning_depth + 1)})
    variants = directed + variants

    out: list[GeneralistGenome] = []
    seen: set[str] = set()
    for variant in variants:
        if len(out) >= max(1, int(count)):
            break
        payload = {**champion.to_dict(), **variant}
        payload["generation"] = champion.generation + 1
        payload["parent_id"] = champion.genome_id
        payload["genome_id"] = _id(champion, len(out), payload)
        candidate = GeneralistGenome(**payload).validate()
        signature = json.dumps(
            {k: v for k, v in candidate.to_dict().items() if k not in {"genome_id", "parent_id", "generation"}},
            sort_keys=True,
        )
        if signature in seen:
            continue
        seen.add(signature)
        out.append(candidate)
    return out


def evolution_cost(genome: GeneralistGenome) -> dict[str, Any]:
    cfg = genome.model_config()
    return {
        "flops_per_token_estimate": estimate_flops_per_token(cfg),
        "context_length": cfg.context_length,
        "layers": cfg.n_layers,
        "width": cfg.d_model,
        "norm_type": cfg.norm_type,
        "position_encoding": cfg.position_encoding,
        "ff_variant": cfg.ff_variant,
    }


def promotion_decision(
    champion_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    minimum_gain: float = 2.0,
    max_domain_regression: float = 0.0,
) -> tuple[bool, str]:
    if candidate_report.get("critical_failures"):
        return False, "candidate has critical generalist failures"
    champion_domains = champion_report.get("domain_scores") or {}
    candidate_domains = candidate_report.get("domain_scores") or {}
    for domain, old_score in champion_domains.items():
        if domain not in candidate_domains:
            return False, f"candidate lost benchmark domain: {domain}"
        if float(candidate_domains[domain]) < float(old_score) - float(max_domain_regression):
            return False, f"candidate regressed in domain: {domain}"
    gain = float(candidate_report.get("score", 0.0)) - float(champion_report.get("score", 0.0))
    if gain < float(minimum_gain):
        return False, "candidate did not beat champion by the generalist promotion margin"
    return True, "generalist capability gain passed"


def weakness_signals(report: dict[str, Any]) -> list[str]:
    scores = report.get("domain_scores") or {}
    signals: list[str] = []
    if float(scores.get("coding", 0.0)) < 1.0:
        signals.append("coding_gap")
    if float(scores.get("data", 0.0)) < 1.0:
        signals.append("data_gap")
    if float(scores.get("tools", 0.0)) < 1.0:
        signals.append("tool_gap")
    if float(scores.get("reasoning", 0.0)) < 1.0:
        signals.append("reasoning_gap")
    return signals
