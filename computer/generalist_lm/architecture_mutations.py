from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

from .architecture_ir import ArchitectureSpec, architecture_id
from .evolution import progressive_scale_candidate


PARAMETER_TIERS = (
    250_000,
    500_000,
    1_250_000,
    3_000_000,
    7_000_000,
    12_000_000,
    20_000_000,
    32_000_000,
    50_000_000,
    64_000_000,
    80_000_000,
    96_000_000,
)


def next_parameter_tier(current: int, *, cap: int = 7_000_000) -> int | None:
    tiers = next_parameter_tiers(current, cap=cap, limit=1)
    return tiers[0] if tiers else None


def next_parameter_tiers(
    current: int,
    *,
    cap: int = 7_000_000,
    limit: int = 5,
) -> list[int]:
    out = [
        int(tier)
        for tier in PARAMETER_TIERS
        if int(current) < tier <= int(cap)
    ]
    return out[: max(0, int(limit))]


def _with_id(parent: ArchitectureSpec, updates: dict[str, Any]) -> ArchitectureSpec:
    payload = {**parent.to_dict(), **updates}
    payload["generation"] = int(parent.generation) + 1
    payload["parent_id"] = parent.architecture_id
    payload.pop("architecture_id", None)
    aid = architecture_id(
        parent.architecture_id,
        payload["generation"],
        payload,
    )
    return ArchitectureSpec(architecture_id=aid, **payload).validate()


def structural_mutations(
    parent: ArchitectureSpec,
    *,
    signals: list[str] | None = None,
) -> list[ArchitectureSpec]:
    parent.validate()
    signals = set(str(x) for x in (signals or []))
    rows: list[dict[str, Any]] = []

    rows.append({
        "norm_type": "rmsnorm" if parent.norm_type == "layernorm" else "layernorm",
    })
    rows.append({
        "position_encoding": (
            "rope"
            if parent.position_encoding != "rope"
            else "sinusoidal"
        ),
    })
    rows.append({
        "attention_type": "gqa" if parent.attention_type == "mha" else "mha",
        "n_kv_heads": (
            max(1, parent.n_heads // 2)
            if parent.attention_type == "mha"
            else parent.n_heads
        ),
    })
    rows.append({
        "norm_placement": "post" if parent.norm_placement == "pre" else "pre",
    })
    rows.append({
        "local_attention_window": (
            max(32, min(parent.context_length, parent.context_length // 2))
            if parent.local_attention_window == 0
            else 0
        ),
        "local_attention_every": (
            2
            if parent.local_attention_window == 0 and parent.n_layers >= 2
            else 0
        ),
    })
    rows.append({
        "tie_embeddings": not parent.tie_embeddings,
    })
    rows.append({
        "ff_variant": (
            "gelu" if parent.ff_variant == "swiglu" else "swiglu"
        ),
        "moe_experts": 1,
        "moe_top_k": 1,
    })
    rows.append({
        "recurrent_depth": (
            min(4, int(parent.recurrent_depth) + 1)
            if int(parent.recurrent_depth) < 4
            else 1
        ),
    })
    rows.append({
        "ff_variant": "moe_swiglu",
        "moe_experts": (
            min(8, max(2, int(parent.moe_experts) * 2))
            if parent.ff_variant == "moe_swiglu"
            else 4
        ),
        "moe_top_k": 1,
    })

    if "language_gap" in signals or "language_collapse" in signals or "autoregressive_collapse" in signals:
        rows.insert(0, {
            "norm_type": "rmsnorm",
            "norm_placement": "pre",
            "position_encoding": "rope",
            "ff_variant": "swiglu",
            "moe_experts": 1,
            "moe_top_k": 1,
        })
        rows.insert(1, {
            "attention_type": "gqa",
            "n_kv_heads": max(1, parent.n_heads // 2),
            "local_attention_window": (
                max(32, min(parent.context_length, parent.context_length // 2))
                if parent.n_layers >= 4
                else 0
            ),
            "local_attention_every": 2 if parent.n_layers >= 4 else 0,
        })
        rows.insert(2, {
            "norm_type": "rmsnorm",
            "norm_placement": "pre",
            "position_encoding": "rope",
            "ff_variant": "swiglu",
            "moe_experts": 1,
            "moe_top_k": 1,
            "attention_type": "gqa",
            "n_kv_heads": max(1, parent.n_heads // 2),
        })

    out: list[ArchitectureSpec] = []
    seen: set[str] = set()
    for updates in rows:
        try:
            candidate = _with_id(parent, updates)
        except Exception:
            continue
        signature = candidate.fingerprint()
        if signature == parent.fingerprint() or signature in seen:
            continue
        seen.add(signature)
        out.append(candidate)
    return out


def scaled_descendant(
    parent: ArchitectureSpec,
    *,
    target_parameters: int,
    max_width: int = 384,
    max_layers: int = 10,
) -> ArchitectureSpec:
    genome = progressive_scale_candidate(
        parent.to_genome(),
        target_parameters=int(target_parameters),
        vocab_size=int(parent.target_vocab_size),
        max_width=int(max_width),
        max_layers=int(max_layers),
        prefer_function_preserving=False,
    )
    spec = ArchitectureSpec.from_genome(
        genome,
        target_vocab_size=parent.target_vocab_size,
    )
    restored = replace(
        spec,
        architecture_id="pending",
        attention_type=parent.attention_type,
        n_kv_heads=parent.n_kv_heads,
        local_attention_window=parent.local_attention_window,
        local_attention_every=parent.local_attention_every,
        norm_placement=parent.norm_placement,
        tie_embeddings=parent.tie_embeddings,
        norm_type=parent.norm_type,
        position_encoding=parent.position_encoding,
        ff_variant=parent.ff_variant,
        recurrent_depth=parent.recurrent_depth,
        moe_experts=parent.moe_experts,
        moe_top_k=parent.moe_top_k,
    )
    return replace(
        restored,
        architecture_id=architecture_id(
            parent.architecture_id,
            restored.generation,
            restored.canonical_payload(),
        ),
    ).validate()


def proposal_set(
    parent: ArchitectureSpec,
    *,
    signals: list[str] | None = None,
    parameter_cap: int = 7_000_000,
    max_candidates: int = 12,
    max_width: int = 384,
    max_layers: int = 10,
) -> list[dict[str, Any]]:
    parent.validate()
    proposals: list[tuple[str, ArchitectureSpec, str]] = []
    signals = list(signals or [])

    for spec in structural_mutations(parent, signals=signals):
        proposals.append((
            "structural_mutation",
            spec,
            "bounded Architecture-IR mutation prompted by observed weaknesses",
        ))

    targets = next_parameter_tiers(
        parent.parameter_estimate(vocab_size=parent.target_vocab_size),
        cap=int(parameter_cap),
        limit=5,
    )
    modern = _with_id(parent, {
        "norm_type": "rmsnorm",
        "norm_placement": "pre",
        "position_encoding": "rope",
        "ff_variant": "swiglu",
        "moe_experts": 1,
        "moe_top_k": 1,
    })
    for target in targets:
        try:
            scaled = scaled_descendant(
                parent,
                target_parameters=target,
                max_width=max_width,
                max_layers=max_layers,
            )
            proposals.append((
                "capacity_scale",
                scaled,
                f"bounded parameter tier {target}",
            ))
        except Exception:
            pass

        # Couple each bounded scale probe with a conservative modernized
        # baseline. If the nearest tier has already failed under the same
        # parent, negative-memory filtering can advance to the next tier
        # instead of rediscovering the same topology forever.
        try:
            scaled_modern = scaled_descendant(
                modern,
                target_parameters=target,
                max_width=max_width,
                max_layers=max_layers,
            )
            proposals.append((
                "capacity_plus_structure",
                scaled_modern,
                f"scale to {target} after a conservative RMSNorm/RoPE/SwiGLU modernization",
            ))
        except Exception:
            pass

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, spec, hypothesis in proposals:
        key = spec.fingerprint()
        if key in seen:
            continue
        seen.add(key)
        unique.append({
            "kind": kind,
            "hypothesis": hypothesis,
            "architecture": spec.to_dict(),
            "fingerprint": key,
            "parameter_estimate": spec.parameter_estimate(),
            "falsification": [
                "fails static/gradient/KV-cache verifier",
                "autoregressive repetition worsens materially",
                "held-out language NLL or generalist gates regress beyond policy",
                "compute/memory cost is dominated by a Pareto-superior candidate",
            ],
        })
        if len(unique) >= max(1, int(max_candidates)):
            break
    return unique
