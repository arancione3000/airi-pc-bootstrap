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
)


def next_parameter_tier(current: int, *, cap: int = 7_000_000) -> int | None:
    for tier in PARAMETER_TIERS:
        if int(current) < tier <= int(cap):
            return int(tier)
    return None


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
        "ff_variant": "gelu" if parent.ff_variant == "swiglu" else "swiglu",
    })

    if "language_gap" in signals or "language_collapse" in signals or "autoregressive_collapse" in signals:
        rows.insert(0, {
            "norm_type": "rmsnorm",
            "norm_placement": "pre",
            "position_encoding": "rope",
            "ff_variant": "swiglu",
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
    return replace(
        spec,
        architecture_id=architecture_id(
            parent.architecture_id,
            spec.generation,
            spec.canonical_payload(),
        ),
        attention_type=parent.attention_type,
        n_kv_heads=parent.n_kv_heads,
        local_attention_window=parent.local_attention_window,
        local_attention_every=parent.local_attention_every,
        norm_placement=parent.norm_placement,
        tie_embeddings=parent.tie_embeddings,
        norm_type=parent.norm_type,
        position_encoding=parent.position_encoding,
        ff_variant=parent.ff_variant,
    ).validate()


def proposal_set(
    parent: ArchitectureSpec,
    *,
    signals: list[str] | None = None,
    parameter_cap: int = 7_000_000,
    max_candidates: int = 12,
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

    target = next_parameter_tier(
        parent.parameter_estimate(vocab_size=parent.target_vocab_size),
        cap=int(parameter_cap),
    )
    if target is not None:
        try:
            scaled = scaled_descendant(
                parent,
                target_parameters=target,
            )
            proposals.append((
                "capacity_scale",
                scaled,
                f"next bounded parameter tier {target}",
            ))
        except Exception:
            pass

        # Couple one scale probe with the modernized baseline rather than
        # assuming the current architecture is the best shape to enlarge.
        modern = _with_id(parent, {
            "norm_type": "rmsnorm",
            "norm_placement": "pre",
            "position_encoding": "rope",
            "ff_variant": "swiglu",
        })
        try:
            scaled_modern = scaled_descendant(
                modern,
                target_parameters=target,
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
