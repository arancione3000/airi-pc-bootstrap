from __future__ import annotations

import math
from typing import Any

from .model import estimate_flops_per_token, estimate_parameter_count

ISLANDS = ("language", "coding", "reasoning", "tools", "efficiency")
_DOMAIN_ALIASES = {
    "structured": "tools",
    "data": "reasoning",
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if math.isfinite(number) else float(default)


def active_learning_weights(
    report: dict[str, Any] | None,
    *,
    base_weights: dict[str, float] | None = None,
    verified_tool_experiences: int = 0,
) -> dict[str, float]:
    """Allocate scarce training budget toward measured weaknesses.

    Weights are bounded so active learning cannot erase replay from strong
    domains. Tool-use receives an extra boost while the system has no verified
    tool experience, because syntactically invalid tool calls are currently a
    measured bottleneck.
    """
    report = report or {}
    base = {str(k): max(1.0, min(3.0, _finite(v, 1.0))) for k, v in (base_weights or {}).items()}
    domains = report.get("domain_nll_per_byte") or report.get("domain_loss") or {}
    clean = {
        str(domain): _finite(value)
        for domain, value in domains.items()
        if math.isfinite(_finite(value, float("nan")))
    }
    if clean:
        best = min(clean.values())
        worst = max(clean.values())
        span = max(1e-9, worst - best)
        for domain, value in clean.items():
            weakness = (value - best) / span
            base[domain] = max(base.get(domain, 1.0), 1.0 + 3.0 * weakness)

    generation = report.get("domain_generation_similarity") or {}
    for domain, value in generation.items():
        similarity = max(0.0, min(1.0, _finite(value)))
        base[str(domain)] = max(base.get(str(domain), 1.0), 1.0 + 2.0 * (1.0 - similarity))

    if int(verified_tool_experiences) <= 0:
        base["tools"] = max(base.get("tools", 1.0), 4.0)
        base["structured"] = max(base.get("structured", 1.0), 2.5)

    return {
        domain: min(4.0, max(1.0, float(weight)))
        for domain, weight in sorted(base.items())
    }


def island_schedule(
    count: int,
    *,
    signals: list[str] | None = None,
) -> list[str]:
    """Create deterministic specialist lanes while prioritizing current gaps."""
    signals = set(str(item) for item in (signals or []))
    priority: list[str] = []
    if {"language_gap", "language_collapse", "autoregressive_collapse"} & signals:
        priority.extend(["language", "language"])
    if {"tool_gap"} & signals:
        priority.extend(["tools", "tools"])
    if {"coding_gap"} & signals:
        priority.append("coding")
    if {"reasoning_gap", "symbolic_reasoning_signal", "deep_symbolic_signal"} & signals:
        priority.append("reasoning")
    priority.append("efficiency")
    priority.extend(ISLANDS)

    out: list[str] = []
    index = 0
    while len(out) < max(0, int(count)):
        out.append(priority[index % len(priority)])
        index += 1
    return out


def island_weights(
    base_weights: dict[str, float] | None,
    island: str,
) -> dict[str, float]:
    """Bias one candidate toward a specialist lane without dropping replay."""
    weights = {
        str(domain): min(4.0, max(1.0, _finite(value, 1.0)))
        for domain, value in (base_weights or {}).items()
    }
    island = str(island or "efficiency")
    focus = {
        "language": ("language",),
        "coding": ("coding",),
        "reasoning": ("reasoning", "data"),
        "tools": ("tools", "structured"),
        "efficiency": (),
    }.get(island, ())
    for domain in focus:
        weights[domain] = max(weights.get(domain, 1.0), 4.0)
    return dict(sorted(weights.items()))


def efficiency_profile(config, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report capability per resource rather than parameter count alone."""
    report = report or {}
    params = max(1, int(estimate_parameter_count(config)))
    flops = max(1, int(estimate_flops_per_token(config)))
    score = _finite(report.get("score"), 0.0)
    generation_similarity = max(0.0, min(1.0, _finite(report.get("generation_similarity"), 0.0)))
    nll = max(0.0, _finite(report.get("nll_per_byte", report.get("loss")), 1_000.0))
    quality_proxy = max(0.0, score) + 10.0 * generation_similarity + 10.0 / (1.0 + nll)
    return {
        "parameters": params,
        "flops_per_token_estimate": flops,
        "quality_proxy": float(quality_proxy),
        "quality_per_million_parameters": float(quality_proxy * 1_000_000.0 / params),
        "quality_per_million_flops": float(quality_proxy * 1_000_000.0 / flops),
        "active_compute_fraction": 1.0,
    }


def efficiency_bonus(profile: dict[str, Any]) -> float:
    """Small bounded selection bonus; quality gates remain authoritative."""
    per_flop = max(0.0, _finite(profile.get("quality_per_million_flops"), 0.0))
    per_param = max(0.0, _finite(profile.get("quality_per_million_parameters"), 0.0))
    return float(min(2.0, 0.35 * math.log1p(per_flop) + 0.15 * math.log1p(per_param)))


def sparse_expert_plan(
    *,
    available_islands: list[str] | tuple[str, ...] = ISLANDS,
) -> dict[str, Any]:
    """System-level sparse experts: load/activate one specialist per request.

    This intentionally avoids a dense in-model MoE on CPU/Android. Specialist
    checkpoints can be trained in parallel and only the selected expert needs
    inference compute.
    """
    available = [item for item in ISLANDS if item in set(available_islands)]
    routes = {
        "conversation": "language",
        "language": "language",
        "coding": "coding",
        "data": "reasoning",
        "reasoning": "reasoning",
        "research": "tools",
        "tools": "tools",
        "structured": "tools",
        "default": "efficiency",
    }
    return {
        "mode": "system_sparse_experts",
        "available_islands": available,
        "routes": {key: value for key, value in routes.items() if value in available},
        "max_active_experts": 1,
        "dense_ensemble": False,
        "mobile_safe": True,
    }


def self_play_policy(
    report: dict[str, Any] | None,
    *,
    verified_tool_experiences: int = 0,
) -> dict[str, Any]:
    """Fail closed until the model is competent enough to propose training tasks."""
    report = report or {}
    similarity = _finite(report.get("generation_similarity"), 0.0)
    repetition = _finite(report.get("generation_repetition_rate"), 1.0)
    exact = _finite(report.get("generation_exact_accuracy"), 0.0)
    enabled = bool(
        similarity >= 0.45
        and repetition <= 0.45
        and exact >= 0.10
        and int(verified_tool_experiences) >= 3
        and not bool(report.get("generation_pathological_repetition", True))
    )
    return {
        "enabled": enabled,
        "mode": "verified_only",
        "maximum_generated_tasks_per_cycle": 32 if enabled else 0,
        "requires_external_verifier": True,
        "requires_training_exclusion_for_verifier_cases": True,
        "reason": (
            "generation/tool-use competence passed verified self-play readiness gate"
            if enabled
            else "self-play disabled until language and tool-use readiness gates pass"
        ),
    }
