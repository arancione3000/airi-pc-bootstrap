from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .architecture_ir import ArchitectureSpec
from .model_factory import build_architecture


@dataclass(frozen=True)
class ArchitectureVerification:
    ok: bool
    report: dict[str, Any]


def verify_architecture(
    spec: ArchitectureSpec,
    *,
    vocab_size: int | None = None,
    batch_size: int = 2,
    sequence_length: int = 24,
    max_parameters: int = 8_000_000,
) -> ArchitectureVerification:
    import torch

    failures: list[str] = []
    try:
        spec.validate()
    except Exception as exc:
        return ArchitectureVerification(False, {
            "ok": False,
            "failures": [f"IR validation: {type(exc).__name__}:{exc}"],
        })

    try:
        built = build_architecture(spec, vocab_size=vocab_size)
    except Exception as exc:
        return ArchitectureVerification(False, {
            "ok": False,
            "failures": [f"factory: {type(exc).__name__}:{exc}"],
        })

    if built.parameter_count > int(max_parameters):
        failures.append(
            f"parameter budget exceeded: {built.parameter_count}>{int(max_parameters)}"
        )

    model = built.model
    cfg = model.config
    length = max(4, min(int(sequence_length), int(cfg.context_length) - 1))
    torch.manual_seed(7341)
    ids = torch.randint(
        low=0,
        high=int(cfg.vocab_size),
        size=(max(1, int(batch_size)), length),
        dtype=torch.long,
    )
    labels = ids.clone()

    forward_shape = None
    loss_value = None
    gradient_norm = None
    cache_equivalent = False

    try:
        result = model(ids, labels=labels)
        logits = result["logits"]
        forward_shape = list(logits.shape)
        loss = result["loss"]
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            failures.append("non-finite forward loss")
        loss.backward()
        norms = []
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                failures.append("non-finite gradient")
                break
            norms.append(float(parameter.grad.detach().float().norm()))
        gradient_norm = float(sum(norms))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            failures.append("missing or invalid gradient signal")
    except Exception as exc:
        failures.append(f"forward/backward: {type(exc).__name__}:{exc}")

    try:
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            full = model(ids)["logits"]
            prefix = ids[:, :-1]
            tail = ids[:, -1:]
            first = model(prefix, use_cache=True)
            cached = model(
                tail,
                past_key_values=first["past_key_values"],
                use_cache=True,
            )["logits"][:, -1, :]
            reference = full[:, -1, :]
            cache_equivalent = bool(
                torch.allclose(reference, cached, atol=2e-4, rtol=2e-4)
            )
        if not cache_equivalent:
            failures.append("KV-cache decoding does not match full-sequence logits")
    except Exception as exc:
        failures.append(f"KV-cache: {type(exc).__name__}:{exc}")

    report = {
        "ok": not failures,
        "architecture_id": spec.architecture_id,
        "fingerprint": spec.fingerprint(),
        "parameters": int(built.parameter_count),
        "flops_per_token_estimate": int(built.flops_per_token_estimate),
        "forward_shape": forward_shape,
        "loss": loss_value,
        "gradient_norm_sum": gradient_norm,
        "kv_cache_equivalent": cache_equivalent,
        "failures": failures,
        "architecture": spec.to_dict(),
        "external_pretrained_weights": False,
        "arbitrary_code_execution": False,
    }
    return ArchitectureVerification(not failures, report)
