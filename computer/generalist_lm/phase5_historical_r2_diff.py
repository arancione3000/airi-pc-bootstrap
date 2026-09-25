from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .phase5_diagnostics import evaluate_phase5_language
from .runtime import GeneralistRuntime


def _sq(value) -> float:
    import torch
    v = value.detach().float()
    return float(torch.sum(v * v).item())


def _stats(delta_sq: float, base_sq: float, coordinates: int) -> dict[str, Any]:
    delta = math.sqrt(max(0.0, delta_sq))
    base = math.sqrt(max(0.0, base_sq))
    return {
        "delta_l2": delta,
        "reference_l2": base,
        "delta_to_reference": delta / max(1e-12, base),
        "coordinates": int(coordinates),
    }


def compare(current_dir: str | Path, historical_dir: str | Path, *, source_d_ff: int) -> dict[str, Any]:
    import torch

    current = GeneralistRuntime.from_checkpoint(Path(current_dir), device="cpu")
    historical = GeneralistRuntime.from_checkpoint(Path(historical_dir), device="cpu")
    if current.config.to_dict() != historical.config.to_dict():
        raise RuntimeError("historical/current architecture mismatch")

    current_state = current.model.state_dict()
    old_state = historical.model.state_dict()
    if tuple(current_state) != tuple(old_state):
        raise RuntimeError("state layout mismatch")

    d_ff = int(current.config.d_ff)
    source = int(source_d_ff)
    groups: dict[str, dict[str, float | int]] = {}

    def add(group: str, now, old) -> None:
        delta = now.detach().float() - old.detach().float()
        row = groups.setdefault(group, {"delta_sq": 0.0, "base_sq": 0.0, "coordinates": 0})
        row["delta_sq"] = float(row["delta_sq"]) + _sq(delta)
        row["base_sq"] = float(row["base_sq"]) + _sq(old)
        row["coordinates"] = int(row["coordinates"]) + int(delta.numel())

    for name, now in current_state.items():
        old = old_state[name]
        if ".ff.up.weight" in name:
            add("ffn_up_legacy_gate", now[:source], old[:source])
            add("ffn_up_residual_gate", now[source:d_ff], old[source:d_ff])
            add("ffn_up_legacy_value", now[d_ff:d_ff+source], old[d_ff:d_ff+source])
            add("ffn_up_residual_value", now[d_ff+source:2*d_ff], old[d_ff+source:2*d_ff])
        elif ".ff.up.bias" in name and now.ndim == 1:
            add("ffn_up_bias_legacy_gate", now[:source], old[:source])
            add("ffn_up_bias_residual_gate", now[source:d_ff], old[source:d_ff])
            add("ffn_up_bias_legacy_value", now[d_ff:d_ff+source], old[d_ff:d_ff+source])
            add("ffn_up_bias_residual_value", now[d_ff+source:2*d_ff], old[d_ff+source:2*d_ff])
        elif ".ff.down.weight" in name:
            add("ffn_down_legacy", now[:, :source], old[:, :source])
            add("ffn_down_residual", now[:, source:d_ff], old[:, source:d_ff])
        elif ".ff." in name:
            add("ffn_other", now, old)
        elif name.startswith("lm_head."):
            add("lm_head", now, old)
        elif "token_embedding" in name or "position_embedding" in name:
            add("embeddings", now, old)
        else:
            add("attention_and_norm", now, old)

    summarized = {
        name: _stats(float(row["delta_sq"]), float(row["base_sq"]), int(row["coordinates"]))
        for name, row in groups.items()
    }

    return {
        "schema": 1,
        "version": "phase5-r2-historical-diff-v1",
        "source_d_ff": source,
        "target_d_ff": d_ff,
        "current_language": evaluate_phase5_language(current),
        "historical_language_current_suite": evaluate_phase5_language(historical),
        "parameter_drift": summarized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--source-d-ff", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.current, args.historical, source_d_ff=args.source_d_ff)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "source_d_ff": report["source_d_ff"],
        "target_d_ff": report["target_d_ff"],
        "current": {
            "nll": report["current_language"]["language_nll"],
            "repetition": report["current_language"]["repetition_rate"],
        },
        "historical": {
            "nll": report["historical_language_current_suite"]["language_nll"],
            "repetition": report["historical_language_current_suite"]["repetition_rate"],
        },
        "parameter_drift": report["parameter_drift"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
