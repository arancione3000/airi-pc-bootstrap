from __future__ import annotations

import argparse
import copy
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .bootstrap_training import _segment_language_gate
from .phase5_diagnostics import evaluate_phase5_language
from .runtime import GeneralistRuntime


def _diversity(report: dict[str, Any]) -> dict[str, float]:
    outputs = [
        re.sub(r"\s+", " ", str(row.get("raw_output") or "")).strip().lower()
        for row in (report.get("traces") or [])
    ]
    unique = len(set(outputs))
    counts: dict[str, int] = {}
    for output in outputs:
        counts[output] = counts.get(output, 0) + 1
    pairs = 0
    jac = 0.0
    seq = 0.0
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            pairs += 1
            seq += SequenceMatcher(None, outputs[i], outputs[j], autojunk=False).ratio()
            a, b = set(outputs[i].split()), set(outputs[j].split())
            union = a | b
            jac += len(a & b) / len(union) if union else 1.0
    return {
        "exact_duplicate_rate": 1.0 - unique / len(outputs) if outputs else 0.0,
        "dominant_generation_fraction": max(counts.values(), default=0) / len(outputs) if outputs else 0.0,
        "mean_pairwise_token_jaccard": jac / pairs if pairs else 0.0,
        "mean_pairwise_sequence_similarity": seq / pairs if pairs else 0.0,
    }


def _project_residual(current, historical, *, source_d_ff: int, alpha: float, up: bool, down: bool) -> dict[str, Any]:
    import torch

    target_d_ff = int(current.config.d_ff)
    source = int(source_d_ff)
    alpha = float(alpha)
    touched = 0
    delta_sq = 0.0

    with torch.no_grad():
        for current_block, old_block in zip(current.model.blocks, historical.model.blocks):
            if down:
                now = current_block.ff.down.weight[:, source:target_d_ff]
                old = old_block.ff.down.weight[:, source:target_d_ff]
                before = now.clone()
                now.lerp_(old, alpha)
                delta = now - before
                delta_sq += float(torch.sum(delta.float() * delta.float()).item())
                touched += int(delta.numel())

            if up:
                now_up = current_block.ff.up.weight
                old_up = old_block.ff.up.weight
                slices = [
                    (slice(source, target_d_ff), slice(None)),
                    (slice(target_d_ff + source, 2 * target_d_ff), slice(None)),
                ]
                for row_slice, col_slice in slices:
                    now = now_up[row_slice, col_slice]
                    old = old_up[row_slice, col_slice]
                    before = now.clone()
                    now.lerp_(old, alpha)
                    delta = now - before
                    delta_sq += float(torch.sum(delta.float() * delta.float()).item())
                    touched += int(delta.numel())

                if current_block.ff.up.bias is not None:
                    now_bias = current_block.ff.up.bias
                    old_bias = old_block.ff.up.bias
                    for row_slice in (
                        slice(source, target_d_ff),
                        slice(target_d_ff + source, 2 * target_d_ff),
                    ):
                        now = now_bias[row_slice]
                        old = old_bias[row_slice]
                        before = now.clone()
                        now.lerp_(old, alpha)
                        delta = now - before
                        delta_sq += float(torch.sum(delta.float() * delta.float()).item())
                        touched += int(delta.numel())

    return {
        "alpha": alpha,
        "up": bool(up),
        "down": bool(down),
        "coordinates_touched": touched,
        "projection_delta_l2": delta_sq ** 0.5,
    }


def run(current_dir: str | Path, historical_dir: str | Path, guard_path: str | Path, *, source_d_ff: int) -> dict[str, Any]:
    current = GeneralistRuntime.from_checkpoint(Path(current_dir), device="cpu")
    historical = GeneralistRuntime.from_checkpoint(Path(historical_dir), device="cpu")
    if current.config.to_dict() != historical.config.to_dict():
        raise RuntimeError("historical/current architecture mismatch")

    guard = json.loads(Path(guard_path).read_text(encoding="utf-8"))
    anchor = dict(guard.get("report") or {})
    if not anchor:
        raise RuntimeError("missing durable guard anchor")

    before = evaluate_phase5_language(current)
    before_diversity = _diversity(before)
    ceiling = float(anchor["repetition_rate"]) + 0.08
    headroom_before = ceiling - float(before["repetition_rate"])

    variants = [
        ("DOWN_A05", 0.05, False, True),
        ("DOWN_A10", 0.10, False, True),
        ("UPDOWN_A05", 0.05, True, True),
        ("UPDOWN_A10", 0.10, True, True),
        ("UPDOWN_A20", 0.20, True, True),
        ("UPDOWN_A35", 0.35, True, True),
    ]
    results: list[dict[str, Any]] = []
    for name, alpha, up, down in variants:
        runtime = GeneralistRuntime(
            copy.deepcopy(current.model),
            current.config,
            tokenizer=current.tokenizer,
            device="cpu",
        )
        projection = _project_residual(
            runtime,
            historical,
            source_d_ff=source_d_ff,
            alpha=alpha,
            up=up,
            down=down,
        )
        after = evaluate_phase5_language(runtime)
        diversity = _diversity(after)
        gate_ok, gate = _segment_language_gate(
            before,
            after,
            anchor,
            attempted_tokens=0,
        )
        headroom_after = ceiling - float(after["repetition_rate"])
        nll_delta = float(after["language_nll"] - before["language_nll"])
        results.append({
            "name": name,
            "projection": projection,
            "guard_ok": bool(gate_ok),
            "rehabilitation_candidate": bool(
                gate_ok
                and headroom_after > headroom_before
                and nll_delta <= 0.015
                and diversity["exact_duplicate_rate"] <= before_diversity["exact_duplicate_rate"] + 1e-12
            ),
            "before": {
                "nll": before["language_nll"],
                "repetition": before["repetition_rate"],
                "headroom": headroom_before,
                "diversity": before_diversity,
            },
            "after": {
                "nll": after["language_nll"],
                "repetition": after["repetition_rate"],
                "headroom": headroom_after,
                "diversity": diversity,
            },
            "delta": {
                "nll": nll_delta,
                "repetition": float(after["repetition_rate"] - before["repetition_rate"]),
                "headroom": float(headroom_after - headroom_before),
            },
            "reasons": gate["local_reasons"] + gate["after_anchor_violations"],
        })
        del runtime

    return {
        "schema": 1,
        "version": "phase5-historical-residual-projection-ablation-v1",
        "read_only": True,
        "source_d_ff": int(source_d_ff),
        "target_d_ff": int(current.config.d_ff),
        "anchor_repetition": float(anchor["repetition_rate"]),
        "ceiling": ceiling,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--source-d-ff", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        args.current,
        args.historical,
        args.guard,
        source_d_ff=args.source_d_ff,
    )
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "variants": [
            {
                "name": row["name"],
                "guard_ok": row["guard_ok"],
                "candidate": row["rehabilitation_candidate"],
                "nll_delta": row["delta"]["nll"],
                "repetition_delta": row["delta"]["repetition"],
                "headroom_after": row["after"]["headroom"],
                "headroom_gain": row["delta"]["headroom"],
                "duplicate_rate_after": row["after"]["diversity"]["exact_duplicate_rate"],
                "reasons": row["reasons"],
            }
            for row in report["results"]
        ]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
