from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .curriculum import validation_rows
from .phase5_diagnostics import evaluate_phase5_language
from .research_cycle import _grouped_validation, _research_score
from .runtime import GeneralistRuntime


def _summary(checkpoint: Path) -> dict[str, Any]:
    runtime = GeneralistRuntime.from_checkpoint(checkpoint, device="cpu")
    phase5 = evaluate_phase5_language(runtime)
    validation = _grouped_validation(
        runtime.model,
        runtime.tokenizer,
        validation_rows(),
        device="cpu",
    )
    parameters = sum(
        int(parameter.numel())
        for parameter in runtime.model.parameters()
        if parameter.requires_grad
    )
    validation["parameters"] = parameters
    validation["score"] = _research_score(validation, parameters)
    return {
        "checkpoint": checkpoint.name,
        "parameters": parameters,
        "tokenizer_version": runtime.tokenizer.version,
        "tokenizer_vocab_size": int(runtime.tokenizer.vocab_size),
        "phase5": phase5,
        "validation": validation,
    }


def audit_bootstrap_checkpoints(
    state_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(state_dir).expanduser().resolve()
    bootstrap = root / "bootstrap-data"
    candidate_path = bootstrap / "candidate"
    if not (candidate_path / "model.pt").is_file():
        raise RuntimeError("missing active Phase 5 candidate checkpoint")

    reports = {
        "post_sft_candidate": _summary(candidate_path),
    }
    best_path = bootstrap / "best"
    if (best_path / "model.pt").is_file():
        reports["causal_best"] = _summary(best_path)

    candidate = reports["post_sft_candidate"]["phase5"]
    best_report = reports.get("causal_best")
    if best_report is None:
        comparison = {
            "available": False,
            "reason": (
                "causal_best checkpoint unavailable after structural capacity "
                "growth; active candidate audited independently"
            ),
        }
    elif (
        int(best_report.get("parameters", 0) or 0)
        != int(reports["post_sft_candidate"].get("parameters", 0) or 0)
    ):
        comparison = {
            "available": False,
            "reason": (
                "causal_best and active candidate have different capacities; "
                "direct post-SFT deltas would be misleading"
            ),
            "causal_best_parameters": int(best_report.get("parameters", 0) or 0),
            "candidate_parameters": int(
                reports["post_sft_candidate"].get("parameters", 0) or 0
            ),
        }
    else:
        best = best_report["phase5"]
        comparison = {
            "available": True,
            "language_nll_delta_post_sft_minus_causal": (
                float(candidate.get("language_nll", 0.0))
                - float(best.get("language_nll", 0.0))
            ),
            "repetition_delta_post_sft_minus_causal": (
                float(candidate.get("repetition_rate", 0.0))
                - float(best.get("repetition_rate", 0.0))
            ),
            "similarity_delta_post_sft_minus_causal": (
                float(candidate.get("generation_similarity", 0.0))
                - float(best.get("generation_similarity", 0.0))
            ),
            "post_sft_introduced_pathological_repetition": (
                not bool(best.get("pathological_repetition"))
                and bool(candidate.get("pathological_repetition"))
            ),
        }
    result = {
        "schema": 1,
        "version": "phase5-checkpoint-audit-v2",
        "training_data_used": False,
        "held_out_suite_used_for_diagnosis_only": True,
        "reports": reports,
        "comparison": comparison,
        "active_candidate_audited": True,
        "causal_best_available": "causal_best" in reports,
    }
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose causal-best vs post-SFT Phase 5 checkpoints"
    )
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = audit_bootstrap_checkpoints(
        args.state_dir,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
