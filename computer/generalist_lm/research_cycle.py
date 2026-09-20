from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from .curriculum import ResearchRow, train_rows, validation_rows
from .evolution import GeneralistGenome, generate_challengers
from .mathesis_bridge import mathesis_signals
from .model import CausalTransformerLM, parameter_count
from .runtime import GeneralistRuntime
from .tokenizer import ByteTokenizer
from .training import loss_on_examples, train_sft


def research_seed() -> GeneralistGenome:
    return GeneralistGenome(
        generation=0,
        parent_id=None,
        genome_id="generalist-research-0-seed",
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        learning_rate=3e-3,
        reasoning_depth=2,
    ).validate()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _grouped_validation(model, tokenizer: ByteTokenizer, rows: list[ResearchRow], *, device: str = "cpu") -> dict[str, Any]:
    all_examples = [row.sft() for row in rows]
    overall = loss_on_examples(model, tokenizer, all_examples, device=device)
    domains: dict[str, float] = {}
    for domain in sorted({row.domain for row in rows}):
        examples = [row.sft() for row in rows if row.domain == domain]
        domains[domain] = loss_on_examples(model, tokenizer, examples, device=device)
    return {
        "loss": float(overall),
        "domain_loss": domains,
        "finite": bool(math.isfinite(overall) and all(math.isfinite(v) for v in domains.values())),
    }


def _research_score(report: dict[str, Any], params: int) -> float:
    loss = float(report["loss"])
    efficiency_penalty = min(1.0, params / 10_000_000.0) * 0.01
    return float(100.0 / (1.0 + loss) - efficiency_penalty)


def _research_eligible(
    champion: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_loss_gain: float,
    max_domain_regression: float,
) -> tuple[bool, str]:
    if not candidate.get("finite"):
        return False, "candidate validation is non-finite"
    old_loss = float(champion["loss"])
    new_loss = float(candidate["loss"])
    if old_loss - new_loss < float(minimum_loss_gain):
        return False, "candidate did not reduce held-out loss by the research margin"
    for domain, old_value in (champion.get("domain_loss") or {}).items():
        if domain not in (candidate.get("domain_loss") or {}):
            return False, f"candidate lost validation domain: {domain}"
        if float(candidate["domain_loss"][domain]) > float(old_value) + float(max_domain_regression):
            return False, f"candidate regressed in validation domain: {domain}"
    return True, "held-out multi-domain loss improvement passed"


def _weaknesses(report: dict[str, Any]) -> list[str]:
    domains = report.get("domain_loss") or {}
    if not domains:
        return []
    ordered = sorted(domains.items(), key=lambda item: float(item[1]), reverse=True)
    signals: list[str] = []
    for domain, _loss in ordered[:3]:
        if domain == "coding":
            signals.append("coding_gap")
        elif domain == "data":
            signals.append("data_gap")
        elif domain == "tools":
            signals.append("tool_gap")
        elif domain == "reasoning":
            signals.append("reasoning_gap")
    return signals


def _champion_paths(root: Path) -> tuple[Path, Path]:
    return root / "champion", root / "champion-genome.json"


def _load_champion(root: Path, *, device: str) -> tuple[GeneralistGenome, GeneralistRuntime] | None:
    checkpoint, genome_path = _champion_paths(root)
    try:
        genome_raw = json.loads(genome_path.read_text(encoding="utf-8"))
        genome = GeneralistGenome(**genome_raw).validate()
        runtime = GeneralistRuntime.from_checkpoint(checkpoint, device=device)
        return genome, runtime
    except Exception:
        return None


def _save_champion(root: Path, genome: GeneralistGenome, runtime: GeneralistRuntime, metrics: dict[str, Any]) -> None:
    checkpoint, genome_path = _champion_paths(root)
    tmp = root / ".champion-next"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    runtime.save_checkpoint(tmp, metadata={
        "role": "research_champion",
        "genome_id": genome.genome_id,
        "generation": genome.generation,
        "production_qualified": False,
    })
    _atomic_json(tmp / "research-metrics.json", metrics)
    backup = root / ".champion-old"
    shutil.rmtree(backup, ignore_errors=True)
    if checkpoint.exists():
        checkpoint.replace(backup)
    tmp.replace(checkpoint)
    shutil.rmtree(backup, ignore_errors=True)
    _atomic_json(genome_path, genome.to_dict())


def _train_genome(
    genome: GeneralistGenome,
    *,
    steps: int,
    seed: int,
    device: str,
) -> tuple[GeneralistRuntime, dict[str, Any]]:
    tokenizer = ByteTokenizer()
    model = CausalTransformerLM(genome.model_config(tokenizer.vocab_size))
    report = train_sft(
        model,
        tokenizer,
        [row.sft() for row in train_rows()],
        steps=steps,
        batch_size=4,
        learning_rate=genome.learning_rate,
        weight_decay=0.0,
        seed=seed,
        device=device,
    )
    runtime = GeneralistRuntime(model, genome.model_config(tokenizer.vocab_size), tokenizer, device=device)
    validation = _grouped_validation(runtime.model, tokenizer, validation_rows(), device=device)
    validation["parameters"] = parameter_count(runtime.model)
    validation["score"] = _research_score(validation, validation["parameters"])
    validation["training"] = report
    return runtime, validation


def run_research_cycle(state_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(state_dir or os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("AIRI_GENERALIST_RESEARCH_DEVICE", "cpu")
    steps = max(2, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_STEPS", "20")), 500))
    challenger_count = max(1, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "2")), 6))
    minimum_loss_gain = max(0.0, float(os.environ.get("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "0.02")))
    max_domain_regression = max(0.0, float(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_DOMAIN_REGRESSION", "0.10")))

    previous = {}
    try:
        previous = json.loads((root / "status.json").read_text(encoding="utf-8"))
    except Exception:
        previous = {}
    cycle = int(previous.get("cycle", 0) or 0) + 1

    loaded = _load_champion(root, device=device)
    if loaded is None:
        champion_genome = research_seed()
        champion_runtime, champion_report = _train_genome(
            champion_genome, steps=max(steps, 30), seed=1000, device=device
        )
        _save_champion(root, champion_genome, champion_runtime, champion_report)
        bootstrapped = True
    else:
        champion_genome, champion_runtime = loaded
        champion_report = _grouped_validation(
            champion_runtime.model, champion_runtime.tokenizer, validation_rows(), device=device
        )
        champion_report["parameters"] = parameter_count(champion_runtime.model)
        champion_report["score"] = _research_score(champion_report, champion_report["parameters"])
        bootstrapped = False

    signals = _weaknesses(champion_report)
    mathesis = None
    mathesis_dir = os.environ.get("MATHESIS_STATE_DIR")
    if mathesis_dir and Path(mathesis_dir).exists():
        mathesis = mathesis_signals(mathesis_dir)
        signals.extend(mathesis.get("signals") or [])
    signals = list(dict.fromkeys(signals))

    challengers = generate_challengers(champion_genome, signals=signals, count=challenger_count)
    trials: list[dict[str, Any]] = []
    winner: tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]] | None = None

    for index, genome in enumerate(challengers):
        runtime, report = _train_genome(
            genome,
            steps=steps,
            seed=2000 + cycle * 17 + index,
            device=device,
        )
        eligible, reason = _research_eligible(
            champion_report,
            report,
            minimum_loss_gain=minimum_loss_gain,
            max_domain_regression=max_domain_regression,
        )
        trials.append({
            "genome": genome.to_dict(),
            "report": report,
            "eligible": eligible,
            "reason": reason,
        })
        if eligible and (winner is None or report["loss"] < winner[2]["loss"]):
            winner = (genome, runtime, report)

    promoted = winner is not None
    if winner is not None:
        champion_genome, champion_runtime, champion_report = winner
        _save_champion(root, champion_genome, champion_runtime, champion_report)

    result = {
        "ok": True,
        "cycle": cycle,
        "bootstrapped": bootstrapped,
        "promoted": promoted,
        "champion": champion_genome.to_dict(),
        "champion_report": champion_report,
        "signals": signals,
        "mathesis": mathesis,
        "trials": trials,
        "policy": {
            "research_only": True,
            "production_qualification_separate": True,
            "minimum_loss_gain": minimum_loss_gain,
            "max_domain_regression": max_domain_regression,
        },
        "updated_at": time.time(),
    }
    _atomic_json(root / "status.json", result)
    with (root / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> int:
    result = run_research_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
