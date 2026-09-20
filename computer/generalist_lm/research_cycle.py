from __future__ import annotations

import hashlib
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
from .model import CausalTransformerLM, estimate_parameter_count, parameter_count
from .runtime import GeneralistRuntime
from .tokenizer import ByteTokenizer
from .training import encode_sft_example, loss_on_examples, train_sft


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
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _teacher_forced_accuracy(model, tokenizer: ByteTokenizer, rows: list[ResearchRow], *, device: str) -> dict[str, Any]:
    import hashlib
    import torch

    encoded = [encode_sft_example(row.sft(), tokenizer, model.config.context_length) for row in rows]
    ids = torch.stack([item[0] for item in encoded]).to(device)
    labels = torch.stack([item[1] for item in encoded]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(ids)["logits"]
    predicted = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    mask = targets != -100
    correct = (predicted == targets) & mask

    total = int(mask.sum().item())
    token_accuracy = float(correct.sum().item() / total) if total else 0.0
    solved: list[str] = []
    domain_correct: dict[str, int] = {}
    domain_total: dict[str, int] = {}
    for index, row in enumerate(rows):
        row_mask = mask[index]
        row_correct = correct[index]
        count = int(row_mask.sum().item())
        good = int(row_correct.sum().item())
        domain_correct[row.domain] = domain_correct.get(row.domain, 0) + good
        domain_total[row.domain] = domain_total.get(row.domain, 0) + count
        if count and bool(torch.all(row_correct[row_mask]).item()):
            prompt = row.messages[0]["content"]
            digest = hashlib.sha256(f"{row.domain}\0{prompt}".encode("utf-8")).hexdigest()[:16]
            solved.append(f"{row.domain}:{digest}")

    return {
        "target_token_accuracy": token_accuracy,
        "domain_token_accuracy": {
            domain: (domain_correct[domain] / domain_total[domain] if domain_total[domain] else 0.0)
            for domain in sorted(domain_total)
        },
        "solved_items": sorted(solved),
        "target_tokens": total,
    }


def _grouped_validation(model, tokenizer: ByteTokenizer, rows: list[ResearchRow], *, device: str = "cpu") -> dict[str, Any]:
    all_examples = [row.sft() for row in rows]
    overall = loss_on_examples(model, tokenizer, all_examples, device=device)
    domains: dict[str, float] = {}
    for domain in sorted({row.domain for row in rows}):
        examples = [row.sft() for row in rows if row.domain == domain]
        domains[domain] = loss_on_examples(model, tokenizer, examples, device=device)
    accuracy = _teacher_forced_accuracy(model, tokenizer, rows, device=device)
    return {
        "loss": float(overall),
        "domain_loss": domains,
        **accuracy,
        "finite": bool(math.isfinite(overall) and all(math.isfinite(v) for v in domains.values())),
    }


def _research_score(report: dict[str, Any], params: int) -> float:
    loss = float(report["loss"])
    efficiency_penalty = min(1.0, params / 10_000_000.0) * 0.01
    return float(100.0 / (1.0 + loss) - efficiency_penalty)


def _genome_training_seed(genome: GeneralistGenome, *, namespace: str = "candidate") -> int:
    payload = genome.to_dict()
    for key in ("genome_id", "parent_id", "generation"):
        payload.pop(key, None)
    raw = (namespace + "\0" + json.dumps(payload, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    return 1 + (value % 2_000_000_000)


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

    old_accuracy = float(champion.get("target_token_accuracy", 0.0))
    new_accuracy = float(candidate.get("target_token_accuracy", 0.0))
    if new_accuracy + 0.01 < old_accuracy:
        return False, "candidate regressed in held-out target-token accuracy"

    old_domains = champion.get("domain_token_accuracy") or {}
    new_domains = candidate.get("domain_token_accuracy") or {}
    for domain, old_value in old_domains.items():
        if domain not in new_domains:
            return False, f"candidate lost token-accuracy domain: {domain}"
        if float(new_domains[domain]) + 0.02 < float(old_value):
            return False, f"candidate regressed in target-token domain: {domain}"

    remembered = set(champion.get("solved_items") or [])
    retained = set(candidate.get("solved_items") or [])
    forgotten = sorted(remembered - retained)
    if forgotten:
        return False, f"candidate forgot {len(forgotten)} previously solved held-out items"

    return True, "held-out multi-domain loss and anti-forgetting gates passed"


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


def _training_rows_for_genome(genome: GeneralistGenome) -> list[ResearchRow]:
    """Turn strategy genes into real curriculum weighting, never metadata-only."""
    base = list(train_rows())
    extra: list[ResearchRow] = []
    if genome.code_adapter:
        extra.extend(row for row in base if row.domain == "coding")
    if genome.data_adapter:
        extra.extend(row for row in base if row.domain == "data")
    if genome.retrieval_adapter:
        extra.extend(row for row in base if row.domain in {"tools", "structured"})
    if genome.symbolic_adapter:
        extra.extend(row for row in base if row.domain == "reasoning")
    for _ in range(max(0, int(genome.reasoning_depth) - 1)):
        extra.extend(row for row in base if row.domain == "reasoning")
    return base + extra


def _research_budget_reason(
    genome: GeneralistGenome,
    *,
    max_params: int,
    max_context: int,
    max_width: int,
    max_layers: int,
) -> str | None:
    if genome.context_length > max_context:
        return f"context_length {genome.context_length} exceeds research max {max_context}"
    if genome.d_model > max_width:
        return f"d_model {genome.d_model} exceeds research max {max_width}"
    if genome.n_layers > max_layers:
        return f"n_layers {genome.n_layers} exceeds research max {max_layers}"
    estimated = estimate_parameter_count(genome.model_config())
    if estimated > max_params:
        return f"estimated parameters {estimated} exceed research max {max_params}"
    return None


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
        [row.sft() for row in _training_rows_for_genome(genome)],
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
    bootstrap_steps = max(2, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "30")), 500))
    challenger_count = max(1, min(int(os.environ.get("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "2")), 6))
    minimum_loss_gain = max(0.0, float(os.environ.get("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "0.02")))
    max_domain_regression = max(0.0, float(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_DOMAIN_REGRESSION", "0.10")))
    max_params = max(100_000, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_PARAMS", "5000000")))
    max_context = max(64, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_CONTEXT", "512")))
    max_width = max(32, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_WIDTH", "256")))
    max_layers = max(1, int(os.environ.get("AIRI_GENERALIST_RESEARCH_MAX_LAYERS", "6")))

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
            champion_genome,
            steps=max(steps, bootstrap_steps),
            seed=_genome_training_seed(champion_genome, namespace="bootstrap"),
            device=device,
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

    challengers = generate_challengers(
        champion_genome,
        signals=signals,
        count=challenger_count,
        exploration_offset=max(0, cycle - 1),
    )
    trials: list[dict[str, Any]] = []
    winner: tuple[GeneralistGenome, GeneralistRuntime, dict[str, Any]] | None = None

    for index, genome in enumerate(challengers):
        budget_reason = _research_budget_reason(
            genome,
            max_params=max_params,
            max_context=max_context,
            max_width=max_width,
            max_layers=max_layers,
        )
        if budget_reason:
            trials.append({
                "genome": genome.to_dict(),
                "report": None,
                "eligible": False,
                "reason": f"research_resource_budget:{budget_reason}",
            })
            continue
        runtime, report = _train_genome(
            genome,
            steps=steps,
            seed=_genome_training_seed(genome),
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
            "research_resource_budget": {
                "max_params": max_params,
                "max_context": max_context,
                "max_width": max_width,
                "max_layers": max_layers,
            },
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
