from __future__ import annotations

import json
from collections import Counter
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import (
    DatasetView,
    _dataset_lock,
    class_counts,
    encode_text,
    ensure_canary_partition,
    load_records,
    persistent_split_records,
    source_family,
    source_family_counts,
)
from .genome import Genome, crossover, mutate, random_genome
from .model import build_model, parameter_count


@dataclass
class EvolutionConfig:
    mode: str = "safe"
    population: int = 8
    generations: int = 3
    candidate_epochs: int = 2
    finalist_epochs: int = 5
    seed: int = 1337
    min_samples: int = 40
    vocab_size: int = 8192
    max_params: int = 600_000
    max_latency_ms: float = 35.0
    min_f1_first_champion: float = 0.55
    min_f1_improvement: float = 0.002
    max_f1_regression: float = 0.01
    max_class_f1_regression: float = 0.03
    max_brier_regression: float = 0.02
    max_efficiency_accuracy_regression: float = 0.002
    max_efficiency_brier_regression: float = 0.005
    promotion_repeats: int = 3
    min_promotion_votes: int = 2
    crossover_probability: float = 0.60
    elite_count: int = 2
    abstain_threshold: float = 0.65
    replay_balance_power: float = 0.65
    min_first_canary_f1: float = 0.45
    max_canary_f1_regression: float = 0.05
    max_canary_class_regression: float = 0.10

    @classmethod
    def for_mode(cls, mode: str, **overrides):
        mode = str(mode or "safe").lower()
        if mode not in {"safe", "experimental"}:
            raise ValueError("mode must be safe or experimental")
        if mode == "experimental":
            base = cls(mode=mode, population=12, generations=5, candidate_epochs=2, finalist_epochs=6,
                       max_params=1_200_000, max_latency_ms=55.0, crossover_probability=0.78, elite_count=2)
        else:
            base = cls(mode=mode)
        for k, v in overrides.items():
            if v is not None and hasattr(base, k):
                setattr(base, k, v)
        base.population = max(4, min(64, int(base.population)))
        base.generations = max(1, min(100, int(base.generations)))
        base.candidate_epochs = max(1, min(30, int(base.candidate_epochs)))
        base.finalist_epochs = max(base.candidate_epochs, min(50, int(base.finalist_epochs)))
        base.elite_count = max(1, min(base.population - 1, int(base.elite_count)))
        base.abstain_threshold = min(0.95, max(0.50, float(base.abstain_threshold)))
        base.replay_balance_power = min(1.0, max(0.0, float(base.replay_balance_power)))
        base.min_first_canary_f1 = min(1.0, max(0.0, float(base.min_first_canary_f1)))
        base.max_class_f1_regression = min(0.5, max(0.0, float(base.max_class_f1_regression)))
        base.max_brier_regression = min(0.5, max(0.0, float(base.max_brier_regression)))
        base.max_efficiency_accuracy_regression = min(0.2, max(0.0, float(base.max_efficiency_accuracy_regression)))
        base.max_efficiency_brier_regression = min(0.2, max(0.0, float(base.max_efficiency_brier_regression)))
        base.max_canary_f1_regression = min(0.5, max(0.0, float(base.max_canary_f1_regression)))
        base.max_canary_class_regression = min(0.5, max(0.0, float(base.max_canary_class_regression)))
        base.promotion_repeats = max(1, min(7, int(base.promotion_repeats)))
        base.min_promotion_votes = max(1, min(base.promotion_repeats, int(base.min_promotion_votes)))
        return base


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _metrics(y_true: list[int], probs: list[float]) -> dict[str, float]:
    pred = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 0)
    precision_real = tp / (tp + fp) if tp + fp else 0.0
    recall_real = tp / (tp + fn) if tp + fn else 0.0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if precision_real + recall_real else 0.0
    precision_fake = tn / (tn + fn) if tn + fn else 0.0
    recall_fake = tn / (tn + fp) if tn + fp else 0.0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if precision_fake + recall_fake else 0.0
    macro_f1 = (f1_real + f1_fake) / 2.0
    acc = (tp + tn) / max(1, len(y_true))
    brier = sum((p - y) ** 2 for y, p in zip(y_true, probs)) / max(1, len(y_true))
    return {
        "accuracy": acc, "precision": precision_real, "recall": recall_real, "f1": macro_f1,
        "f1_real": f1_real, "f1_fake": f1_fake, "brier": brier,
    }


def _loader(records: list[dict], genome: Genome, vocab_size: int, shuffle: bool, balance_power: float = 0.65):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = DatasetView(records, genome.max_len, vocab_size)
    if shuffle and records:
        families = [source_family(row) for row in records]
        counts = Counter(families)
        if len(counts) > 1 and balance_power > 0:
            weights = [1.0 / (counts[family] ** float(balance_power)) for family in families]
            sampler = WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.double),
                num_samples=len(records),
                replacement=True,
            )
            return DataLoader(dataset, batch_size=genome.batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=genome.batch_size, shuffle=shuffle)


def train_model(genome: Genome, train_records: list[dict], epochs: int, vocab_size: int, seed: int, balance_power: float = 0.65):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(genome, vocab_size=vocab_size)
    device = _device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _loader(train_records, genome, vocab_size, True, balance_power)
    model.train()
    losses = []
    for _ in range(epochs):
        total = 0.0
        seen = 0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        losses.append(total / max(1, seen))
    return model, {"loss": losses[-1] if losses else None, "loss_curve": losses, "device": str(device)}


def evaluate_model(model, genome: Genome, records: list[dict], vocab_size: int, latency_repeats: int = 8) -> dict[str, Any]:
    import torch
    device = next(model.parameters()).device
    model.eval()
    ys: list[int] = []
    probs: list[float] = []
    with torch.inference_mode():
        for ids, mask, y in _loader(records, genome, vocab_size, False):
            logits = model(ids.to(device), mask.to(device))
            pr = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            probs.extend(float(p) for p in pr)
            ys.extend(int(v) for v in y.tolist())
    m = _metrics(ys, probs)
    params = parameter_count(model)
    cpu_model = model.to("cpu")
    sample_text = records[0]["text"] if records else "sample"
    ids, mask = encode_text(sample_text, genome.max_len, vocab_size)
    ids_t = torch.tensor([ids], dtype=torch.long)
    mask_t = torch.tensor([mask], dtype=torch.bool)
    with torch.inference_mode():
        for _ in range(2):
            cpu_model(ids_t, mask_t)
        timings = []
        for _ in range(max(3, latency_repeats)):
            t0 = time.perf_counter()
            cpu_model(ids_t, mask_t)
            timings.append((time.perf_counter() - t0) * 1000)
    latency_ms = statistics.median(timings)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(cpu_model.state_dict(), tmp_path)
        bytes_size = tmp_path.stat().st_size
    finally:
        tmp_path.unlink(missing_ok=True)
    return {**m, "params": params, "model_bytes": bytes_size, "latency_ms": latency_ms}


def evaluate_source_families(model, genome: Genome, records: list[dict], vocab_size: int) -> dict[str, Any]:
    import torch
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(source_family(row), []).append(row)
    device = next(model.parameters()).device
    model.eval()
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for family, rows in sorted(grouped.items()):
            ys: list[int] = []
            probs: list[float] = []
            for ids, mask, y in _loader(rows, genome, vocab_size, False):
                logits = model(ids.to(device), mask.to(device))
                pr = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
                probs.extend(float(p) for p in pr)
                ys.extend(int(v) for v in y.tolist())
            result[family] = {
                "records": len(rows),
                "class_counts": class_counts(rows),
                **_metrics(ys, probs),
            }
    return result


def _canary_decision(candidate: dict, old: dict | None, cfg: EvolutionConfig) -> tuple[bool, str]:
    if not candidate:
        return True, "no canary configured"
    if old is None:
        ok = float(candidate.get("f1", 0.0)) >= cfg.min_first_canary_f1
        return ok, "first champion canary threshold met" if ok else "first champion canary macro-F1 below threshold"
    if float(candidate.get("f1", 0.0)) < float(old.get("f1", 0.0)) - cfg.max_canary_f1_regression:
        return False, "canary macro-F1 regression exceeds gate"
    for key in ("f1_real", "f1_fake"):
        if float(candidate.get(key, 0.0)) < float(old.get(key, 0.0)) - cfg.max_canary_class_regression:
            return False, f"canary {key} regression exceeds gate"
    return True, "canary gate passed"


def fitness(metrics: dict[str, Any], cfg: EvolutionConfig) -> dict[str, Any]:
    params = max(1, int(metrics["params"]))
    latency = max(0.01, float(metrics["latency_ms"]))
    size_score = 1.0 / (1.0 + params / 250_000.0)
    latency_score = 1.0 / (1.0 + latency / 12.0)
    calibration_score = max(0.0, 1.0 - float(metrics["brier"]))
    score = 0.73 * float(metrics["f1"]) + 0.09 * float(metrics["accuracy"]) + 0.06 * calibration_score + 0.07 * size_score + 0.05 * latency_score
    feasible = params <= cfg.max_params and latency <= cfg.max_latency_ms
    if not feasible:
        score -= 0.35
    return {"fitness": score, "feasible": feasible, "size_score": size_score, "latency_score": latency_score}


def pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    front: list[str] = []
    for a in rows:
        dominated = False
        for b in rows:
            if a is b:
                continue
            better_or_equal = b["f1"] >= a["f1"] and b["params"] <= a["params"] and b["latency_ms"] <= a["latency_ms"]
            strictly = b["f1"] > a["f1"] or b["params"] < a["params"] or b["latency_ms"] < a["latency_ms"]
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            front.append(a["genome_id"])
    return front


def _save_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _champion_complete(path: Path) -> bool:
    path = Path(path)
    return all((path / name).exists() for name in ("genome.json", "model.pt", "metrics.json", "provenance.json"))


def _recover_champion_state(state_dir: Path) -> dict[str, Any]:
    state_dir = Path(state_dir)
    champion = state_dir / "champion"
    backup = state_dir / ".champion-old"
    pending = state_dir / ".champion-new"
    actions: list[str] = []

    if _champion_complete(champion):
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
            actions.append("removed_stale_backup")
        if pending.exists():
            shutil.rmtree(pending, ignore_errors=True)
            actions.append("removed_stale_pending")
        return {"ok": True, "actions": actions}

    if champion.exists():
        corrupt = state_dir / f".champion-corrupt-{int(time.time())}"
        try:
            champion.rename(corrupt)
            actions.append("quarantined_incomplete_champion")
        except OSError:
            shutil.rmtree(champion, ignore_errors=True)
            actions.append("removed_incomplete_champion")

    if _champion_complete(backup):
        backup.rename(champion)
        actions.append("restored_backup")
        if pending.exists():
            shutil.rmtree(pending, ignore_errors=True)
        return {"ok": True, "actions": actions}

    if _champion_complete(pending):
        pending.rename(champion)
        actions.append("promoted_complete_pending")
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        return {"ok": True, "actions": actions}

    return {"ok": not (backup.exists() or pending.exists()), "actions": actions}


def _load_champion(champion_dir: Path, vocab_size: int):
    import torch
    gp = champion_dir / "genome.json"
    mp = champion_dir / "model.pt"
    if not gp.exists() or not mp.exists():
        return None
    genome = Genome.from_dict(json.loads(gp.read_text(encoding="utf-8")))
    model = build_model(genome, vocab_size=vocab_size)
    payload = torch.load(mp, map_location="cpu", weights_only=True)
    model.load_state_dict(payload)
    return genome, model


def _promotion_decision(candidate: dict, old: dict | None, cfg: EvolutionConfig) -> tuple[bool, str]:
    if not candidate.get("feasible"):
        return False, "candidate violates parameter/latency constraints"
    if old is None:
        ok = candidate["f1"] >= cfg.min_f1_first_champion
        return ok, "first champion threshold met" if ok else "first champion macro-F1 below threshold"

    if candidate["f1"] < old["f1"] - cfg.max_f1_regression:
        return False, "macro-F1 regression exceeds gate"

    # Quality gates are evaluated before efficiency. A faster/smaller model is
    # never promoted merely because it is cheaper if calibration or one class
    # materially regresses.
    for key in ("f1_real", "f1_fake"):
        if key in candidate and key in old:
            if float(candidate[key]) < float(old[key]) - cfg.max_class_f1_regression:
                return False, f"{key} regression exceeds gate"
    if "brier" in candidate and "brier" in old:
        if float(candidate["brier"]) > float(old["brier"]) + cfg.max_brier_regression:
            return False, "calibration regression exceeds gate"

    accuracy_win = candidate["f1"] >= old["f1"] + cfg.min_f1_improvement
    efficiency_quality_ok = True
    if "accuracy" in candidate and "accuracy" in old:
        efficiency_quality_ok = efficiency_quality_ok and (
            float(candidate["accuracy"]) >= float(old["accuracy"]) - cfg.max_efficiency_accuracy_regression
        )
    if "brier" in candidate and "brier" in old:
        efficiency_quality_ok = efficiency_quality_ok and (
            float(candidate["brier"]) <= float(old["brier"]) + cfg.max_efficiency_brier_regression
        )

    compact_win = (
        candidate["f1"] >= old["f1"] - 0.002
        and efficiency_quality_ok
        and candidate["params"] <= old["params"] * 0.90
    )
    latency_win = (
        candidate["f1"] >= old["f1"] - 0.002
        and efficiency_quality_ok
        and candidate["latency_ms"] <= old["latency_ms"] * 0.85
    )
    if accuracy_win:
        return True, "macro-F1 improved"
    if compact_win:
        return True, "quality-stable candidate with >=10% fewer parameters"
    if latency_win:
        return True, "quality-stable candidate with >=15% lower latency"
    return False, "candidate did not beat champion quality/efficiency criteria"

def _genome_parameter_count(genome: Genome, vocab_size: int) -> int:
    model = build_model(genome, vocab_size=vocab_size)
    return parameter_count(model)


def _random_feasible_genome(rng: random.Random, genome_id: str, cfg: EvolutionConfig, generation: int = 0) -> Genome:
    best = None
    best_params = None
    for attempt in range(40):
        candidate = random_genome(rng, genome_id, cfg.mode, generation)
        params = _genome_parameter_count(candidate, cfg.vocab_size)
        if params <= cfg.max_params:
            return candidate
        if best is None or params < best_params:
            best, best_params = candidate, params
    raise RuntimeError(
        f"unable to sample a genome under max_params={cfg.max_params}; smallest sampled={best_params}"
    )


def _select_parent(rows: list[dict], rng: random.Random) -> dict:
    k = min(3, len(rows))
    sample = rng.sample(rows, k)
    return max(sample, key=lambda x: x["fitness"])


def _prune_old_trial_weights(state_dir: Path, keep_runs: int = 5) -> None:
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.exists():
        return
    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_run in run_dirs[max(1, int(keep_runs)):]:
        for pattern in ("selected-trial.pt", "promotion-trial-*.pt"):
            for path in old_run.glob(pattern):
                path.unlink(missing_ok=True)


def run_evolution(state_dir: Path, cfg: EvolutionConfig) -> dict[str, Any]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with _dataset_lock(state_dir / "evolution-run", timeout=2.0, stale_after=21600.0):
        return _run_evolution_unlocked(state_dir, cfg)


def _run_evolution_unlocked(state_dir: Path, cfg: EvolutionConfig) -> dict[str, Any]:
    import torch
    state_dir = Path(state_dir)
    recovery = _recover_champion_state(state_dir)
    if not recovery.get("ok"):
        raise RuntimeError(f"champion state recovery failed: {recovery}")
    data_path = state_dir / "data" / "verified.jsonl"
    champion_dir = state_dir / "champion"
    records = load_records(data_path)
    counts = class_counts(records)
    if len(records) < cfg.min_samples:
        raise ValueError(f"need at least {cfg.min_samples} verified samples; found {len(records)}")
    if min(counts.values()) < 4:
        raise ValueError("need at least 4 verified samples in each class")
    remaining, canary, canary_info = ensure_canary_partition(state_dir, records, seed=cfg.seed)
    train, val, test, split_info = persistent_split_records(state_dir, remaining, cfg.seed)
    run_id = time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_dir = state_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(run_dir / "config.json", asdict(cfg))
    _save_json(run_dir / "dataset.json", {
        "records": len(records),
        "class_counts": counts,
        "source_families": source_family_counts(records),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "split_manifest": split_info,
        "canary": canary_info,
    })

    rng = random.Random(cfg.seed + int(time.time()) % 100_000)
    population = [_random_feasible_genome(rng, f"g0-{i:03d}", cfg, 0) for i in range(cfg.population)]
    history: list[dict] = []
    best_row = None
    best_genome = None

    for generation in range(cfg.generations):
        rows = []
        for idx, genome in enumerate(population):
            candidate_seed = cfg.seed + generation * 10_000 + idx
            try:
                static_params = _genome_parameter_count(genome, cfg.vocab_size)
                if static_params > cfg.max_params:
                    row = {
                        "generation": generation,
                        "genome_id": genome.genome_id,
                        "fitness": -1.0,
                        "feasible": False,
                        "params": static_params,
                        "constraint_rejected": "max_params",
                        "genome": genome.to_dict(),
                    }
                    rows.append(row)
                    history.append(row)
                    with (run_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                model, train_info = train_model(genome, train, cfg.candidate_epochs, cfg.vocab_size, candidate_seed, cfg.replay_balance_power)
                metrics = evaluate_model(model, genome, val, cfg.vocab_size)
                fit = fitness(metrics, cfg)
                row = {"generation": generation, "genome_id": genome.genome_id, **metrics, **fit, "train": train_info, "genome": genome.to_dict()}
            except Exception as exc:
                row = {"generation": generation, "genome_id": genome.genome_id, "fitness": -999.0, "feasible": False, "error": repr(exc), "genome": genome.to_dict()}
            rows.append(row)
            history.append(row)
            with (run_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        valid = [r for r in rows if "f1" in r]
        if not valid:
            raise RuntimeError("all candidates failed; inspect candidates.jsonl")
        for r in valid:
            r["pareto"] = r["genome_id"] in pareto_front(valid)
        generation_best = max(valid, key=lambda x: x["fitness"])
        if best_row is None or generation_best["fitness"] > best_row["fitness"]:
            best_row = generation_best
            best_genome = Genome.from_dict(generation_best["genome"])
        _save_json(run_dir / f"generation-{generation:03d}.json", {"best": generation_best, "pareto": pareto_front(valid)})

        ranked = sorted(valid, key=lambda x: x["fitness"], reverse=True)
        next_population: list[Genome] = []
        for elite_idx, row in enumerate(ranked[:cfg.elite_count]):
            elite = Genome.from_dict(row["genome"]).clone(f"g{generation+1}-elite-{elite_idx:02d}")
            elite.generation = generation + 1
            next_population.append(elite)
        while len(next_population) < cfg.population:
            p1row = _select_parent(ranked, rng)
            p1 = Genome.from_dict(p1row["genome"])
            child_id = f"g{generation+1}-{len(next_population):03d}"
            if rng.random() < cfg.crossover_probability and len(ranked) > 1:
                p2row = _select_parent(ranked, rng)
                p2 = Genome.from_dict(p2row["genome"])
                child = crossover(p1, p2, rng, child_id, cfg.mode)
                if rng.random() < 0.85:
                    child = mutate(child, rng, child_id, cfg.mode)
            else:
                child = mutate(p1, rng, child_id, cfg.mode)
            next_population.append(child)
        population = next_population

    assert best_genome is not None and best_row is not None

    old_eval = None
    old_canary = None
    old_loaded = _load_champion(champion_dir, cfg.vocab_size)
    if old_loaded is not None:
        old_genome, old_model = old_loaded
        old_eval = {"genome_id": old_genome.genome_id, **evaluate_model(old_model, old_genome, test, cfg.vocab_size, latency_repeats=12)}
        old_eval.update(fitness(old_eval, cfg))
        if canary:
            old_canary = evaluate_model(old_model, old_genome, canary, cfg.vocab_size, latency_repeats=3)

    promotion_trials = []
    votes = 0
    trial_state_paths: dict[int, str] = {}
    for trial in range(cfg.promotion_repeats):
        trial_model, trial_train = train_model(
            best_genome,
            train + val,
            cfg.finalist_epochs,
            cfg.vocab_size,
            cfg.seed + 777_777 + trial * 97,
            cfg.replay_balance_power,
        )
        trial_metrics = evaluate_model(trial_model, best_genome, test, cfg.vocab_size, latency_repeats=8)
        trial_fit = fitness(trial_metrics, cfg)
        trial_canary = evaluate_model(trial_model, best_genome, canary, cfg.vocab_size, latency_repeats=3) if canary else {}
        primary_ok, primary_reason = _promotion_decision({**trial_metrics, **trial_fit}, old_eval, cfg)
        canary_ok, canary_reason = _canary_decision(trial_canary, old_canary, cfg)
        trial_promote = primary_ok and canary_ok
        state_path = run_dir / f"promotion-trial-{trial:02d}.pt"
        torch.save(trial_model.to("cpu").state_dict(), state_path)
        trial_state_paths[trial] = str(state_path)
        trial_row = {
            "trial": trial,
            "genome_id": best_genome.genome_id,
            **trial_metrics,
            **trial_fit,
            "train": trial_train,
            "canary": trial_canary,
            "promotion_vote": trial_promote,
            "promotion_reason": f"{primary_reason}; {canary_reason}",
            "state_path": str(state_path),
        }
        promotion_trials.append(trial_row)
        votes += int(trial_promote)

    promote = votes >= cfg.min_promotion_votes
    reason = f"{votes}/{cfg.promotion_repeats} independent promotion votes; " + ("majority gate passed" if promote else "majority gate failed")
    eligible = [row for row in promotion_trials if row["promotion_vote"]] or promotion_trials
    candidate_final = max(eligible, key=lambda row: (row["f1"], -row["brier"]))
    selected_trial = int(candidate_final["trial"])
    best_state = torch.load(trial_state_paths[selected_trial], map_location="cpu", weights_only=True)
    candidate_final = dict(candidate_final)
    candidate_final.pop("state_path", None)
    candidate_final["selected_trial"] = selected_trial
    selected_model = build_model(best_genome, vocab_size=cfg.vocab_size)
    selected_model.load_state_dict(best_state)
    selected_model.eval()
    candidate_final["source_families"] = evaluate_source_families(selected_model, best_genome, test, cfg.vocab_size)

    # Keep only the selected trial weights for auditability; loser trial state dicts
    # are redundant and would otherwise make long-running autopilot state grow quickly.
    selected_trial_path = run_dir / f"promotion-trial-{selected_trial:02d}.pt"
    selected_audit_path = run_dir / "selected-trial.pt"
    if selected_trial_path.exists():
        selected_trial_path.replace(selected_audit_path)
    for path in run_dir.glob("promotion-trial-*.pt"):
        path.unlink(missing_ok=True)
    for row in promotion_trials:
        row["state_path"] = str(selected_audit_path) if int(row["trial"]) == selected_trial and selected_audit_path.exists() else None
    candidate_final["selected_trial_path"] = str(selected_audit_path) if selected_audit_path.exists() else None

    if promote:
        champion_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = state_dir / ".champion-new"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        _save_json(tmp_dir / "genome.json", best_genome.to_dict())
        torch.save(best_state, tmp_dir / "model.pt")
        _save_json(tmp_dir / "metrics.json", candidate_final)
        _save_json(tmp_dir / "provenance.json", {
            "run_id": run_id,
            "promoted_at": time.time(),
            "reason": reason,
            "dataset_records": len(records),
            "class_counts": counts,
            "source_families": source_family_counts(records),
            "canary": canary_info,
            "mode": cfg.mode,
            "abstain_threshold": cfg.abstain_threshold,
        })
        backup = state_dir / ".champion-old"
        shutil.rmtree(backup, ignore_errors=True)
        if champion_dir.exists() and any(champion_dir.iterdir()):
            champion_dir.rename(backup)
        tmp_dir.rename(champion_dir)
        shutil.rmtree(backup, ignore_errors=True)
        # Edge artifacts are champion-specific; never keep an INT8 model from an older genome.
        shutil.rmtree(state_dir / "edge", ignore_errors=True)

    result = {
        "ok": True,
        "run_id": run_id,
        "mode": cfg.mode,
        "dataset_records": len(records),
        "class_counts": counts,
        "source_families": source_family_counts(records),
        "canary": canary_info,
        "old_canary": old_canary,
        "best_search_candidate": best_row,
        "candidate": candidate_final,
        "previous_champion": old_eval,
        "promotion_trials": promotion_trials,
        "promoted": promote,
        "promotion_reason": reason,
        "run_dir": str(run_dir),
    }
    _save_json(run_dir / "result.json", result)
    with (state_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in result.items() if k != "best_search_candidate"}, ensure_ascii=False) + "\n")
    _prune_old_trial_weights(state_dir, keep_runs=5)
    return result


def load_champion_for_prediction(state_dir: Path, vocab_size: int = 8192):
    loaded = _load_champion(Path(state_dir) / "champion", vocab_size)
    if loaded is None:
        raise FileNotFoundError("no champion model yet; ingest verified data and run evolution")
    return loaded


def decision_from_probability(p_real: float, threshold: float = 0.65) -> tuple[str, float]:
    p_real = min(1.0, max(0.0, float(p_real)))
    threshold = min(0.95, max(0.50, float(threshold)))
    confidence = max(p_real, 1.0 - p_real)
    if confidence < threshold:
        return "uncertain", confidence
    return ("likely_real" if p_real >= 0.5 else "likely_fake"), confidence


def predict_text(state_dir: Path, text: str, vocab_size: int = 8192) -> dict[str, Any]:
    import torch
    state_dir = Path(state_dir)
    genome, model = load_champion_for_prediction(state_dir, vocab_size)
    ids, mask = encode_text(text, genome.max_len, vocab_size)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor([ids], dtype=torch.long), torch.tensor([mask], dtype=torch.bool))
        p_real = float(torch.softmax(logits, dim=-1)[0, 1])
    p_fake = 1.0 - p_real
    provenance = {}
    try:
        provenance = json.loads((state_dir / "champion" / "provenance.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    threshold = float(provenance.get("abstain_threshold", 0.65))
    label, confidence = decision_from_probability(p_real, threshold)
    return {
        "label": label,
        "binary_preference": "likely_real" if p_real >= 0.5 else "likely_fake",
        "real_probability": p_real,
        "fake_probability": p_fake,
        "confidence": confidence,
        "abstain_threshold": threshold,
        "genome_id": genome.genome_id,
        "architecture": genome.to_dict(),
        "warning": "This is a learned reliability estimate, not proof that a claim is true or false. 'uncertain' means confidence is below the champion abstention threshold.",
    }
