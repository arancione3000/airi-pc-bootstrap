from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import DatasetView, class_counts, encode_text, load_records, split_records
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
    promotion_repeats: int = 3
    min_promotion_votes: int = 2
    crossover_probability: float = 0.60
    elite_count: int = 2
    abstain_threshold: float = 0.65

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


def _loader(records: list[dict], genome: Genome, vocab_size: int, shuffle: bool):
    from torch.utils.data import DataLoader
    return DataLoader(DatasetView(records, genome.max_len, vocab_size), batch_size=genome.batch_size, shuffle=shuffle)


def train_model(genome: Genome, train_records: list[dict], epochs: int, vocab_size: int, seed: int):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(genome, vocab_size=vocab_size)
    device = _device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _loader(train_records, genome, vocab_size, True)
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
    accuracy_win = candidate["f1"] >= old["f1"] + cfg.min_f1_improvement
    compact_win = candidate["f1"] >= old["f1"] - 0.002 and candidate["params"] <= old["params"] * 0.90
    latency_win = candidate["f1"] >= old["f1"] - 0.002 and candidate["latency_ms"] <= old["latency_ms"] * 0.85
    if accuracy_win:
        return True, "macro-F1 improved"
    if compact_win:
        return True, "similar macro-F1 with >=10% fewer parameters"
    if latency_win:
        return True, "similar macro-F1 with >=15% lower latency"
    return False, "candidate did not beat champion promotion criteria"


def _select_parent(rows: list[dict], rng: random.Random) -> dict:
    k = min(3, len(rows))
    sample = rng.sample(rows, k)
    return max(sample, key=lambda x: x["fitness"])


def run_evolution(state_dir: Path, cfg: EvolutionConfig) -> dict[str, Any]:
    import torch
    state_dir = Path(state_dir)
    data_path = state_dir / "data" / "verified.jsonl"
    champion_dir = state_dir / "champion"
    records = load_records(data_path)
    counts = class_counts(records)
    if len(records) < cfg.min_samples:
        raise ValueError(f"need at least {cfg.min_samples} verified samples; found {len(records)}")
    if min(counts.values()) < 4:
        raise ValueError("need at least 4 verified samples in each class")
    train, val, test = split_records(records, cfg.seed)
    run_id = time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_dir = state_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(run_dir / "config.json", asdict(cfg))
    _save_json(run_dir / "dataset.json", {"records": len(records), "class_counts": counts, "train": len(train), "val": len(val), "test": len(test)})

    rng = random.Random(cfg.seed + int(time.time()) % 100_000)
    population = [random_genome(rng, f"g0-{i:03d}", cfg.mode, 0) for i in range(cfg.population)]
    history: list[dict] = []
    best_row = None
    best_genome = None

    for generation in range(cfg.generations):
        rows = []
        for idx, genome in enumerate(population):
            candidate_seed = cfg.seed + generation * 10_000 + idx
            try:
                model, train_info = train_model(genome, train, cfg.candidate_epochs, cfg.vocab_size, candidate_seed)
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
    finalist_model, finalist_train = train_model(best_genome, train + val, cfg.finalist_epochs, cfg.vocab_size, cfg.seed + 777_777)
    finalist_metrics = evaluate_model(finalist_model, best_genome, test, cfg.vocab_size, latency_repeats=12)
    finalist_fit = fitness(finalist_metrics, cfg)
    candidate_final = {"genome_id": best_genome.genome_id, **finalist_metrics, **finalist_fit, "train": finalist_train}

    old_eval = None
    old_loaded = _load_champion(champion_dir, cfg.vocab_size)
    if old_loaded is not None:
        old_genome, old_model = old_loaded
        old_eval = {"genome_id": old_genome.genome_id, **evaluate_model(old_model, old_genome, test, cfg.vocab_size, latency_repeats=12)}
        old_eval.update(fitness(old_eval, cfg))

    promotion_trials = []
    votes = 0
    for trial in range(cfg.promotion_repeats):
        trial_model, trial_train = train_model(best_genome, train + val, cfg.finalist_epochs, cfg.vocab_size, cfg.seed + 777_777 + trial * 97)
        trial_metrics = evaluate_model(trial_model, best_genome, test, cfg.vocab_size, latency_repeats=8)
        trial_fit = fitness(trial_metrics, cfg)
        trial_row = {"trial": trial, "genome_id": best_genome.genome_id, **trial_metrics, **trial_fit, "train": trial_train}
        trial_promote, trial_reason = _promotion_decision(trial_row, old_eval, cfg)
        trial_row["promotion_vote"] = trial_promote
        trial_row["promotion_reason"] = trial_reason
        promotion_trials.append(trial_row)
        votes += int(trial_promote)
    promote = votes >= cfg.min_promotion_votes
    reason = f"{votes}/{cfg.promotion_repeats} independent promotion votes; " + ("majority gate passed" if promote else "majority gate failed")
    candidate_final = max(promotion_trials, key=lambda row: row["f1"])

    if promote:
        champion_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = state_dir / ".champion-new"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        _save_json(tmp_dir / "genome.json", best_genome.to_dict())
        torch.save(finalist_model.to("cpu").state_dict(), tmp_dir / "model.pt")
        _save_json(tmp_dir / "metrics.json", candidate_final)
        _save_json(tmp_dir / "provenance.json", {"run_id": run_id, "promoted_at": time.time(), "reason": reason, "dataset_records": len(records), "class_counts": counts, "mode": cfg.mode, "abstain_threshold": cfg.abstain_threshold})
        backup = state_dir / ".champion-old"
        shutil.rmtree(backup, ignore_errors=True)
        if champion_dir.exists() and any(champion_dir.iterdir()):
            champion_dir.rename(backup)
        tmp_dir.rename(champion_dir)
        shutil.rmtree(backup, ignore_errors=True)

    result = {
        "ok": True,
        "run_id": run_id,
        "mode": cfg.mode,
        "dataset_records": len(records),
        "class_counts": counts,
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
