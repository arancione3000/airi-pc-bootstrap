from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution.data import append_verified, class_counts, encode_text, load_records, split_records, text_fingerprint
from evolution.engine import EvolutionConfig, _metrics, _promotion_decision, fitness, pareto_front
from evolution.genome import crossover, mutate, random_genome


def test_hash_encoder_is_stable_and_padded():
    a, am = encode_text("Hello world!", 12)
    b, bm = encode_text("Hello world!", 12)
    assert a == b and am == bm
    assert len(a) == len(am) == 12
    assert sum(am) == 3


def test_verified_dataset_deduplicates_and_splits(tmp_path: Path):
    path = tmp_path / "verified.jsonl"
    for label in (0, 1):
        for i in range(10):
            append_verified(path, {"text": f"sample text class {label} item {i}", "label": label})
    duplicate = append_verified(path, {"text": "sample text class 1 item 1", "label": 1})
    rows = load_records(path)
    assert duplicate["duplicate"] is True
    assert len(rows) == 20
    assert class_counts(rows) == {"fake": 10, "real": 10}
    tr, va, te = split_records(rows, 42)
    assert len(tr) + len(va) + len(te) == 20
    assert {r["id"] for r in tr}.isdisjoint({r["id"] for r in te})


def test_genomes_mutate_and_crossover_with_valid_shapes():
    rng = random.Random(7)
    a = random_genome(rng, "a", "safe")
    b = random_genome(rng, "b", "safe")
    m = mutate(a, rng, "m", "safe")
    c = crossover(a, b, rng, "c", "safe")
    assert m.genome_id == "m" and c.genome_id == "c"
    assert 1 <= len(m.blocks) <= 4
    assert 1 <= len(c.blocks) <= 4
    assert all(block.width > 0 for block in m.blocks + c.blocks)


def test_macro_f1_penalizes_single_class_collapse():
    m = _metrics([0, 0, 1, 1], [0.9, 0.9, 0.9, 0.9])
    assert m["f1_real"] > 0.6
    assert m["f1_fake"] == 0.0
    assert m["f1"] < 0.5


def test_fitness_penalizes_constraint_violations():
    cfg = EvolutionConfig.for_mode("safe")
    good = fitness({"f1": .8, "accuracy": .82, "brier": .12, "params": 100_000, "latency_ms": 5}, cfg)
    bad = fitness({"f1": .8, "accuracy": .82, "brier": .12, "params": 3_000_000, "latency_ms": 100}, cfg)
    assert good["feasible"] is True
    assert bad["feasible"] is False
    assert good["fitness"] > bad["fitness"]


def test_promotion_gate_and_pareto():
    cfg = EvolutionConfig.for_mode("safe")
    old = {"f1": .80, "params": 200_000, "latency_ms": 10}
    faster = {"f1": .799, "params": 200_000, "latency_ms": 8, "feasible": True}
    promote, reason = _promotion_decision(faster, old, cfg)
    assert promote is True and "latency" in reason.lower()
    rows = [
        {"genome_id": "a", "f1": .80, "params": 100_000, "latency_ms": 10},
        {"genome_id": "b", "f1": .79, "params": 200_000, "latency_ms": 12},
        {"genome_id": "c", "f1": .82, "params": 150_000, "latency_ms": 11},
    ]
    front = pareto_front(rows)
    assert "b" not in front
    assert "a" in front and "c" in front


def test_model_forward_for_all_block_types():
    import pytest
    torch = pytest.importorskip("torch")
    from evolution.genome import BlockGene, Genome
    from evolution.model import build_model
    for kind in ("conv", "attention", "gru", "mlp"):
        g = Genome("x", embed_dim=32, max_len=16, blocks=[BlockGene(kind, 32, heads=2)]).normalize()
        m = build_model(g, vocab_size=128)
        ids = torch.randint(2, 128, (3, 16))
        mask = torch.ones((3, 16), dtype=torch.bool)
        out = m(ids, mask)
        assert tuple(out.shape) == (3, 2)


def test_conflicting_labels_are_rejected(tmp_path: Path):
    path = tmp_path / "verified.jsonl"
    append_verified(path, {"text": "The same verified claim appears here", "label": 1})
    import pytest
    with pytest.raises(ValueError, match="conflicting verified labels"):
        append_verified(path, {"text": "  the same verified claim appears here  ", "label": 0})


def test_split_never_leaks_same_normalized_text():
    rows = []
    for label in (0, 1):
        for i in range(8):
            text = f"unique class {label} verified sample {i}"
            rows.append({"id": f"{label}-{i}", "text": text, "text_id": text_fingerprint(text), "label": label})
    tr, va, te = split_records(rows, 99)
    groups = [{r["text_id"] for r in part} for part in (tr, va, te)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_promotion_repeat_config_is_bounded():
    cfg = EvolutionConfig.for_mode("safe", promotion_repeats=99, min_promotion_votes=99)
    assert cfg.promotion_repeats == 7
    assert cfg.min_promotion_votes == 7
