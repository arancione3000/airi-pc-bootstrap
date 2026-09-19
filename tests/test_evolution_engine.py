from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution.data import append_verified, append_verified_many, class_counts, encode_text, ensure_canary_partition, load_records, persistent_split_records, source_family, source_family_counts, split_records, text_fingerprint
from evolution.engine import EvolutionConfig, _canary_decision, _metrics, _promotion_decision, decision_from_probability, fitness, pareto_front
from evolution.edge import edge_acceptance
from evolution.monitoring import detect_drift
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


def test_prediction_can_abstain():
    label, confidence = decision_from_probability(0.54, 0.65)
    assert label == "uncertain"
    assert 0.53 < confidence < 0.55
    label, confidence = decision_from_probability(0.91, 0.65)
    assert label == "likely_real"
    assert confidence == 0.91
    label, confidence = decision_from_probability(0.08, 0.65)
    assert label == "likely_fake"
    assert confidence == 0.92


def test_drift_detector_flags_performance_and_distribution_changes():
    result = detect_drift(
        {"f1": 0.90, "brier": 0.10},
        {"f1": 0.74, "brier": 0.24},
        baseline_real_fraction=0.50,
        current_real_fraction=0.82,
    )
    assert result["drift"] is True
    assert "macro_f1_drop" in result["reasons"]
    assert "calibration_degradation" in result["reasons"]
    assert "class_distribution_shift" in result["reasons"]


def test_edge_gate_requires_quality_and_real_efficiency_gain():
    good = edge_acceptance(
        float_bytes=1_000_000,
        int8_bytes=600_000,
        float_latency_ms=10.0,
        int8_latency_ms=7.0,
        mean_probability_delta=0.01,
        label_agreement=1.0,
    )
    assert good["accepted"] is True

    bad_quality = edge_acceptance(
        float_bytes=1_000_000,
        int8_bytes=500_000,
        float_latency_ms=10.0,
        int8_latency_ms=5.0,
        mean_probability_delta=0.20,
        label_agreement=0.75,
    )
    assert bad_quality["accepted"] is False

    no_gain = edge_acceptance(
        float_bytes=1_000_000,
        int8_bytes=950_000,
        float_latency_ms=10.0,
        int8_latency_ms=9.8,
        mean_probability_delta=0.01,
        label_agreement=1.0,
    )
    assert no_gain["accepted"] is False


def test_source_families_are_stable():
    rows = [
        {"source": "LIAR:123"},
        {"source": "ClaimReview consensus: a.example, b.example"},
        {"source": "manual:teacher"},
        {"source": ""},
    ]
    assert [source_family(row) for row in rows] == ["liar", "claimreview", "manual", "unknown"]
    assert source_family_counts(rows) == {"liar": 1, "claimreview": 1, "manual": 1, "unknown": 1}


def test_canary_partition_is_persistent_and_excluded(tmp_path: Path):
    rows = []
    for label in (0, 1):
        for i in range(20):
            text = f"persistent canary class {label} sample {i}"
            rows.append({
                "id": f"{label}-{i}",
                "text": text,
                "text_id": text_fingerprint(text),
                "label": label,
                "source": "LIAR:test",
            })
    remaining1, canary1, info1 = ensure_canary_partition(tmp_path, rows, seed=7)
    remaining2, canary2, info2 = ensure_canary_partition(tmp_path, rows, seed=7)
    ids1 = {row["text_id"] for row in canary1}
    ids2 = {row["text_id"] for row in canary2}
    assert ids1 == ids2
    assert ids1.isdisjoint({row["text_id"] for row in remaining1})
    assert len(canary1) >= 4
    assert info1["created_now"] is True
    assert info2["created_now"] is False


def test_canary_gate_blocks_hidden_regression():
    cfg = EvolutionConfig.for_mode("safe")
    old = {"f1": 0.90, "f1_real": 0.92, "f1_fake": 0.88}
    candidate = {"f1": 0.78, "f1_real": 0.90, "f1_fake": 0.66}
    ok, reason = _canary_decision(candidate, old, cfg)
    assert ok is False
    assert "canary" in reason


def test_batch_verified_ingestion_handles_duplicates_and_conflicts(tmp_path: Path):
    path = tmp_path / "verified.jsonl"
    batch = append_verified_many(path, [
        {"text": "Batch sample alpha is verified true", "label": 1},
        {"text": "Batch sample beta is verified false", "label": 0},
        {"text": "Batch sample alpha is verified true", "label": 1},
        {"text": "Batch sample beta is verified false", "label": 1},
        {"text": "x", "label": 1},
    ])
    assert batch["accepted"] == 2
    assert batch["duplicates"] == 1
    assert batch["conflicts"] == 1
    assert batch["invalid"] == 1
    assert len(load_records(path)) == 2


def test_canary_never_grows_from_previously_seen_records(tmp_path: Path):
    rows = []
    for label in (0, 1):
        for i in range(20):
            text = f"initial hidden canary class {label} item {i}"
            rows.append({"text": text, "text_id": text_fingerprint(text), "label": label, "source": "LIAR:test"})
    remaining1, canary1, _ = ensure_canary_partition(tmp_path, rows, seed=11)
    original = {row["text_id"] for row in canary1}

    expanded = list(rows)
    for label in (0, 1):
        for i in range(20, 60):
            text = f"later arriving class {label} item {i}"
            expanded.append({"text": text, "text_id": text_fingerprint(text), "label": label, "source": "ClaimReview consensus: a.example,b.example"})
    remaining2, canary2, info2 = ensure_canary_partition(tmp_path, expanded, seed=11)
    assert {row["text_id"] for row in canary2} == original
    assert info2["created_now"] is False
    assert original.isdisjoint({row["text_id"] for row in remaining2})


def test_persistent_split_assignments_do_not_move_when_data_grows(tmp_path: Path):
    rows = []
    for label in (0, 1):
        for i in range(20):
            text = f"stable split class {label} item {i}"
            rows.append({"text": text, "text_id": text_fingerprint(text), "label": label})
    tr1, va1, te1, manifest1 = persistent_split_records(tmp_path, rows, seed=17)
    first = dict(manifest1["assignments"])

    expanded = list(rows)
    for label in (0, 1):
        for i in range(20, 40):
            text = f"stable split later class {label} item {i}"
            expanded.append({"text": text, "text_id": text_fingerprint(text), "label": label})
    tr2, va2, te2, manifest2 = persistent_split_records(tmp_path, expanded, seed=17)
    for tid, split in first.items():
        assert manifest2["assignments"][tid] == split
    sets = [
        {row["text_id"] for row in tr2},
        {row["text_id"] for row in va2},
        {row["text_id"] for row in te2},
    ]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_stale_dataset_lock_is_recovered(tmp_path: Path):
    import os
    import time
    path = tmp_path / "verified.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    lock.write_text("stale", encoding="utf-8")
    old = time.time() - 300
    os.utime(lock, (old, old))
    row = append_verified(path, {"text": "stale lock recovery sample text", "label": 1})
    assert row["duplicate"] is False
    assert not lock.exists()
