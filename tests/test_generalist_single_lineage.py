from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from generalist_lm.evolution import GeneralistGenome
from generalist_lm.lineage_migration import (
    active_lineage_snapshot,
    adaptive_architecture_parameter_cap,
    adopt_verified_descendant,
    migrate_live_lineage,
)
from generalist_lm.runtime import GeneralistRuntime


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _tiny_genome(*, genome_id: str = "live-airi") -> GeneralistGenome:
    return GeneralistGenome(
        generation=3,
        parent_id="previous-airi",
        genome_id=genome_id,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()


def _checkpoint(
    path: Path,
    genome: GeneralistGenome,
    *,
    role: str,
    lineage_parent_model_sha256: str | None = None,
) -> GeneralistRuntime:
    runtime = GeneralistRuntime.fresh(genome.model_config(), device="cpu")
    runtime.save_checkpoint(
        path,
        metadata={
            "role": role,
            "production_qualified": role == "production_champion",
            "lineage_parent_model_sha256": lineage_parent_model_sha256,
        },
    )
    return runtime


def _model_sha(path: Path) -> str:
    return hashlib.sha256((path / "model.pt").read_bytes()).hexdigest()


def _bootstrap_state(root: Path) -> tuple[GeneralistGenome, Path]:
    genome = _tiny_genome()
    _checkpoint(root / "champion", genome, role="research_champion")
    _write_json(root / "champion-genome.json", genome.to_dict())

    candidate = root / "bootstrap-data" / "candidate"
    _checkpoint(candidate, genome, role="phase5_language_bootstrap_candidate")
    _write_json(
        root / "bootstrap-data" / "progress.json",
        {
            "schema": 1,
            "version": "test",
            "base_champion_model_sha256": "test-anchor",
            "target_tokens": 100_000_000,
            "tokens_processed": 25_250_000,
            "steps": 1234,
            "completed_rungs": [1_000_000, 5_000_000, 20_000_000],
            "sft_completed_rungs": [1_000_000, 5_000_000, 20_000_000],
            "capacity_genome": genome.to_dict(),
        },
    )
    (root / "bootstrap-data" / "optimizer.pt").write_bytes(b"optimizer-layout-is-stale")
    _write_json(root / "status.json", {"cycle": 42})
    return genome, candidate


def test_active_lineage_prefers_unfinished_bootstrap_candidate(tmp_path: Path):
    genome, candidate = _bootstrap_state(tmp_path)

    live = active_lineage_snapshot(tmp_path)

    assert live["checkpoint"] == candidate
    assert live["bootstrap_active"] is True
    assert live["tokens_processed"] == 25_250_000
    assert live["target_tokens"] == 100_000_000
    assert live["genome"].genome_id == genome.genome_id


def test_parameter_cap_never_falls_below_live_lineage(tmp_path: Path):
    _bootstrap_state(tmp_path)
    live = active_lineage_snapshot(tmp_path)

    cap = adaptive_architecture_parameter_cap(tmp_path, 10_000)

    assert cap >= live["parameters"]


def test_completed_rung_candidate_remains_the_live_learning_lineage(tmp_path: Path):
    genome, candidate = _bootstrap_state(tmp_path)
    progress_path = tmp_path / "bootstrap-data" / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["tokens_processed"] = 100_000_000
    progress["completed_rungs"].append(100_000_000)
    _write_json(progress_path, progress)

    live = active_lineage_snapshot(tmp_path)

    assert live["checkpoint"] == candidate
    assert live["bootstrap_active"] is True
    assert live["tokens_processed"] == 100_000_000
    assert live["genome"].genome_id == genome.genome_id


def test_architecture_migration_preserves_cumulative_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    genome, candidate = _bootstrap_state(tmp_path)
    research = tmp_path / "research-winner"
    target_genome = _tiny_genome(genome_id="verified-new-topology")
    _checkpoint(research, target_genome, role="architecture_research_candidate")

    # This test isolates the transactional lineage mechanics from benchmark
    # quality. The benchmark functions have their own suites elsewhere.
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_lineage_runtime",
        lambda runtime, cycle: {"finite": True, "parameters": 1, "score": 1.0},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration._research_eligible",
        lambda *args, **kwargs: (True, "retention gate passed"),
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_phase5_language",
        lambda runtime: {"pathological_repetition": False},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.degeneration_gate",
        lambda *args, **kwargs: (True, "degeneration gate passed"),
    )

    before = json.loads(
        (tmp_path / "bootstrap-data" / "progress.json").read_text(encoding="utf-8")
    )

    result = migrate_live_lineage(
        tmp_path,
        target_checkpoint=research,
        target_genome=target_genome,
        cycle=43,
        candidate_id="architecture-winner-43",
        research_kind="architecture_control",
    )

    after = json.loads(
        (tmp_path / "bootstrap-data" / "progress.json").read_text(encoding="utf-8")
    )
    lineage = json.loads((tmp_path / "lineage.json").read_text(encoding="utf-8"))

    assert result["accepted"] is True
    assert result["token_progress_preserved"] is True
    assert result["tokens_processed_before"] == before["tokens_processed"]
    assert result["tokens_processed_after"] == before["tokens_processed"]
    assert after["tokens_processed"] == before["tokens_processed"]
    assert after["steps"] == before["steps"]
    assert after["target_tokens"] == before["target_tokens"]
    assert after["lineage_id"] == lineage["lineage_id"]
    assert after["architecture_migrations"][-1]["tokens_processed"] == before["tokens_processed"]
    assert not (tmp_path / "bootstrap-data" / "optimizer.pt").exists()
    assert (candidate / "model.pt").is_file()
    assert lineage["single_active_model"] is True
    assert lineage["active_checkpoint"] == "bootstrap-data/candidate"
    assert lineage["rollback_snapshots_are_not_competing_models"] is True

    rollback = tmp_path / result["rollback_checkpoint"]
    assert (rollback / "model.pt").is_file()
    assert (rollback / "config.json").is_file()


def test_rejected_architecture_never_replaces_live_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    genome, candidate = _bootstrap_state(tmp_path)
    research = tmp_path / "research-winner"
    target_genome = _tiny_genome(genome_id="rejected-topology")
    _checkpoint(research, target_genome, role="architecture_research_candidate")

    model_before = (candidate / "model.pt").read_bytes()
    progress_before = (tmp_path / "bootstrap-data" / "progress.json").read_bytes()

    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_lineage_runtime",
        lambda runtime, cycle: {"finite": True, "parameters": 1, "score": 1.0},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration._research_eligible",
        lambda *args, **kwargs: (False, "retention regression"),
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_phase5_language",
        lambda runtime: {"pathological_repetition": False},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.degeneration_gate",
        lambda *args, **kwargs: (True, "ok"),
    )

    result = migrate_live_lineage(
        tmp_path,
        target_checkpoint=research,
        target_genome=target_genome,
        cycle=44,
        candidate_id="rejected-44",
    )

    assert result["accepted"] is False
    assert (candidate / "model.pt").read_bytes() == model_before
    assert (tmp_path / "bootstrap-data" / "progress.json").read_bytes() == progress_before
    assert not (tmp_path / "lineage.json").exists()


def test_verified_descendant_keeps_new_weights_and_training_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    genome, candidate = _bootstrap_state(tmp_path)
    parent_sha = _model_sha(candidate)

    descendant_genome = _tiny_genome(genome_id="trained-descendant")
    descendant = tmp_path / "trained-descendant"
    runtime = GeneralistRuntime.fresh(descendant_genome.model_config(), device="cpu")
    with torch.no_grad():
        marker = next(runtime.model.parameters())
        marker.fill_(0.123456)
    runtime.save_checkpoint(
        descendant,
        metadata={
            "role": "free_speed_candidate",
            "lineage_parent_model_sha256": parent_sha,
            "lineage_id": "test-lineage",
        },
    )

    before_progress = json.loads(
        (tmp_path / "bootstrap-data" / "progress.json").read_text(encoding="utf-8")
    )
    descendant_state = torch.load(
        descendant / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_lineage_runtime",
        lambda runtime, cycle: {"finite": True, "parameters": 1, "score": 1.0},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration._research_eligible",
        lambda *args, **kwargs: (True, "retention passed"),
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_phase5_language",
        lambda runtime: {"pathological_repetition": False},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.degeneration_gate",
        lambda *args, **kwargs: (True, "degeneration passed"),
    )

    result = adopt_verified_descendant(
        tmp_path,
        candidate_checkpoint=descendant,
        candidate_genome=descendant_genome,
        cycle=45,
        candidate_id="trained-descendant",
        research_kind="reasoning",
        expected_parent_model_sha256=parent_sha,
    )

    after_progress = json.loads(
        (tmp_path / "bootstrap-data" / "progress.json").read_text(encoding="utf-8")
    )
    active_state = torch.load(
        candidate / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert result["accepted"] is True
    assert result["parent_lock_passed"] is True
    assert result["candidate_weights_preserved"] is True
    assert after_progress["tokens_processed"] == before_progress["tokens_processed"]
    assert after_progress["steps"] == before_progress["steps"]
    assert active_state.keys() == descendant_state.keys()
    for key in active_state:
        assert torch.equal(active_state[key], descendant_state[key])


def test_wrong_parent_hash_cannot_replace_live_airi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    genome, candidate = _bootstrap_state(tmp_path)
    descendant_genome = _tiny_genome(genome_id="stale-descendant")
    descendant = tmp_path / "stale-descendant"
    _checkpoint(
        descendant,
        descendant_genome,
        role="free_speed_candidate",
        lineage_parent_model_sha256="deadbeef",
    )

    model_before = (candidate / "model.pt").read_bytes()
    progress_before = (tmp_path / "bootstrap-data" / "progress.json").read_bytes()

    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_lineage_runtime",
        lambda runtime, cycle: {"finite": True, "parameters": 1, "score": 1.0},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration._research_eligible",
        lambda *args, **kwargs: (True, "retention passed"),
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.evaluate_phase5_language",
        lambda runtime: {"pathological_repetition": False},
    )
    monkeypatch.setattr(
        "generalist_lm.lineage_migration.degeneration_gate",
        lambda *args, **kwargs: (True, "degeneration passed"),
    )

    result = adopt_verified_descendant(
        tmp_path,
        candidate_checkpoint=descendant,
        candidate_genome=descendant_genome,
        cycle=46,
        candidate_id="stale-descendant",
        expected_parent_model_sha256="deadbeef",
    )

    assert result["accepted"] is False
    assert result["parent_lock_passed"] is False
    assert (candidate / "model.pt").read_bytes() == model_before
    assert (tmp_path / "bootstrap-data" / "progress.json").read_bytes() == progress_before
