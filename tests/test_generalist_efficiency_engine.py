from __future__ import annotations

import json
from pathlib import Path

import pytest

from generalist_lm.efficiency_engine import (
    active_learning_weights,
    efficiency_bonus,
    efficiency_profile,
    island_schedule,
    island_weights,
    self_play_policy,
    sparse_expert_plan,
)
from generalist_lm.evolution import GeneralistGenome, compression_candidate
from generalist_lm.model import GeneralistLMConfig
from generalist_lm.specialist_router import classify_specialist, specialist_manifest
from generalist_lm.verified_self_play import generate_verified_self_play_rows


class _SelfPlayBackend:
    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt: str, *, max_new_tokens: int = 80) -> str:
        return self.output


def _healthy_report():
    return {
        "score": 32.0,
        "nll_per_byte": 1.9,
        "generation_similarity": 0.55,
        "generation_exact_accuracy": 0.15,
        "generation_repetition_rate": 0.25,
        "generation_pathological_repetition": False,
        "domain_nll_per_byte": {
            "language": 3.0,
            "coding": 2.0,
            "reasoning": 2.5,
            "tools": 4.0,
            "structured": 3.5,
        },
        "domain_generation_similarity": {
            "language": 0.45,
            "coding": 0.65,
            "reasoning": 0.50,
            "tools": 0.10,
            "structured": 0.20,
        },
    }


def test_active_learning_focuses_measured_weakness_and_tool_gap():
    weights = active_learning_weights(
        _healthy_report(),
        base_weights={"language": 1.0, "coding": 1.0},
        verified_tool_experiences=0,
    )
    assert weights["tools"] == 4.0
    assert weights["structured"] >= 2.5
    assert weights["language"] > weights["coding"]
    assert all(1.0 <= value <= 4.0 for value in weights.values())


def test_island_schedule_and_sparse_expert_plan_are_bounded():
    islands = island_schedule(8, signals=["tool_gap", "language_collapse"])
    assert len(islands) == 8
    assert islands.count("tools") >= 2
    assert islands.count("language") >= 2

    weighted = island_weights({"coding": 1.0, "tools": 1.0}, "coding")
    assert weighted["coding"] == 4.0
    assert weighted["tools"] == 1.0

    plan = sparse_expert_plan(available_islands=islands)
    assert plan["max_active_experts"] == 1
    assert plan["dense_ensemble"] is False
    assert plan["mobile_safe"] is True


def test_compression_candidate_reduces_depth_without_width_reset():
    champion = GeneralistGenome(
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=192,
        tokenizer_version="byte-v1",
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
    ).validate()
    compressed = compression_candidate(champion, vocab_size=264, target_ratio=0.72)
    assert compressed is not None
    assert compressed.d_model == champion.d_model
    assert compressed.d_ff == champion.d_ff
    assert compressed.n_layers < champion.n_layers
    assert compressed.model_config(264).d_model == champion.model_config(264).d_model


def test_efficiency_profile_rewards_same_quality_with_less_compute():
    report = {
        "score": 30.0,
        "nll_per_byte": 2.0,
        "generation_similarity": 0.4,
    }
    small = GeneralistLMConfig(
        vocab_size=264,
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    large = GeneralistLMConfig(
        vocab_size=264,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=384,
    )
    a = efficiency_profile(small, report)
    b = efficiency_profile(large, report)
    assert a["flops_per_token_estimate"] < b["flops_per_token_estimate"]
    assert a["quality_per_million_flops"] > b["quality_per_million_flops"]
    assert 0.0 <= efficiency_bonus(a) <= 2.0


def test_verified_self_play_accepts_only_independently_correct_proposals():
    readiness = {
        "enabled": True,
        "reason": "ready",
    }
    good = _SelfPlayBackend('{"expression":"12*7-4","result":80}')
    rows, report = generate_verified_self_play_rows(
        good,
        readiness,
        cycle=7,
        max_tasks=2,
    )
    assert len(rows) == 1  # duplicate proposal is rejected deterministically
    assert report["accepted"] == 1
    assert "<tool_call>" in rows[0].messages[1]["content"]
    assert '"expression":"12*7-4"' in rows[0].messages[1]["content"]

    bad = _SelfPlayBackend('{"expression":"12*7-4","result":999}')
    rows, report = generate_verified_self_play_rows(
        bad,
        readiness,
        cycle=8,
        max_tasks=1,
    )
    assert rows == []
    assert report["accepted"] == 0
    assert report["rejected"] == 1


def test_self_play_readiness_is_fail_closed():
    policy = self_play_policy(
        {
            "generation_similarity": 0.2,
            "generation_repetition_rate": 0.8,
            "generation_exact_accuracy": 0.0,
            "generation_pathological_repetition": True,
        },
        verified_tool_experiences=0,
    )
    assert policy["enabled"] is False
    assert policy["maximum_generated_tasks_per_cycle"] == 0


def test_specialist_router_is_deterministic_and_manifest_is_research_only(tmp_path: Path):
    assert classify_specialist("Fix this Python unit test") == "coding"
    assert classify_specialist("Search the web for a source") == "tools"
    assert classify_specialist("Prove this logical statement") == "reasoning"
    assert classify_specialist("Ciao, parliamo un po'") == "language"

    folder = tmp_path / "specialists" / "coding"
    folder.mkdir(parents=True)
    (folder / "model.pt").write_bytes(b"placeholder")
    (folder / "summary.json").write_text(
        json.dumps({
            "candidate_id": "c1",
            "research_only": True,
            "production_qualified": False,
        }),
        encoding="utf-8",
    )
    manifest = specialist_manifest(tmp_path)
    assert manifest["max_active_experts"] == 1
    assert manifest["specialists"]["coding"]["candidate_id"] == "c1"


def test_memory_search_tool_reads_only_verified_generalist_experiences(tmp_path: Path, monkeypatch):
    from control_plane import generalist_agent_bridge

    rows = [
        {
            "id": "verified-1",
            "cycle": 5,
            "source": "champion",
            "domain": "tools",
            "messages": [
                {"role": "user", "content": "inspect task lifecycle"},
                {"role": "assistant", "content": "verified safe flow"},
            ],
        },
        {
            "id": "ignored",
            "cycle": 6,
            "source": "x",
            "domain": "language",
            "messages": [
                {"role": "user", "content": "inspect task lifecycle"},
            ],
        },
    ]
    path = tmp_path / "airi-pc-lab-experiences.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STATE", str(tmp_path))

    result = generalist_agent_bridge.execute_readonly_tool(
        "memory_search",
        {"query": "task lifecycle", "limit": 5},
    )
    assert result["verified_only"] is True
    assert result["count"] == 1
    assert result["matches"][0]["id"] == "verified-1"


def test_commercial_audit_fails_closed_without_explicit_license(tmp_path: Path):
    import importlib.util

    source = Path(__file__).resolve().parents[1] / "scripts" / "airi-commercial-audit.py"
    spec = importlib.util.spec_from_file_location("airi_commercial_audit", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    (tmp_path / "computer" / "generalist_lm").mkdir(parents=True)
    (tmp_path / "computer" / "generalist_lm" / "generalist_data_growth.py").write_text(
        "# provenance policy\n",
        encoding="utf-8",
    )
    (tmp_path / "SECURITY.md").write_text("security\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("x\n", encoding="utf-8")
    report = module.audit(tmp_path)
    assert report["commercial_ready"] is False
    assert any("LICENSE" in item for item in report["blockers"])
    assert report["legal_clearance"] is False



def test_runtime_tool_curriculum_covers_readonly_airi_pc_surface():
    from generalist_lm.curriculum import train_rows, validation_rows

    train_targets = "\n".join(
        message.get("content", "")
        for row in train_rows()
        if row.domain == "tools"
        for message in row.messages
        if message.get("role") == "assistant"
    )
    validation_targets = "\n".join(
        message.get("content", "")
        for row in validation_rows()
        if row.domain == "tools"
        for message in row.messages
        if message.get("role") == "assistant"
    )
    for tool in ("calculator", "web_search", "web_read", "file_read", "file_search", "project_analyze", "memory_search"):
        assert f'"name":"{tool}"' in train_targets
    for tool in ("file_read", "file_search", "project_analyze", "memory_search"):
        assert f'"name":"{tool}"' in validation_targets


def test_persisted_specialist_keeps_distillation_gate_evidence(tmp_path: Path):
    from generalist_lm.generalist_swarm import _persist_specialist_checkpoints

    trial = tmp_path / "trial"
    checkpoint = trial / "best-checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"research-checkpoint")
    result_path = trial / "result.json"
    row = {
        "candidate_id": "specialist-tools-1",
        "stage": 3,
        "island": "tools",
        "parameters": 123456,
        "score": 31.0,
        "mean_fitness_score": 31.0,
        "mean_nll_per_byte": 1.9,
        "mean_generation_similarity": 0.55,
        "mean_generation_repetition_rate": 0.20,
        "all_seed_eligible": True,
        "any_seed_eligible": True,
        "any_generation_pathological_repetition": False,
        "worst_domain_regression": 0.02,
        "checkpoint_dir": "best-checkpoint",
        "genome": GeneralistGenome().to_dict(),
    }
    state = tmp_path / "state"
    state.mkdir()
    report = _persist_specialist_checkpoints(
        state,
        [(result_path, row)],
        cycle=9,
    )
    saved = json.loads(
        (state / "specialists" / "tools" / "summary.json").read_text(encoding="utf-8")
    )
    assert report["specialists"]["tools"]["candidate_id"] == "specialist-tools-1"
    assert saved["all_seed_eligible"] is True
    assert saved["any_generation_pathological_repetition"] is False
    assert saved["worst_domain_regression"] == pytest.approx(0.02)
    assert saved["genome"]["genome_id"] == GeneralistGenome().genome_id



def test_unique_candidates_keeps_distinct_learned_weight_lineages():
    from generalist_lm.generalist_swarm import _unique_candidates

    genome = GeneralistGenome().validate()
    rows = [
        ("continual", genome),
        ("language_fusion", genome),
        ("specialist_fusion", genome),
        ("architecture", genome),
    ]
    candidates = _unique_candidates(rows, count=8)
    kinds = [row["kind"] for row in candidates]
    assert "continual" in kinds
    assert "language_fusion" in kinds
    assert "specialist_fusion" in kinds
    # Ordinary architecture duplicate shares the topology-only lineage.
    assert kinds.count("architecture") == 0
