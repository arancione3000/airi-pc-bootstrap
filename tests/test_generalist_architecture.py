from __future__ import annotations

import json
from pathlib import Path

import pytest

from generalist_lm.architecture_ir import ArchitectureSpec
from generalist_lm.architecture_mutations import (
    next_parameter_tier,
    next_parameter_tiers,
    proposal_set,
    structural_mutations,
)
from generalist_lm.architecture_search import (
    _negative_architecture_memory,
    prioritize_architecture_proposals,
)
from generalist_lm.data_quality import assess_text
from generalist_lm.evolution import GeneralistGenome
from generalist_lm.internet_quarantine import (
    SourceProvenance,
    quarantine_records,
)
from generalist_lm.meta_controller import decide_next_action
from generalist_lm.pretraining import CorpusDocument
from generalist_lm.generalist_swarm import (
    GENERALIST_SWARM_VERSION,
    _capacity_budget_multiplier,
    _language_rescue_weights,
    select_survivors,
)


def test_architecture_ir_roundtrips_current_genome():
    genome = GeneralistGenome(
        genome_id="test-parent",
        tokenizer_version="bpe-v1",
        norm_type="rmsnorm",
        position_encoding="rope",
        attention_type="gqa",
        n_kv_heads=2,
        local_attention_window=64,
        local_attention_every=2,
        n_layers=4,
    ).validate()
    spec = ArchitectureSpec.from_genome(genome, target_vocab_size=768)
    restored = spec.to_genome()
    assert restored.attention_type == "gqa"
    assert restored.n_kv_heads == 2
    assert restored.local_attention_window == 64
    assert spec.fingerprint() == ArchitectureSpec.from_dict(spec.to_dict()).fingerprint()


def test_structural_mutations_change_real_topology():
    parent = ArchitectureSpec(
        architecture_id="parent",
        generation=3,
        parent_id=None,
        context_length=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=384,
        tokenizer_version="bpe-v1",
        target_vocab_size=768,
    ).validate()
    mutations = structural_mutations(
        parent,
        signals=["language_collapse", "reasoning_gap"],
    )
    assert mutations
    assert any(row.attention_type == "gqa" for row in mutations)
    assert any(row.local_attention_window > 0 for row in mutations)
    assert all(row.parent_id == "parent" for row in mutations)
    assert len({row.fingerprint() for row in mutations}) == len(mutations)


def test_parameter_ladder_reaches_multi_million_scale():
    assert next_parameter_tier(115_328) == 250_000
    assert next_parameter_tier(500_000) == 1_250_000
    assert next_parameter_tier(1_250_000) == 3_000_000
    assert next_parameter_tier(3_000_000) == 7_000_000
    assert next_parameter_tier(7_000_000) is None


def test_parameter_ladder_exposes_multiple_future_tiers():
    assert next_parameter_tiers(115_328, limit=2) == [250_000, 500_000]
    assert next_parameter_tiers(500_000, limit=3) == [
        1_250_000,
        3_000_000,
        7_000_000,
    ]

    assert next_parameter_tiers(115_328) == [
        250_000,
        500_000,
        1_250_000,
        3_000_000,
        7_000_000,
    ]


def test_capacity_probes_receive_more_training_budget():
    assert _capacity_budget_multiplier(
        "architecture_control",
        candidate_parameters=115_000,
        current_parameters=115_000,
    ) == 1.0
    assert _capacity_budget_multiplier(
        "continual",
        candidate_parameters=115_000,
        current_parameters=1,
    ) == 1.0
    scaled = _capacity_budget_multiplier(
        "architecture_capacity_plus_structure",
        candidate_parameters=240_000,
        current_parameters=115_000,
    )
    assert 1.4 < scaled < 1.5
    assert _capacity_budget_multiplier(
        "architecture_capacity_scale",
        candidate_parameters=7_000_000,
        current_parameters=115_000,
    ) == 2.25


def test_negative_memory_blocks_same_parent_failures_only(tmp_path: Path):
    research = tmp_path / "architecture-research"
    research.mkdir()
    (research / "last-plan.json").write_text(
        json.dumps({
            "architecture_parent": {"fingerprint": "parent-fp"},
        }),
        encoding="utf-8",
    )
    (research / "leaderboard.json").write_text(
        json.dumps({
            "entries": [
                {
                    "fingerprint": "failed-fp",
                    "all_seed_eligible": False,
                },
                {
                    "fingerprint": "eligible-fp",
                    "all_seed_eligible": True,
                },
            ],
        }),
        encoding="utf-8",
    )
    rejected, report = _negative_architecture_memory(
        tmp_path,
        parent_fingerprint="parent-fp",
    )
    assert rejected == {"failed-fp"}
    assert report["count"] == 1

    rejected_other, report_other = _negative_architecture_memory(
        tmp_path,
        parent_fingerprint="different-parent",
    )
    assert rejected_other == set()
    assert report_other["count"] == 0


def test_capacity_proposals_cannot_be_crowded_out_by_micro_mutations():
    parent = ArchitectureSpec(
        architecture_id="tiny-parent",
        generation=2,
        parent_id=None,
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        tokenizer_version="bpe-v1",
        target_vocab_size=384,
    ).validate()
    proposals = proposal_set(
        parent,
        signals=["language_collapse"],
        parameter_cap=7_000_000,
        max_candidates=16,
    )
    ordered = prioritize_architecture_proposals(proposals)
    kinds = [row["kind"] for row in ordered]
    capacity_positions = [
        index
        for index, kind in enumerate(kinds)
        if kind in {"capacity_plus_structure", "capacity_scale"}
    ]
    assert capacity_positions
    first_structural = kinds.index("structural_mutation")
    assert max(capacity_positions) < first_structural
    assert all(
        ordered[index]["parameter_estimate"] > parent.parameter_estimate()
        for index in capacity_positions
    )
    assert len({
        ordered[index]["parameter_estimate"]
        for index in capacity_positions
    }) >= 2


def test_safe_capacity_probe_is_preserved_by_successive_halving(tmp_path: Path):
    rows = [
        {
            "version": GENERALIST_SWARM_VERSION,
            "ok": True,
            "candidate_index": 0,
            "candidate_id": "control",
            "kind": "architecture_control",
            "stage": 1,
            "worst_domain_regression": 0.02,
            "any_generation_pathological_repetition": False,
            "any_seed_eligible": True,
            "mean_generation_accuracy": 0.2,
            "mean_generation_similarity": 0.4,
            "mean_generation_nonempty_rate": 1.0,
            "mean_generation_repetition_rate": 0.2,
            "mean_nll_per_byte": 1.6,
            "parameters": 115_000,
            "score": 20.0,
        },
        {
            "version": GENERALIST_SWARM_VERSION,
            "ok": True,
            "candidate_index": 1,
            "candidate_id": "micro",
            "kind": "architecture_structural_mutation",
            "stage": 1,
            "worst_domain_regression": 0.03,
            "any_generation_pathological_repetition": False,
            "any_seed_eligible": True,
            "mean_generation_accuracy": 0.18,
            "mean_generation_similarity": 0.38,
            "mean_generation_nonempty_rate": 1.0,
            "mean_generation_repetition_rate": 0.21,
            "mean_nll_per_byte": 1.7,
            "parameters": 106_000,
            "score": 19.0,
        },
        {
            "version": GENERALIST_SWARM_VERSION,
            "ok": True,
            "candidate_index": 2,
            "candidate_id": "capacity",
            "kind": "architecture_capacity_plus_structure",
            "stage": 1,
            "worst_domain_regression": 0.10,
            "any_generation_pathological_repetition": False,
            "any_seed_eligible": False,
            "mean_generation_accuracy": 0.05,
            "mean_generation_similarity": 0.12,
            "mean_generation_nonempty_rate": 0.8,
            "mean_generation_repetition_rate": 0.35,
            "mean_nll_per_byte": 2.1,
            "parameters": 240_000,
            "score": 10.0,
        },
    ]
    paths = []
    for row in rows:
        path = tmp_path / f"{row['candidate_index']}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)

    result = select_survivors(
        paths,
        tmp_path / "selection.json",
        survivors=2,
    )
    selected = {row["candidate_id"] for row in result["selected"]}
    assert "control" in selected
    assert "capacity" in selected
    assert result["protected_progressive_scale"] is True


def test_quality_filter_rejects_repetition_without_external_llm():
    bad = "spam " * 100
    report = assess_text(bad)
    assert not report.accepted
    assert "lexical_repetition" in report.reasons or "dominant_word" in report.reasons

    good = assess_text(
        "The small cat sleeps near the window while the morning light reaches the room."
    )
    assert good.accepted
    assert good.score > report.score


def test_quarantine_fails_closed_on_ambiguous_license(tmp_path: Path):
    provenance = SourceProvenance(
        source_id="example",
        url="https://example.org/corpus",
        revision="abc123",
        license="UNKNOWN",
        license_url="https://example.org/license",
        language="en",
        domain="language",
    )
    manifest = quarantine_records(
        [
            "This is a sufficiently long natural sentence for a language corpus.",
            "Another independent sentence describes a quiet room and a sleeping cat.",
        ],
        provenance,
        output_dir=tmp_path,
    )
    assert manifest["admitted_records"] == 0
    assert manifest["training_license_allowed"] is False
    assert manifest["external_model_quality_judge_used"] is False


def test_meta_controller_routes_language_collapse_to_architecture_search(tmp_path: Path):
    (tmp_path / "champion").mkdir()
    (tmp_path / "champion" / "research-metrics.json").write_text(
        json.dumps({
            "parameters": 115_328,
            "generation_repetition_rate": 0.84,
            "generation_pathological_repetition": True,
        }),
        encoding="utf-8",
    )
    (tmp_path / "status.json").write_text(
        json.dumps({
            "cycle": 94,
            "phase5_bootstrap": {
                "minimum_success": False,
                "after": {
                    "pathological_repetition": True,
                    "repetition_rate": 0.84,
                    "language_nll": 3.05,
                },
            },
        }),
        encoding="utf-8",
    )
    decision = decide_next_action(tmp_path)
    assert decision["action"] == "architecture_search"
    assert decision["observations"]["next_parameter_tier"] == 250_000
    assert decision["permissions"]["may_weaken_promotion_gates"] is False


def test_language_rescue_assigns_majority_pretraining_mass_to_natural_text():
    documents = [
        CorpusDocument("general:a", "General prose sentence long enough.", "a", 35, "general"),
        CorpusDocument("language:b", "Italian and English language material.", "b", 38, "language"),
        CorpusDocument("dialogue:c", "Natural human dialogue response example.", "c", 39, "dialogue"),
        CorpusDocument("code:d", "def f(x): return x + 1", "d", 24, "code"),
        CorpusDocument("data:e", "1,2,3,4,5", "e", 9, "data"),
        CorpusDocument("reasoning:f", "If A then B; A, therefore B.", "f", 31, "reasoning"),
    ]
    task, pretrain, report = _language_rescue_weights(
        {
            "language": 3.0,
            "coding": 1.8,
            "data": 2.4,
            "reasoning": 2.2,
            "structured": 1.1,
            "tools": 1.0,
        },
        documents,
        {"language_gap", "language_collapse"},
    )
    assert report["enabled"] is True
    assert task["language"] >= 3.0
    assert pretrain is not None
    natural = sum(
        weight
        for domain, weight in pretrain.items()
        if domain in {"general", "language", "dialogue"}
    )
    assert natural >= 0.67
    assert abs(sum(pretrain.values()) - 1.0) < 1e-9


def test_language_rescue_is_inactive_without_language_signals():
    task, pretrain, report = _language_rescue_weights(
        {"language": 1.2, "coding": 1.5},
        [],
        {"coding_gap"},
    )
    assert task == {"language": 1.2, "coding": 1.5}
    assert pretrain is None
    assert report["enabled"] is False


def test_gqa_local_attention_and_postnorm_pass_real_verifier():
    torch = pytest.importorskip("torch")
    from generalist_lm.architecture_verifier import verify_architecture
    from generalist_lm.model import (
        CausalTransformerLM,
        estimate_parameter_count,
        parameter_count,
    )

    spec = ArchitectureSpec(
        architecture_id="gqa-smoke",
        generation=1,
        parent_id="legacy",
        context_length=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        tokenizer_version="bpe-v1",
        target_vocab_size=384,
        norm_type="rmsnorm",
        norm_placement="post",
        position_encoding="rope",
        ff_variant="swiglu",
        attention_type="gqa",
        n_kv_heads=2,
        local_attention_window=24,
        local_attention_every=2,
        tie_embeddings=False,
    ).validate()
    result = verify_architecture(
        spec,
        vocab_size=384,
        batch_size=1,
        sequence_length=12,
        max_parameters=1_000_000,
    )
    assert result.ok, result.report

    cfg = spec.to_model_config(vocab_size=384)
    model = CausalTransformerLM(cfg)
    assert estimate_parameter_count(cfg) == parameter_count(model)
