from __future__ import annotations

import json
from pathlib import Path

import pytest

from generalist_lm.architecture_ir import ArchitectureSpec
from generalist_lm.architecture_mutations import (
    next_parameter_tier,
    proposal_set,
    structural_mutations,
)
from generalist_lm.architecture_search import prioritize_architecture_proposals
from generalist_lm.data_quality import assess_text
from generalist_lm.evolution import GeneralistGenome
from generalist_lm.internet_quarantine import (
    SourceProvenance,
    quarantine_records,
)
from generalist_lm.meta_controller import decide_next_action


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
    assert kinds[:2] == ["capacity_plus_structure", "capacity_scale"]
    assert ordered[0]["parameter_estimate"] > parent.parameter_estimate()
    assert ordered[1]["parameter_estimate"] > parent.parameter_estimate()


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
