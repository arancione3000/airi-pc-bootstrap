from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.evolution import GeneralistGenome, generate_challengers


STRUCTURAL_FIELDS = {
    "norm_type",
    "position_encoding",
    "ff_variant",
    "d_model",
    "d_ff",
    "n_layers",
    "context_length",
    "dropout",
}


def changed_fields(champion, candidate):
    a = champion.to_dict()
    b = candidate.to_dict()
    return {
        key for key in a
        if key not in {"generation", "parent_id", "genome_id"}
        and a[key] != b[key]
    }


def test_directed_signals_do_not_starve_structural_exploration():
    champion = GeneralistGenome().validate()
    challengers = generate_challengers(
        champion,
        signals=[
            "tokenizer_efficiency_gap",
            "coding_gap",
            "data_gap",
            "tool_gap",
            "symbolic_reasoning_signal",
        ],
        count=2,
        exploration_offset=0,
    )
    assert len(challengers) == 2
    assert challengers[0].tokenizer_version == "bpe-v1"
    assert any(changed_fields(champion, row) & STRUCTURAL_FIELDS for row in challengers)


def test_noop_directed_variants_are_not_emitted():
    champion = GeneralistGenome(
        code_adapter=True,
        data_adapter=True,
        retrieval_adapter=True,
        symbolic_adapter=True,
        reasoning_depth=16,
    ).validate()
    challengers = generate_challengers(
        champion,
        signals=["coding_gap", "data_gap", "tool_gap", "symbolic_reasoning_signal"],
        count=4,
    )
    assert challengers
    for row in challengers:
        assert changed_fields(champion, row)


def test_invalid_structural_variant_is_skipped_not_fatal():
    # Head dimension 3 is valid for learned positions but invalid for RoPE.
    champion = GeneralistGenome(
        d_model=96,
        n_heads=32,
        d_ff=192,
        position_encoding="learned",
    ).validate()
    challengers = generate_challengers(
        champion,
        signals=[],
        count=2,
        exploration_offset=1,  # starts from learned -> RoPE structural variant
    )
    assert len(challengers) == 2
    assert all(row.position_encoding != "rope" for row in challengers)


def test_structural_rotation_changes_exploration_across_cycles():
    champion = GeneralistGenome().validate()
    first = generate_challengers(champion, signals=[], count=1, exploration_offset=0)[0]
    second = generate_challengers(champion, signals=[], count=1, exploration_offset=1)[0]
    assert changed_fields(champion, first) != changed_fields(champion, second)
