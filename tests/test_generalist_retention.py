from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.curriculum import DOMAINS, ResearchRow
from generalist_lm.research_cycle import (
    _domain_balanced_rows,
    _training_rows_for_genome,
    research_seed,
)


def row(domain: str, index: int) -> ResearchRow:
    return ResearchRow(domain, [
        {"role": "user", "content": f"{domain} prompt {index}"},
        {"role": "assistant", "content": f"{domain} answer {index}"},
    ])


def counts(rows):
    return Counter(item.domain for item in rows)


def test_domain_balancer_equalizes_skewed_replay():
    rows = [
        row("language", 0),
        row("coding", 0), row("coding", 1), row("coding", 2), row("coding", 3),
        row("data", 0), row("data", 1),
        row("reasoning", 0), row("reasoning", 1), row("reasoning", 2),
    ]
    balanced = _domain_balanced_rows(rows)
    observed = counts(balanced)
    assert observed == {
        "coding": 4,
        "data": 4,
        "language": 4,
        "reasoning": 4,
    }


def test_base_genome_gets_equal_domain_mass_even_with_skewed_memory():
    replay = []
    replay.extend(row("reasoning", i) for i in range(20))
    replay.extend(row("coding", i) for i in range(12))
    replay.extend(row("language", i) for i in range(2))
    replay.extend(row("data", i) for i in range(3))
    replay.extend(row("tools", i) for i in range(1))
    replay.extend(row("structured", i) for i in range(1))

    genome = research_seed()
    assert not any([
        genome.code_adapter,
        genome.data_adapter,
        genome.retrieval_adapter,
        genome.symbolic_adapter,
    ])
    training = _training_rows_for_genome(genome, replay)
    observed = counts(training)
    assert set(observed) == set(DOMAINS)
    assert len(set(observed.values())) == 1


def test_explicit_focus_gene_is_applied_after_balanced_baseline():
    replay = [row("language", 0)] + [row("reasoning", i) for i in range(10)]
    genome = research_seed()
    genome = type(genome)(**{**genome.to_dict(), "code_adapter": True}).validate()

    training = _training_rows_for_genome(genome, replay)
    observed = counts(training)
    non_coding = [observed[domain] for domain in DOMAINS if domain != "coding"]
    assert len(set(non_coding)) == 1
    assert observed["coding"] > non_coding[0]


def test_balancing_is_deterministic_and_preserves_rows():
    rows = [row("language", 0), row("coding", 0), row("coding", 1)]
    first = _domain_balanced_rows(rows)
    second = _domain_balanced_rows(rows)
    assert first == second
    assert all(item in first for item in rows)
