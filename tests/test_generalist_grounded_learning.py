from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.corpus import repository_corpus
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.pretraining import CorpusDocument, pretrain_causal
from generalist_lm.tokenizer import ByteTokenizer


def tiny_config():
    return GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()


def test_repository_corpus_excludes_exams_governance_and_sensitive_files(tmp_path: Path):
    (tmp_path / "src").mkdir()
    token = "ghp_" + "A" * 24
    (tmp_path / "src" / "app.py").write_text(
        ("def useful_function(x):\n    return x + 1\n" * 4)
        + f"\n# leaked-looking token {token}\n",
        encoding="utf-8",
    )

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_exam.py").write_text(
        "EXAM_ANSWER_MARKER = 'never train on me'\n" * 8,
        encoding="utf-8",
    )

    protected = tmp_path / "computer" / "generalist_lm"
    protected.mkdir(parents=True)
    (protected / "benchmarks.py").write_text(
        "PROTECTED_BENCHMARK_MARKER = 'hidden exam'\n" * 8,
        encoding="utf-8",
    )

    (tmp_path / ".env").write_text("TOKEN=should-not-appear\n" * 8, encoding="utf-8")
    (tmp_path / ".ai").mkdir()
    (tmp_path / ".ai" / "private.md").write_text("PRIVATE_STATE_MARKER\n" * 8, encoding="utf-8")

    docs, manifest = repository_corpus(tmp_path, max_bytes=1_000_000, chunk_chars=256)
    joined = "\n".join(row.text for row in docs)

    assert "useful_function" in joined
    assert token not in joined
    assert "[REDACTED]" in joined
    assert "EXAM_ANSWER_MARKER" not in joined
    assert "PROTECTED_BENCHMARK_MARKER" not in joined
    assert "should-not-appear" not in joined
    assert "PRIVATE_STATE_MARKER" not in joined
    assert manifest.excluded_protected >= 2
    assert manifest.files >= 1
    assert manifest.documents >= 1


def test_repository_corpus_is_deterministic_and_skips_symlink_escape(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def alpha():\n    return 'alpha'\n" * 5, encoding="utf-8")
    (repo / "b.md").write_text("# docs\nGrounded model documentation.\n" * 5, encoding="utf-8")

    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_SECRET_MARKER\n" * 8, encoding="utf-8")
    try:
        (repo / "escape.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    first, manifest1 = repository_corpus(repo, chunk_chars=256)
    second, manifest2 = repository_corpus(repo, chunk_chars=256)
    joined = "\n".join(row.text for row in first)

    assert "OUTSIDE_SECRET_MARKER" not in joined
    assert manifest1.digest == manifest2.digest
    assert [row.sha256 for row in first] == [row.sha256 for row in second]


def test_pretraining_evaluation_is_bounded():
    pytest.importorskip("torch")
    docs = [
        CorpusDocument(
            source="repo:demo.py#0",
            text=("def add(a, b):\n    return a + b\n" * 80),
            sha256="a" * 64,
            bytes=3000,
        )
    ]
    model = CausalTransformerLM(tiny_config())
    report = pretrain_causal(
        model,
        ByteTokenizer(),
        docs,
        steps=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=19,
        max_eval_blocks=3,
    )
    assert report["blocks"] > 3
    assert report["eval_blocks"] == 3
    assert report["steps"] == 2
    assert report["ok"] is True


def test_research_cycle_records_grounded_pretraining(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    (repo / "module.py").write_text(
        ("def multiply(a, b):\n    return a * b\n\n"
         "class Example:\n    def value(self):\n        return 42\n") * 12,
        encoding="utf-8",
    )

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CORPUS_ROOT", str(repo))
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_PRETRAIN_STEPS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CORPUS_MAX_BYTES", "100000")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CORPUS_MAX_DOCUMENTS", "32")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_GRADIENT_ACCUMULATION", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_PRECISION", "fp32")

    result = run_research_cycle(state)
    grounded = result["grounded_pretraining"]
    assert grounded["steps"] == 1
    assert grounded["corpus"]["enabled"] is True
    assert grounded["corpus"]["documents"] >= 1
    assert grounded["corpus"]["digest"]
    assert result["champion_report"]["pretraining"]["steps"] == 1
    assert result["policy"]["grounded_pretraining"]["protected_exam_sources_excluded"] is True
