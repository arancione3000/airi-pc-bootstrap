from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution import claimreview, runtime
from evolution.artifacts import report
from evolution.data import load_records
from evolution.liar import import_tsv
from evolution.pipeline import queue_claim, queue_list, verify_queued_claim


def _claimreview_html(claim: str, verdict: str) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "ClaimReview",
        "claimReviewed": claim,
        "reviewRating": {"@type": "Rating", "alternateName": verdict},
    }
    return '<html><script type="application/ld+json">' + json.dumps(payload) + '</script></html>'


def test_claimreview_parser_accepts_strong_verdicts_and_rejects_ambiguous():
    rows = claimreview.extract_claimreviews(_claimreview_html("The moon is made of rock", "False"), "https://one.example/check")
    assert len(rows) == 1
    assert rows[0]["label"] == 0
    rows = claimreview.extract_claimreviews(_claimreview_html("A claim", "Mostly true"), "https://one.example/check")
    assert rows[0]["label"] is None


def test_claimreview_consensus_requires_independent_domains(monkeypatch):
    claim = "The moon is made of green cheese"
    mapping = {
        "https://one.example/a": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://one.example/a"),
        "https://two.example/b": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://two.example/b"),
    }
    monkeypatch.setattr(claimreview, "fetch_claimreviews", lambda url: mapping[url])
    result = claimreview.verify_consensus(claim, list(mapping), min_sources=2)
    assert result["verified"] is True
    assert result["label"] == 0
    assert result["source_count"] == 2


def test_claimreview_conflict_is_not_ingested(monkeypatch, tmp_path: Path):
    claim = "A disputed claim used in a test"
    mapping = {
        "https://one.example/a": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://one.example/a"),
        "https://two.example/b": claimreview.extract_claimreviews(_claimreview_html(claim, "True"), "https://two.example/b"),
    }
    monkeypatch.setattr(claimreview, "fetch_claimreviews", lambda url: mapping[url])
    row = queue_claim(tmp_path, claim)
    result = verify_queued_claim(tmp_path, row["id"], list(mapping), min_sources=2)
    assert result["status"] == "conflict"
    assert load_records(tmp_path / "data" / "verified.jsonl") == []


def test_queue_verified_claim_is_ingested(monkeypatch, tmp_path: Path):
    claim = "A fact checked claim for the queue"
    mapping = {
        "https://one.example/a": claimreview.extract_claimreviews(_claimreview_html(claim, "True"), "https://one.example/a"),
        "https://two.example/b": claimreview.extract_claimreviews(_claimreview_html(claim, "Correct"), "https://two.example/b"),
    }
    monkeypatch.setattr(claimreview, "fetch_claimreviews", lambda url: mapping[url])
    queued = queue_claim(tmp_path, claim)
    result = verify_queued_claim(tmp_path, queued["id"], list(mapping), min_sources=2)
    assert result["status"] == "verified"
    assert len(load_records(tmp_path / "data" / "verified.jsonl")) == 1
    assert queue_list(tmp_path, status="verified")[0]["id"] == queued["id"]


def test_liar_import_uses_only_unambiguous_labels(tmp_path: Path):
    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "1\ttrue\tA sufficiently long true statement\n"
        "2\tmostly-true\tA sufficiently long mostly true statement\n"
        "3\tfalse\tA sufficiently long false statement\n"
        "4\tpants-fire\tA sufficiently long pants fire statement\n"
        "5\thalf-true\tA sufficiently long half true statement\n"
        "6\tbarely-true\tA sufficiently long barely true statement\n",
        encoding="utf-8",
    )
    out = tmp_path / "verified.jsonl"
    stats = import_tsv(tsv, out, "train", "local-test")
    rows = load_records(out)
    assert stats["accepted"] == 4
    assert stats["ambiguous"] == 2
    assert sorted(row["label"] for row in rows) == [0, 0, 1, 1]


def test_report_works_before_first_champion(tmp_path: Path):
    result = report(tmp_path)
    assert result["ok"] is True
    assert result["dataset_records"] == 0
    assert result["champion"]["metrics"] is None


def test_claimreview_fetch_rejects_local_and_non_http_urls():
    import pytest
    with pytest.raises(ValueError, match="public http/https"):
        claimreview.fetch_claimreviews("file:///etc/passwd")
    with pytest.raises(ValueError, match="non-public"):
        claimreview.fetch_claimreviews("http://127.0.0.1/internal")


def test_runtime_export_is_confined_to_evolution_state(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runtime, "STATE", tmp_path)
    result = runtime.export("/tmp/escape.zip")
    assert result["ok"] is False
    assert "exports directory" in result["error"]
