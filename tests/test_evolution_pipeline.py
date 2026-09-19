from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution import claimreview, runtime
from evolution.artifacts import report
from evolution.audit import audit_state
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


def test_registrable_domain_collapses_same_organization_subdomains():
    assert claimreview.registrable_domain("www.news.example.com") == "example.com"
    assert claimreview.registrable_domain("factcheck.example.com") == "example.com"
    assert claimreview.registrable_domain("foo.example.co.uk") == "example.co.uk"


def test_consensus_does_not_double_count_same_registrable_domain(monkeypatch):
    claim = "A claim checked by one organization twice"
    mapping = {
        "https://a.example.com/check": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://a.example.com/check"),
        "https://b.example.com/check": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://b.example.com/check"),
    }
    monkeypatch.setattr(claimreview, "fetch_claimreviews", lambda url: mapping[url])
    result = claimreview.verify_consensus(claim, list(mapping), min_sources=2)
    assert result["verified"] is False
    assert result["reason"] == "not enough independent unambiguous ClaimReview sources"


def test_multilingual_verdicts_are_conservative():
    assert claimreview.verdict_to_binary("Vero") == 1
    assert claimreview.verdict_to_binary("Falso") == 0
    assert claimreview.verdict_to_binary("Corretto") == 1
    assert claimreview.verdict_to_binary("Bufala") == 0
    assert claimreview.verdict_to_binary("Parzialmente vero") is None
    assert claimreview.verdict_to_binary("Fuorviante") is None
    assert claimreview.verdict_to_binary("Sin contexto") is None


def test_redirect_to_private_address_is_rejected_before_follow(monkeypatch):
    import urllib.error
    from email.message import Message

    class FakeOpener:
        def open(self, request, timeout=None):
            headers = Message()
            headers["Location"] = "http://127.0.0.1/private"
            raise urllib.error.HTTPError(request.full_url, 302, "Found", headers, None)

    monkeypatch.setattr(claimreview.urllib.request, "build_opener", lambda *args, **kwargs: FakeOpener())
    import pytest
    with pytest.raises(ValueError, match="non-public"):
        claimreview._open_public_url("http://8.8.8.8/start", timeout=1)


def test_self_audit_detects_canary_split_overlap(tmp_path: Path):
    from evolution.data import append_verified_many, ensure_canary_partition, persistent_split_records
    path = tmp_path / "data" / "verified.jsonl"
    records = []
    for label in (0, 1):
        for i in range(12):
            records.append({"text": f"audit class {label} sample number {i}", "label": label})
    append_verified_many(path, records)
    rows = load_records(path)
    remaining, canary, _ = ensure_canary_partition(tmp_path, rows, seed=5)
    persistent_split_records(tmp_path, remaining, seed=5)
    healthy = audit_state(tmp_path)
    assert healthy["ok"] is True

    split_path = tmp_path / "data" / "split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["assignments"][canary[0]["text_id"]] = "test"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    broken = audit_state(tmp_path)
    assert broken["ok"] is False
    assert any("golden-canary" in msg for msg in broken["errors"])


def test_self_audit_detects_malformed_raw_jsonl(tmp_path: Path):
    data = tmp_path / "data" / "verified.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"text":"valid enough sample","label":1}\n{broken\n', encoding="utf-8")
    result = audit_state(tmp_path)
    assert result["ok"] is False
    assert result["checks"]["dataset_malformed_lines"] == 1


def test_runtime_start_blocks_on_hard_audit_error(monkeypatch):
    monkeypatch.setattr(runtime, "status", lambda: {
        "running": False,
        "dataset_records": 40,
        "class_counts": {"fake": 20, "real": 20},
    })
    monkeypatch.setattr(runtime, "audit", lambda: {"ok": False, "errors": ["partition overlap"], "warnings": [], "checks": {}})
    result = runtime.start(auto_setup=False)
    assert result["ok"] is False
    assert result["reason"] == "state_audit_failed"


def test_verified_queue_conflict_with_existing_dataset_is_quarantined(monkeypatch, tmp_path: Path):
    claim = "Existing verified claim must not flip labels"
    from evolution.data import append_verified
    append_verified(tmp_path / "data" / "verified.jsonl", {"text": claim, "label": 1, "source": "manual:test", "evidence": "trusted"})
    mapping = {
        "https://one.example/a": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://one.example/a"),
        "https://two.example/b": claimreview.extract_claimreviews(_claimreview_html(claim, "False"), "https://two.example/b"),
    }
    monkeypatch.setattr(claimreview, "fetch_claimreviews", lambda url: mapping[url])
    queued = queue_claim(tmp_path, claim)
    result = verify_queued_claim(tmp_path, queued["id"], list(mapping), min_sources=2)
    assert result["status"] == "conflict"
    assert "conflicting verified labels" in result["ingest_error"]
    rows = load_records(tmp_path / "data" / "verified.jsonl")
    assert len(rows) == 1 and rows[0]["label"] == 1
