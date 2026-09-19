from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from evolution import claimreview, runtime
from evolution.evidence_graph import record_verification, status as evidence_status
from evolution.engine import EvolutionConfig, _promotion_decision
from evolution.pipeline import queue_claim, verify_queued_claim
from evolution import read_only_research


def _claimreview_html(claim: str, verdict: str) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "ClaimReview",
        "claimReviewed": claim,
        "reviewRating": {"@type": "Rating", "alternateName": verdict},
    }
    return '<html><script type="application/ld+json">' + json.dumps(payload) + "</script></html>"


def _verification(label: int, claim: str) -> dict:
    verdict = "True" if label == 1 else "False"
    return {
        "ok": True,
        "verified": True,
        "label": label,
        "source_count": 2,
        "domains": ["one.example", "two.example"],
        "evidence": [
            {
                "claim": claim,
                "verdict": verdict,
                "label": label,
                "domain": "one.example",
                "url": "https://one.example/check",
                "similarity": 0.98,
            },
            {
                "claim": claim,
                "verdict": verdict,
                "label": label,
                "domain": "two.example",
                "url": "https://two.example/check",
                "similarity": 0.96,
            },
        ],
        "errors": [],
    }


def test_evidence_graph_keeps_true_and_plausible_false_claims_separate(tmp_path: Path):
    true_claim = "Australia's capital is Canberra."
    false_claim = "Australia's capital is Sydney."

    true_node = record_verification(tmp_path, true_claim, _verification(1, true_claim))
    false_node = record_verification(tmp_path, false_claim, _verification(0, false_claim))

    assert true_node["decision"] == "verified_true"
    assert true_node["label"] == 1
    assert false_node["decision"] == "verified_false"
    assert false_node["label"] == 0
    assert true_node["evidence_confidence"] > 0.7
    assert false_node["evidence_confidence"] > 0.7

    st = evidence_status(tmp_path)
    assert st["claims"] == 2
    assert st["decisions"] == {"verified_false": 1, "verified_true": 1}


def test_evidence_graph_marks_disagreement_as_conflict(tmp_path: Path):
    claim = "A polished but disputed factual claim"
    result = {
        "ok": False,
        "verified": False,
        "reason": "independent ClaimReview sources disagree",
        "evidence": [
            {"label": 1, "domain": "one.example", "url": "https://one.example/a", "similarity": 0.9},
            {"label": 0, "domain": "two.example", "url": "https://two.example/b", "similarity": 0.9},
        ],
        "errors": [],
    }
    node = record_verification(tmp_path, claim, result)
    assert node["decision"] == "conflict"
    assert node["label"] is None
    assert node["evidence_confidence"] == 0.0


def test_pipeline_records_evidence_graph_for_true_and_false_consensus(monkeypatch, tmp_path: Path):
    cases = [
        ("Australia's capital is Canberra.", "True", 1),
        ("Australia's capital is Sydney.", "False", 0),
    ]
    for idx, (claim, verdict, label) in enumerate(cases):
        urls = [f"https://one{idx}.example/a", f"https://two{idx}.example/b"]
        mapping = {
            urls[0]: claimreview.extract_claimreviews(_claimreview_html(claim, verdict), urls[0]),
            urls[1]: claimreview.extract_claimreviews(_claimreview_html(claim, verdict), urls[1]),
        }
        monkeypatch.setattr(claimreview, "fetch_claimreviews", lambda url, m=mapping: m[url])
        # pipeline imported verify_consensus from claimreview, and that function
        # resolves fetch_claimreviews from its defining module at call time.
        queued = queue_claim(tmp_path, claim)
        result = verify_queued_claim(tmp_path, queued["id"], urls, min_sources=2)
        assert result["status"] == "verified"
        assert result["verification"]["label"] == label
        assert result["evidence_graph"]["label"] == label


def test_read_only_research_rejects_local_non_https_and_private_targets():
    with pytest.raises(ValueError):
        read_only_research.fetch_text("file:///etc/passwd")
    with pytest.raises(ValueError, match="requires HTTPS"):
        read_only_research.fetch_text("http://8.8.8.8/plain")
    with pytest.raises(ValueError):
        read_only_research.fetch_text("http://127.0.0.1/private")


def test_search_parser_unwraps_duckduckgo_result_links():
    parser = read_only_research._SearchParser()
    parser.feed(
        '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ffact">'
        "Example fact check</a>"
    )
    assert parser.items == [
        {"url": "https://example.com/fact", "title": "Example fact check"}
    ]


def test_quality_regression_blocks_a_much_faster_candidate():
    cfg = EvolutionConfig.for_mode("safe")
    old = {
        "f1": 0.90,
        "f1_real": 0.91,
        "f1_fake": 0.89,
        "accuracy": 0.90,
        "brier": 0.08,
        "params": 220_000,
        "latency_ms": 10.0,
    }
    candidate = {
        "f1": 0.899,
        "f1_real": 0.91,
        "f1_fake": 0.888,
        "accuracy": 0.895,
        "brier": 0.10,
        "params": 220_000,
        "latency_ms": 5.0,
        "feasible": True,
    }
    promoted, reason = _promotion_decision(candidate, old, cfg)
    assert promoted is False
    assert "quality/efficiency" in reason or "calibration" in reason


def test_latency_win_is_allowed_when_quality_is_stable():
    cfg = EvolutionConfig.for_mode("safe")
    old = {
        "f1": 0.90,
        "f1_real": 0.91,
        "f1_fake": 0.89,
        "accuracy": 0.90,
        "brier": 0.08,
        "params": 220_000,
        "latency_ms": 10.0,
    }
    candidate = {
        "f1": 0.8995,
        "f1_real": 0.91,
        "f1_fake": 0.889,
        "accuracy": 0.899,
        "brier": 0.083,
        "params": 220_000,
        "latency_ms": 7.0,
        "feasible": True,
    }
    promoted, reason = _promotion_decision(candidate, old, cfg)
    assert promoted is True
    assert "latency" in reason.lower()


def test_research_maintenance_retries_pending_claims_conservatively(monkeypatch):
    rows = [
        {"id": "abc", "claim": "A credible-looking claim to verify", "attempts": []},
    ]
    monkeypatch.setattr(runtime, "queue_items", lambda status_filter=None, limit=100: rows if status_filter == "pending" else [])
    monkeypatch.setattr(
        read_only_research,
        "research_claim",
        lambda claim, max_sources=8: {
            "ok": True,
            "claim": claim,
            "candidate_urls": ["https://one.example/a", "https://two.example/b"],
        },
    )
    monkeypatch.setattr(
        runtime,
        "queue_verify",
        lambda qid, urls, min_sources=2, auto_evolve=True, mode="safe": {
            "id": qid,
            "status": "verified",
            "evidence_graph": {"decision": "verified_false", "label": 0},
        },
    )
    result = runtime.research_maintenance(max_claims=1)
    assert result["ok"] is True
    assert result["read_only_web"] is True
    assert result["processed"][0]["status"] == "verified"


def test_evolution_watchdog_recovers_stale_cloud_heartbeat():
    text = (ROOT / ".github" / "workflows" / "evolution-watchdog.yml").read_text(encoding="utf-8")
    assert "Airi Evolution Watchdog" in text
    assert "cron: '8,38 * * * *'" in text
    assert "MAX_AGE_SECONDS: '4200'" in text
    assert "airi-evolution-state" in text
    assert "actions/workflows/evolution-continuum.yml/dispatches" in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "heartbeat_older_than_70_minutes" in text


def test_evolution_continuum_keeps_hourly_primary_schedule_and_state_heartbeat():
    text = (ROOT / ".github" / "workflows" / "evolution-continuum.yml").read_text(encoding="utf-8")
    assert "cron: '23 * * * *'" in text
    assert "AIRI_CLOUD_MAX_CYCLES_PER_DATASET: '0'" in text
    assert "sync_to_git(force_heartbeat=True)" in text
    assert "commit" in text and "remote_sha" in text


def test_search_web_math_catalog_survives_search_provider_failures(monkeypatch):
    import evolution.read_only_research as research

    monkeypatch.setattr(research, "_duckduckgo_search", lambda query, limit: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(research, "_wikipedia_search", lambda query, limit: [])
    monkeypatch.setattr(research, "_arxiv_search", lambda query, limit: [])

    rows = research.search_web("algebra SymPy Lean theorem proof mathematics", limit=4)
    assert len(rows) >= 3
    assert len({row["domain"] for row in rows}) == len(rows)
    assert all(row["url"].startswith("https://") for row in rows)
    assert any("sympy" in row["domain"] for row in rows)
    assert any("lean-lang.org" in row["domain"] for row in rows)
