from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any

STRONG_REAL = {"true", "correct", "accurate", "verified", "authentic", "fact", "factual"}
STRONG_FAKE = {"false", "incorrect", "fake", "hoax", "fabricated", "pants on fire", "pants fire"}
AMBIGUOUS_MARKERS = {
    "mostly", "partly", "partial", "half", "mixture", "mixed", "misleading",
    "missing context", "needs context", "unproven", "unsupported", "satire",
    "outdated", "in dispute", "uncertain",
}


class JsonLdParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_jsonld = False
        self.buf: list[str] = []
        self.payloads: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "script":
            return
        values = {str(k).lower(): str(v or "") for k, v in attrs}
        if "ld+json" in values.get("type", "").lower():
            self.in_jsonld = True
            self.buf = []

    def handle_endtag(self, tag):
        if tag.lower() == "script" and self.in_jsonld:
            self.payloads.append("".join(self.buf))
            self.in_jsonld = False
            self.buf = []

    def handle_data(self, data):
        if self.in_jsonld:
            self.buf.append(data)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _is_claimreview(obj: dict) -> bool:
    kind = obj.get("@type")
    if isinstance(kind, str):
        return kind.lower() == "claimreview"
    if isinstance(kind, list):
        return any(str(item).lower() == "claimreview" for item in kind)
    return False


def normalize_verdict(value: Any) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower())
    return " ".join(text.split())


def verdict_to_binary(verdict: Any) -> int | None:
    value = normalize_verdict(verdict)
    if not value or any(marker in value for marker in AMBIGUOUS_MARKERS):
        return None
    if value in STRONG_REAL:
        return 1
    if value in STRONG_FAKE:
        return 0
    return None


def extract_claimreviews(html: str, url: str = "") -> list[dict[str, Any]]:
    parser = JsonLdParser()
    parser.feed(html)
    domain = (urllib.parse.urlparse(url).hostname or "").lower()
    results = []
    for payload in parser.payloads:
        try:
            raw = json.loads(payload)
        except Exception:
            continue
        for obj in _walk(raw):
            if not isinstance(obj, dict) or not _is_claimreview(obj):
                continue
            rating = obj.get("reviewRating") or {}
            if not isinstance(rating, dict):
                rating = {}
            verdict = rating.get("alternateName") or rating.get("name") or rating.get("ratingExplanation")
            claim = str(obj.get("claimReviewed") or "").strip()
            if not claim:
                continue
            results.append({
                "claim": claim,
                "verdict": str(verdict or "").strip(),
                "label": verdict_to_binary(verdict),
                "url": url,
                "domain": domain,
                "date_published": obj.get("datePublished"),
            })
    return results


def fetch_claimreviews(url: str, timeout: int = 25, max_bytes: int = 4_000_000) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Airi-PC-NeuroEvolution/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("fact-check page exceeds configured size limit")
    charset = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type)
    if match:
        charset = match.group(1)
    return extract_claimreviews(data.decode(charset, errors="replace"), url=url)


def _canon(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def claim_similarity(left: str, right: str) -> float:
    a, b = _canon(left), _canon(right)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    sa, sb = set(a.split()), set(b.split())
    jac = len(sa & sb) / max(1, len(sa | sb))
    return max(seq, jac)


def verify_consensus(claim: str, urls: list[str], min_sources: int = 2, min_similarity: float = 0.45) -> dict[str, Any]:
    accepted = []
    errors = []
    seen_domains = set()
    for url in urls:
        try:
            rows = fetch_claimreviews(url)
        except Exception as exc:
            errors.append({"url": url, "error": repr(exc)})
            continue
        ranked = sorted(
            ((claim_similarity(claim, row["claim"]), row) for row in rows),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked:
            errors.append({"url": url, "error": "no ClaimReview metadata found"})
            continue
        similarity, best = ranked[0]
        if similarity < min_similarity:
            errors.append({"url": url, "error": "ClaimReview does not match queued claim", "similarity": similarity})
            continue
        if best.get("label") not in (0, 1):
            errors.append({"url": url, "error": "ambiguous ClaimReview verdict", "verdict": best.get("verdict")})
            continue
        domain = best.get("domain") or (urllib.parse.urlparse(url).hostname or "").lower()
        if not domain or domain in seen_domains:
            continue
        seen_domains.add(domain)
        accepted.append({**best, "similarity": similarity})
    labels = {row["label"] for row in accepted}
    if len(labels) > 1:
        return {"ok": False, "verified": False, "reason": "independent ClaimReview sources disagree", "evidence": accepted, "errors": errors}
    if len(accepted) < max(1, int(min_sources)):
        return {"ok": False, "verified": False, "reason": "not enough independent unambiguous ClaimReview sources", "evidence": accepted, "errors": errors}
    label = accepted[0]["label"] if accepted else None
    return {
        "ok": True,
        "verified": True,
        "label": label,
        "source_count": len(accepted),
        "domains": sorted(seen_domains),
        "evidence": accepted,
        "errors": errors,
    }
