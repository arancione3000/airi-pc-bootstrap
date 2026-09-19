from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
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


COMMON_MULTI_LABEL_SUFFIXES = {
    "co.uk", "org.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.nz", "com.br", "com.mx", "co.jp",
    "co.in", "com.sg", "com.tr", "com.cn",
}


def registrable_domain(host: str) -> str:
    host = str(host or "").strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return host
    suffix2 = ".".join(parts[-2:])
    if suffix2 in COMMON_MULTI_LABEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return suffix2


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_public_url(url: str, timeout: int, max_redirects: int = 4):
    opener = urllib.request.build_opener(_NoRedirect)
    current = url
    for _ in range(max(0, int(max_redirects)) + 1):
        safe_url, _domain = _public_http_url(current)
        request = urllib.request.Request(
            safe_url,
            headers={"User-Agent": "Airi-PC-NeuroEvolution/1.0"},
        )
        try:
            response = opener.open(request, timeout=timeout)
            final_url = response.geturl()
            _public_http_url(final_url)
            return response, final_url
        except urllib.error.HTTPError as exc:
            if 300 <= int(exc.code) < 400:
                location = exc.headers.get("Location")
                if not location:
                    raise ValueError("redirect response is missing Location") from exc
                current = urllib.parse.urljoin(safe_url, location)
                _public_http_url(current)
                continue
            raise
    raise ValueError("too many fact-check redirects")


def _public_http_url(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only public http/https fact-check URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("credential-bearing URLs are not allowed")
    host = parsed.hostname.strip(".").lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("local fact-check URLs are not allowed")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"fact-check hostname cannot be resolved: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError("fact-check URL resolves to a non-public address")
    domain = registrable_domain(host)
    return parsed.geturl(), domain


def extract_claimreviews(html: str, url: str = "") -> list[dict[str, Any]]:
    parser = JsonLdParser()
    parser.feed(html)
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    domain = registrable_domain(host)
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
    response, final_url = _open_public_url(url, timeout)
    with response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("fact-check page exceeds configured size limit")
    charset = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type)
    if match:
        charset = match.group(1)
    return extract_claimreviews(data.decode(charset, errors="replace"), url=final_url)


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
        domain = best.get("domain") or registrable_domain(urllib.parse.urlparse(url).hostname or "")
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
