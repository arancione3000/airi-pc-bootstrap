from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any

from .claimreview import _open_public_url, _public_http_url, registrable_domain

MAX_SEARCH_RESULTS = 12
MAX_PAGE_BYTES = 2_000_000
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/atom+xml",
    "text/xml",
)

_MATH_MARKERS = {
    "math", "mathematics", "mathematical", "algebra", "calculus", "geometry",
    "trigonometry", "number theory", "combinatorics", "equation", "inequality",
    "sequence", "polynomial", "sympy", "lean", "mathlib", "theorem", "proof",
    "optimization", "probability", "discrete", "matrix", "linear algebra",
}


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, str]] = []
        self._href = ""
        self._active = False
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        values = {str(k): str(v or "") for k, v in attrs}
        href = values.get("href", "")
        css = values.get("class", "")
        looks_like_result = (
            "result__a" in css
            or "result-link" in css
            or "uddg=" in href
            or href.startswith("//duckduckgo.com/l/")
        )
        if href and looks_like_result:
            self._href = href
            self._active = True
            self._text = []

    def handle_data(self, data):
        if self._active:
            self._text.append(str(data))

    def handle_endtag(self, tag):
        if tag.lower() != "a" or not self._active:
            return
        href = urllib.parse.unquote(self._href or "")
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("uddg"):
            href = urllib.parse.unquote(query["uddg"][0])
        title = re.sub(r"\s+", " ", " ".join(self._text)).strip()
        if href.startswith(("http://", "https://")):
            self.items.append({"url": href, "title": title})
        self._href = ""
        self._active = False
        self._text = []


def _read_public(
    url: str,
    *,
    timeout: int = 20,
    max_bytes: int = MAX_PAGE_BYTES,
) -> dict[str, Any]:
    # Autonomous evidence collection is HTTPS-only. _open_public_url then adds
    # public-address DNS validation, redirect revalidation, credential
    # rejection and HTTPS downgrade blocking.
    safe_url, _domain = _public_http_url(url)
    if urllib.parse.urlparse(safe_url).scheme.lower() != "https":
        raise ValueError("read-only researcher requires HTTPS")
    response, final_url = _open_public_url(safe_url, timeout=timeout)
    with response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not any(kind in content_type for kind in ALLOWED_CONTENT_TYPES):
            raise ValueError(f"read-only researcher rejected content type: {content_type[:120]}")
        payload = response.read(max(1, int(max_bytes)) + 1)
    if len(payload) > max_bytes:
        raise ValueError("read-only researcher response exceeds configured size limit")
    charset = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type)
    if match:
        charset = match.group(1)
    return {
        "url": final_url,
        "content_type": content_type,
        "text": payload.decode(charset, errors="replace"),
        "bytes": len(payload),
    }


def fetch_text(url: str, *, timeout: int = 20, max_bytes: int = MAX_PAGE_BYTES) -> dict[str, Any]:
    return _read_public(url, timeout=timeout, max_bytes=max_bytes)


def _safe_result(url: str, title: str = "") -> dict[str, str] | None:
    try:
        safe_url, domain = _public_http_url(url)
        if urllib.parse.urlparse(safe_url).scheme.lower() != "https":
            return None
    except Exception:
        return None
    return {"url": safe_url, "title": str(title or "").strip(), "domain": domain}


def _duckduckgo_search(query: str, limit: int) -> list[dict[str, str]]:
    search_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    page = _read_public(search_url, timeout=20, max_bytes=MAX_PAGE_BYTES)
    parser = _SearchParser()
    parser.feed(page["text"])
    out = []
    for item in parser.items:
        safe = _safe_result(item["url"], item.get("title", ""))
        if safe:
            out.append(safe)
        if len(out) >= limit:
            break
    return out


def _wikipedia_search(query: str, limit: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "format": "json",
        "utf8": "1",
        "srlimit": max(1, min(10, int(limit))),
        "srsearch": query,
    })
    page = _read_public(
        "https://en.wikipedia.org/w/api.php?" + params,
        timeout=20,
        max_bytes=MAX_PAGE_BYTES,
    )
    payload = json.loads(page["text"])
    rows = ((payload.get("query") or {}).get("search") or [])
    out = []
    for row in rows:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()_-")
        safe = _safe_result(url, title)
        if safe:
            out.append(safe)
        if len(out) >= limit:
            break
    return out


def _arxiv_search(query: str, limit: int) -> list[dict[str, str]]:
    # arXiv is especially useful for the mathematical curriculum. Keep the
    # query bounded and use its read-only Atom API.
    words = re.findall(r"[A-Za-z0-9_-]+", query)[:12]
    if not words:
        return []
    search_expr = " AND ".join(f"all:{word}" for word in words)
    params = urllib.parse.urlencode({
        "search_query": search_expr,
        "start": 0,
        "max_results": max(1, min(6, int(limit))),
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    page = _read_public(
        "https://export.arxiv.org/api/query?" + params,
        timeout=20,
        max_bytes=MAX_PAGE_BYTES,
    )
    root = ET.fromstring(page["text"])
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("atom:entry", ns):
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns)).strip()
        identifier = entry.findtext("atom:id", default="", namespaces=ns).strip()
        if identifier.startswith("http://"):
            identifier = "https://" + identifier[len("http://"):]
        safe = _safe_result(identifier, title)
        if safe:
            out.append(safe)
        if len(out) >= limit:
            break
    return out


def _math_catalog(query: str) -> list[dict[str, str]]:
    lower = query.lower()
    if not any(marker in lower for marker in _MATH_MARKERS):
        return []

    catalog = [
        ("https://docs.sympy.org/latest/index.html", "SymPy documentation"),
        ("https://lean-lang.org/doc/reference/latest/", "Lean language reference"),
        ("https://leanprover-community.github.io/mathlib4_docs/", "Mathlib documentation"),
        ("https://dlmf.nist.gov/", "NIST Digital Library of Mathematical Functions"),
    ]
    if any(word in lower for word in ("sequence", "combinatorics", "number theory", "integer")):
        catalog.append(("https://oeis.org/", "Online Encyclopedia of Integer Sequences"))

    out = []
    for url, title in catalog:
        safe = _safe_result(url, title)
        if safe:
            out.append(safe)
    return out


def search_web(query: str, limit: int = 8) -> list[dict[str, str]]:
    """Discover public HTTPS sources with multiple independent fallbacks.

    Search-engine HTML is best-effort and can change or throttle bots. The
    mathematical agent therefore also uses stable read-only APIs and an
    authoritative documentation catalog. Returned results remain candidate
    evidence only; downstream proof/consensus gates decide what is trustworthy.
    """

    query = str(query or "").strip()
    if not query:
        raise ValueError("search query is required")
    limit = max(1, min(MAX_SEARCH_RESULTS, int(limit)))

    collected: list[dict[str, str]] = []
    providers = [
        lambda: _duckduckgo_search(query, limit),
        lambda: _wikipedia_search(query, limit),
    ]
    if any(marker in query.lower() for marker in _MATH_MARKERS):
        providers.extend([
            lambda: _arxiv_search(query, limit),
            lambda: _math_catalog(query),
        ])

    for provider in providers:
        try:
            collected.extend(provider())
        except Exception:
            # Search provider failure is non-fatal; source diversity is built
            # from every provider that remains reachable.
            continue

    out: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()
    for item in collected:
        safe = _safe_result(item.get("url", ""), item.get("title", ""))
        if not safe:
            continue
        url = safe["url"]
        domain = safe["domain"]
        if url in seen_urls or domain in seen_domains:
            continue
        seen_urls.add(url)
        seen_domains.add(domain)
        out.append(safe)
        if len(out) >= limit:
            break
    return out


def research_claim(claim: str, *, max_sources: int = 8) -> dict[str, Any]:
    claim = str(claim or "").strip()
    if len(claim) < 8:
        raise ValueError("claim is too short")
    max_sources = max(2, min(MAX_SEARCH_RESULTS, int(max_sources)))
    queries = [
        f'"{claim}" fact check',
        f'{claim} factcheck true false',
    ]
    collected: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()

    for query in queries:
        try:
            results = search_web(query, limit=max_sources)
        except Exception as exc:
            errors.append({"query": query, "error": repr(exc)})
            continue
        for item in results:
            url = item["url"]
            domain = item.get("domain") or registrable_domain(urllib.parse.urlparse(url).hostname or "")
            if url in seen_urls or domain in seen_domains:
                continue
            seen_urls.add(url)
            seen_domains.add(domain)
            collected.append(item)
            if len(collected) >= max_sources:
                break
        if len(collected) >= max_sources:
            break

    return {
        "ok": bool(collected),
        "claim": claim,
        "method": "read_only_multi_provider_discovery",
        "contract": {
            "network_methods": ["GET"],
            "credentials": False,
            "writes_to_remote_sites": False,
            "private_network_targets": False,
            "https_downgrade": False,
            "max_response_bytes": MAX_PAGE_BYTES,
        },
        "queries": queries,
        "sources": collected,
        "candidate_urls": [item["url"] for item in collected],
        "errors": errors,
    }
