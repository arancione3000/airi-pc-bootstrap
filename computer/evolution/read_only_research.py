from __future__ import annotations

import re
import urllib.parse
import urllib.request
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
    "text/xml",
)


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
        if href and ("result__a" in css or "result-link" in css):
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
    # _open_public_url enforces http/https only, public-address DNS resolution,
    # redirect revalidation, credential rejection and HTTPS downgrade blocking.
    response, final_url = _open_public_url(url, timeout=timeout)
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


def search_web(query: str, limit: int = 8) -> list[dict[str, str]]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("search query is required")
    limit = max(1, min(MAX_SEARCH_RESULTS, int(limit)))
    search_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    page = _read_public(search_url, timeout=20, max_bytes=MAX_PAGE_BYTES)
    parser = _SearchParser()
    parser.feed(page["text"])

    out: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()
    for item in parser.items:
        url = item["url"]
        try:
            safe_url, domain = _public_http_url(url)
        except Exception:
            continue
        if safe_url in seen_urls:
            continue
        seen_urls.add(safe_url)
        # Discovery deliberately favours source diversity. Additional pages from
        # the same organization can still be supplied explicitly by a caller.
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        out.append({"url": safe_url, "title": item.get("title", ""), "domain": domain})
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
        "method": "read_only_search_discovery",
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
