from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from evolution.read_only_research import fetch_text, search_web


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            value = re.sub(r"\s+", " ", str(data)).strip()
            if value:
                self.parts.append(value)

    def text(self, limit: int = 6000) -> str:
        return " ".join(self.parts)[:limit]


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _authority_hint(domain: str) -> float:
    if domain.endswith((".edu", ".ac.uk")):
        return 0.9
    if domain in {"lean-lang.org", "microsoft.github.io", "docs.python.org"}:
        return 0.95
    if domain.endswith((".org", ".gov")):
        return 0.75
    return 0.55


class ReadOnlyResearcher:
    """Web is a hypothesis/evidence source, never the truth oracle.

    All network access is delegated to the already hardened Airi read-only
    researcher: HTTPS-only GET, no credentials, no private-network targets,
    bounded response sizes and redirect revalidation.
    """

    def search(self, claim: str, max_sources: int = 5) -> dict[str, Any]:
        claim = str(claim or "").strip()
        if len(claim) < 3:
            raise ValueError("research claim is too short")

        results = search_web(claim, limit=max(2, min(10, int(max_sources))))
        sources: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        seen_domains: set[str] = set()

        for item in results:
            url = item["url"]
            domain = item.get("domain") or _domain(url)
            if not domain or domain in seen_domains:
                continue
            seen_domains.add(domain)
            try:
                page = fetch_text(url, timeout=15, max_bytes=1_000_000)
                parser = _TextExtractor()
                if "html" in page.get("content_type", ""):
                    parser.feed(page["text"])
                    excerpt = parser.text()
                else:
                    excerpt = re.sub(r"\s+", " ", page["text"]).strip()[:6000]
                sources.append(
                    {
                        "url": page["url"],
                        "domain": domain,
                        "title": item.get("title", ""),
                        "authority_hint": _authority_hint(domain),
                        "excerpt": excerpt,
                    }
                )
            except Exception as exc:
                errors.append({"url": url, "error": repr(exc)})
            if len(sources) >= max_sources:
                break

        return {
            "ok": bool(sources),
            "claim": claim,
            "status": "evidence_only",
            "sources": sources,
            "errors": errors,
            "contract": {
                "https_only": True,
                "methods": ["GET"],
                "remote_writes": False,
                "credentials": False,
                "private_network": False,
            },
            "truth_policy": "web evidence cannot by itself produce VERIFIED",
        }
