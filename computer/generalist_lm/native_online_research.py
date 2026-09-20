from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


NATIVE_ONLINE_RESEARCH_VERSION = "native-online-research-v1"
_ALLOWED_HOSTS = {"export.arxiv.org", "api.github.com", "api.openalex.org"}
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_TEXT = 4_000

_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "optimizer": (
        "optimizer", "adamw", "learning rate", "weight decay", "warmup",
        "schedule", "gradient clipping",
    ),
    "tokenizer": (
        "tokenizer", "tokenization", "bpe", "byte pair", "vocabulary",
    ),
    "gqa": (
        "grouped query attention", "grouped-query attention", "gqa",
        "multi query attention", "multi-query attention",
    ),
    "long-context": (
        "long context", "long-context", "context length", "rotary",
        "rope", "positional encoding",
    ),
    "curriculum": (
        "curriculum", "data mixture", "data mixing", "sampling weight",
        "domain mixture",
    ),
    "reasoning": (
        "reasoning", "chain of thought", "mathematical reasoning",
        "symbolic reasoning",
    ),
    "code": ("code generation", "program synthesis", "coding"),
    "data-analysis": ("data analysis", "tabular", "data science"),
    "efficiency": (
        "efficient transformer", "efficiency", "memory efficient",
        "kv cache", "throughput", "compute optimal",
    ),
    "dataset": (
        "dataset", "corpus", "training data", "text data", "code data",
    ),
}

_DEFAULT_TOPICS = (
    "language model optimizer training",
    "transformer tokenization efficiency",
    "grouped query attention transformer",
    "long context rotary embedding",
    "curriculum data mixture language model",
    "open permissive text dataset corpus language model",
    "open Italian text corpus dataset",
    "open code dataset permissive license",
)


def _compact_text(value: Any, limit: int = _MAX_TEXT) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[: max(1, int(limit))]


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tags_for(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    tags = [
        tag
        for tag, keywords in _TAG_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return tuple(sorted(set(tags)))


@dataclass(frozen=True)
class NativeOnlineEvidence:
    provider: str
    title: str
    summary: str
    url: str
    published: str | None
    tags: tuple[str, ...]
    source_id: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


def _read_limited(response: Any, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(262_144, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("online research response exceeds byte budget")
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch(
    url: str,
    *,
    token: str | None = None,
    timeout_seconds: float = 20.0,
    max_bytes: int = _MAX_RESPONSE_BYTES,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or host not in _ALLOWED_HOSTS:
        raise PermissionError(f"online research host is not allowed: {host!r}")

    headers = {
        "User-Agent": "AIRI-Native-Research/1.0 (+https://github.com/arancione3000/airi-pc-bootstrap)",
    }
    if host == "export.arxiv.org":
        # arXiv's Atom endpoint can reject broad/mixed Accept headers with
        # HTTP 406. Request the media type it actually serves.
        headers["Accept"] = "application/atom+xml"
    elif host == "api.github.com":
        headers["Accept"] = "application/vnd.github+json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    elif host == "api.openalex.org":
        headers["Accept"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers, method="GET")
    response = (opener or urlopen)(request, timeout=float(timeout_seconds))
    try:
        final_url = response.geturl() if hasattr(response, "geturl") else url
        final = urlparse(final_url)
        final_host = (final.hostname or "").lower()
        if final.scheme.lower() != "https" or final_host not in _ALLOWED_HOSTS:
            raise PermissionError(
                f"online research redirected to unapproved host: {final_host!r}"
            )
        return _read_limited(response, int(max_bytes))
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _evidence(
    *,
    provider: str,
    title: str,
    summary: str,
    url: str,
    published: str | None,
    source_id: str,
) -> NativeOnlineEvidence:
    title = _compact_text(title, 500)
    summary = _compact_text(summary)
    url = _compact_text(url, 2_000)
    source_id = _compact_text(source_id, 500)
    tags = _tags_for(f"{title} {summary}")
    base = {
        "provider": provider,
        "title": title,
        "summary": summary,
        "url": url,
        "published": published,
        "tags": list(tags),
        "source_id": source_id,
    }
    return NativeOnlineEvidence(
        provider=provider,
        title=title,
        summary=summary,
        url=url,
        published=published,
        tags=tags,
        source_id=source_id,
        digest=_sha256_json(base),
    )


def search_arxiv(
    topics: Iterable[str],
    *,
    max_results_per_topic: int = 3,
    timeout_seconds: float = 20.0,
    opener: Callable[..., Any] | None = None,
) -> list[NativeOnlineEvidence]:
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[NativeOnlineEvidence] = []
    for topic in topics:
        query = _compact_text(topic, 160)
        if not query:
            continue
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all%3A{quote_plus(query)}&start=0&"
            f"max_results={max(1, min(int(max_results_per_topic), 10))}&"
            "sortBy=submittedDate&sortOrder=descending"
        )
        blob = _fetch(
            url,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        root = ET.fromstring(blob)
        for entry in root.findall("atom:entry", namespace):
            title = entry.findtext("atom:title", default="", namespaces=namespace)
            summary = entry.findtext("atom:summary", default="", namespaces=namespace)
            source_id = entry.findtext("atom:id", default="", namespaces=namespace)
            published = entry.findtext("atom:published", default="", namespaces=namespace) or None
            link = source_id
            for candidate in entry.findall("atom:link", namespace):
                if candidate.attrib.get("rel") == "alternate" and candidate.attrib.get("href"):
                    link = candidate.attrib["href"]
                    break
            results.append(_evidence(
                provider="arxiv",
                title=title,
                summary=summary,
                url=link,
                published=published,
                source_id=source_id or link,
            ))
    return results


def _openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, raw_positions in inverted_index.items():
        if not isinstance(token, str) or not isinstance(raw_positions, list):
            continue
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except Exception:
                continue
            positions.append((position, token))
    if not positions:
        return ""
    positions.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positions)


def search_openalex(
    topics: Iterable[str],
    *,
    api_key: str | None = None,
    max_results_per_topic: int = 3,
    timeout_seconds: float = 20.0,
    opener: Callable[..., Any] | None = None,
) -> list[NativeOnlineEvidence]:
    """Search OpenAlex as an academic fallback when arXiv is unavailable."""
    results: list[NativeOnlineEvidence] = []
    for topic in topics:
        query = _compact_text(topic, 160)
        if not query:
            continue
        per_page = max(1, min(int(max_results_per_topic), 10))
        key = api_key or os.environ.get("OPENALEX_API_KEY")
        url = (
            "https://api.openalex.org/works?"
            f"search={quote_plus(query)}&per_page={per_page}&sort=-publication_date"
        )
        if key:
            url += f"&api_key={quote_plus(key)}"
        blob = _fetch(
            url,
            token=key,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        payload = json.loads(blob.decode("utf-8"))
        for item in payload.get("results", [])[:per_page]:
            if not isinstance(item, dict):
                continue
            title = _compact_text(
                item.get("display_name") or item.get("title"),
                500,
            )
            source_id = _compact_text(item.get("id"), 500)
            primary_location = (
                item.get("primary_location")
                if isinstance(item.get("primary_location"), dict)
                else {}
            )
            landing_url = _compact_text(
                primary_location.get("landing_page_url") or source_id,
                1_500,
            )
            abstract = _compact_text(
                _openalex_abstract(item.get("abstract_inverted_index")),
                2_500,
            )
            topic_names = " ".join(
                _compact_text(row.get("display_name"), 120)
                for row in item.get("topics", [])
                if isinstance(row, dict) and row.get("display_name")
            )
            open_access = (
                item.get("open_access")
                if isinstance(item.get("open_access"), dict)
                else {}
            )
            summary = _compact_text(
                f"{abstract} topics={topic_names} "
                f"oa={open_access.get('is_oa')} "
                f"oa_status={open_access.get('oa_status')}",
                _MAX_TEXT,
            )
            results.append(_evidence(
                provider="openalex",
                title=title,
                summary=summary,
                url=landing_url or source_id,
                published=_compact_text(item.get("publication_date"), 100) or None,
                source_id=source_id or landing_url,
            ))
    return results


def search_github_repositories(
    topics: Iterable[str],
    *,
    token: str | None = None,
    max_results_per_topic: int = 3,
    timeout_seconds: float = 20.0,
    opener: Callable[..., Any] | None = None,
) -> list[NativeOnlineEvidence]:
    results: list[NativeOnlineEvidence] = []
    for topic in topics:
        query = _compact_text(topic, 160)
        if not query:
            continue
        per_page = max(1, min(int(max_results_per_topic), 10))
        url = (
            "https://api.github.com/search/repositories?"
            f"q={quote_plus(query)}&sort=updated&order=desc&per_page={per_page}"
        )
        blob = _fetch(
            url,
            token=token,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        payload = json.loads(blob.decode("utf-8"))
        for item in payload.get("items", [])[:per_page]:
            if not isinstance(item, dict):
                continue
            full_name = _compact_text(item.get("full_name"), 300)
            html_url = _compact_text(item.get("html_url"), 1_500)
            description = _compact_text(item.get("description"), 2_000)
            license_info = item.get("license") if isinstance(item.get("license"), dict) else {}
            license_key = _compact_text(license_info.get("spdx_id"), 100)
            topics_text = " ".join(
                _compact_text(row, 100)
                for row in item.get("topics", [])
                if isinstance(row, str)
            )
            summary = (
                f"{description} license={license_key or 'unknown'} "
                f"topics={topics_text} stars={int(item.get('stargazers_count') or 0)}"
            )
            results.append(_evidence(
                provider="github",
                title=full_name,
                summary=summary,
                url=html_url,
                published=_compact_text(item.get("updated_at"), 100) or None,
                source_id=full_name or html_url,
            ))
    return results


def topics_from_signals(signals: Iterable[str] | None = None) -> tuple[str, ...]:
    topics = list(_DEFAULT_TOPICS)
    for signal in signals or ():
        key = str(signal).strip().lower()
        if key in {"symbolic_reasoning_signal", "deep_symbolic_signal", "reasoning_gap"}:
            topics.append("mathematical reasoning transformer architecture")
        elif key == "coding_gap":
            topics.append("code language model training curriculum")
        elif key == "data_gap":
            topics.append("data analysis language model training")
        elif key == "tool_gap":
            topics.append("tool use language model training")
        elif key == "tokenizer_efficiency_gap":
            topics.append("tokenizer vocabulary efficiency language model")
        elif key == "research_curriculum_signal":
            topics.append("curriculum learning data mixture language model")
    return tuple(dict.fromkeys(topics))


def discover_native_research(
    *,
    signals: Iterable[str] | None = None,
    github_token: str | None = None,
    max_results_per_topic: int = 2,
    max_evidence: int = 24,
    timeout_seconds: float = 20.0,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Search bounded public research metadata without executing remote content.

    Online text is treated as untrusted evidence only. It may influence which
    *predefined* local mutation families are explored, never inject code,
    commands, file paths, or arbitrary hyperparameter values.
    """
    topics = topics_from_signals(signals)
    evidence: list[NativeOnlineEvidence] = []
    errors: list[dict[str, str]] = []

    arxiv_ok = False
    try:
        arxiv_rows = search_arxiv(
            topics,
            max_results_per_topic=max_results_per_topic,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        evidence.extend(arxiv_rows)
        arxiv_ok = bool(arxiv_rows)
    except Exception as exc:
        errors.append({
            "provider": "arxiv",
            "error": f"{type(exc).__name__}:{exc}",
        })

    # arXiv is useful but occasionally returns service-side 406/429/503
    # responses. OpenAlex is the independent academic metadata fallback so
    # research discovery remains available without trusting GitHub alone.
    if not arxiv_ok:
        try:
            evidence.extend(search_openalex(
                topics,
                api_key=os.environ.get("OPENALEX_API_KEY"),
                max_results_per_topic=max_results_per_topic,
                timeout_seconds=timeout_seconds,
                opener=opener,
            ))
        except Exception as exc:
            errors.append({
                "provider": "openalex",
                "error": f"{type(exc).__name__}:{exc}",
            })

    try:
        evidence.extend(search_github_repositories(
            topics,
            token=github_token or os.environ.get("GITHUB_TOKEN"),
            max_results_per_topic=max_results_per_topic,
            timeout_seconds=timeout_seconds,
            opener=opener,
        ))
    except Exception as exc:
        errors.append({
            "provider": "github",
            "error": f"{type(exc).__name__}:{exc}",
        })

    deduped: list[NativeOnlineEvidence] = []
    seen: set[str] = set()
    for row in evidence:
        if row.digest in seen:
            continue
        seen.add(row.digest)
        deduped.append(row)
        if len(deduped) >= max(1, int(max_evidence)):
            break

    tag_counts: dict[str, int] = {}
    for row in deduped:
        for tag in row.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    payload = [row.to_dict() for row in deduped]
    permissive_markers = (
        "license=apache-2.0",
        "license=mit",
        "license=bsd-2-clause",
        "license=bsd-3-clause",
        "license=cc0-1.0",
        "license=cc-by-4.0",
    )
    source_candidates = [
        {
            "provider": row.provider,
            "source_id": row.source_id,
            "url": row.url,
            "title": row.title,
            "digest": row.digest,
            "status": "proposal_only",
            "admission_requirements": [
                "explicit permissive/public-domain license",
                "immutable content URL or revision",
                "pinned SHA-256",
                "Phase-2 corpus audit",
            ],
        }
        for row in deduped
        if row.provider == "github"
        and "dataset" in row.tags
        and any(marker in row.summary.lower() for marker in permissive_markers)
    ]
    return {
        "ok": bool(deduped),
        "version": NATIVE_ONLINE_RESEARCH_VERSION,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "topics": list(topics),
        "evidence": payload,
        "evidence_digest": _sha256_json(payload),
        "tag_counts": dict(sorted(tag_counts.items())),
        "source_candidates": source_candidates,
        "errors": errors,
        "remote_code_execution": False,
        "remote_content_trusted": False,
        "academic_fallback": "openalex" if any(
            row.provider == "openalex" for row in deduped
        ) else None,
        "policy": (
            "online material is untrusted evidence; only bounded local mutation "
            "families may consume derived tags; OpenAlex is used when arXiv "
            "research metadata is unavailable"
        ),
    }
