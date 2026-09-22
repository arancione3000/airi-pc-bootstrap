from __future__ import annotations

from dataclasses import dataclass
import bz2
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.request import Request, urlopen

from .phase5_diagnostics import protected_bootstrap_texts
from .pretraining import CorpusDocument
from .training import SFTExample


BOOTSTRAP_DATA_VERSION = "phase5-bootstrap-data-v1"
BOOTSTRAP_REPLAY_VERSION = "phase5-bootstrap-replay-v1"
OASST1_REVISION = "cdc771654ed9e7ad1b4cf4d94ae38272daa48437"

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "tatoeba-en-cc0",
        "url": "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2",
        "source_page": "https://tatoeba.org/en/downloads",
        "license": "CC0-1.0",
        "license_url": "https://en.wiki.tatoeba.org/articles/show/cc0-contributions",
        "revision": "rolling export; immutable after first-run SHA-256 pin",
        "language": "en",
        "domain": "language",
        "format": "tatoeba-basic-bz2",
        "max_download_bytes": 8_000_000,
        "attribution": "Tatoeba (CC0 subset); source retained for provenance.",
    },
    {
        "id": "tatoeba-it-ccby",
        "url": "https://downloads.tatoeba.org/exports/per_language/ita/ita_sentences_detailed.tsv.bz2",
        "source_page": "https://tatoeba.org/en/downloads",
        "license": "CC-BY-2.0-FR",
        "license_url": "https://en.wiki.tatoeba.org/articles/show/using-the-tatoeba-corpus",
        "revision": "rolling export; immutable after first-run SHA-256 pin",
        "language": "it",
        "domain": "language",
        "format": "tatoeba-detailed-bz2",
        "max_download_bytes": 30_000_000,
        "attribution": "Sentence author names are retained in the bootstrap manifest; Tatoeba attribution is required.",
    },
    {
        "id": "oasst1-human",
        "url": (
            "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
            + OASST1_REVISION
            + "/2023-04-12_oasst_ready.messages.jsonl.gz?download=true"
        ),
        "source_page": (
            "https://huggingface.co/datasets/OpenAssistant/oasst1/tree/"
            + OASST1_REVISION
        ),
        "license": "Apache-2.0",
        "license_url": (
            "https://huggingface.co/datasets/OpenAssistant/oasst1/blob/"
            + OASST1_REVISION
            + "/LICENSE"
        ),
        "revision": OASST1_REVISION,
        "language": "en,it",
        "domain": "dialogue",
        "format": "oasst1-ready-jsonl-gz",
        "max_download_bytes": 45_000_000,
        "attribution": "OpenAssistant/OASST1, Apache-2.0. Synthetic rows are excluded.",
    },
)

STREAMING_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "fineweb2-it",
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "ita_Latn",
        "revision": "main",
        "language": "it",
        "domain": "general",
        "license": "ODC-By-1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "source_page": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        "attribution": "HuggingFaceFW/FineWeb2 (Italian filtered subset), ODC-By 1.0.",
    },
    {
        "id": "fineweb-en",
        "dataset": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "revision": "main",
        "language": "en",
        "domain": "general",
        "license": "ODC-By-1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "source_page": "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
        "attribution": "HuggingFaceFW/FineWeb English sample, ODC-By 1.0; CommonCrawl terms also apply.",
    },
)


_URL_RE = re.compile(r"https?://", re.IGNORECASE)


@dataclass
class BootstrapDataBundle:
    train_documents: list[CorpusDocument]
    validation_documents: list[CorpusDocument]
    sft_train: list[SFTExample]
    sft_validation: list[SFTExample]
    manifest: dict[str, Any]


@dataclass
class BootstrapReplayBundle:
    documents: list[CorpusDocument]
    sft_train: list[SFTExample]
    manifest: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_score(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


def _normal(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def _download(source: dict[str, Any], cache_dir: Path) -> tuple[bytes, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_id = str(source["id"])
    suffix = ".bz2" if str(source["url"]).split("?", 1)[0].endswith(".bz2") else ".gz"
    target = cache_dir / f"{source_id}{suffix}"
    meta_path = cache_dir / f"{source_id}.download.json"
    if target.is_file() and meta_path.is_file():
        raw = target.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _sha256_bytes(raw) == str(meta.get("sha256") or ""):
            return raw, meta

    request = Request(
        str(source["url"]),
        headers={
            "User-Agent": "AIRI-Generalist-Phase5/1.0 (+https://github.com/arancione3000/airi-pc-bootstrap)",
            "Accept-Encoding": "identity",
        },
    )
    limit = int(source.get("max_download_bytes", 50_000_000))
    chunks: list[bytes] = []
    total = 0
    headers: dict[str, str] = {}
    with urlopen(request, timeout=90) as response:
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RuntimeError(f"bootstrap source exceeded byte cap: {source_id}")
            chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        raise RuntimeError(f"bootstrap source was empty: {source_id}")
    digest = _sha256_bytes(raw)
    target.write_bytes(raw)
    meta = {
        "source_id": source_id,
        "url": source["url"],
        "sha256": digest,
        "download_bytes": len(raw),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return raw, meta


def _quality_sentence(text: str) -> bool:
    text = str(text).strip()
    if len(text) < 2 or len(text) > 420:
        return False
    if "\x00" in text or _URL_RE.search(text):
        return False
    printable = sum(ch.isprintable() for ch in text)
    if printable / max(1, len(text)) < 0.98:
        return False
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return False
    alpha = sum(ch.isalpha() for ch in visible)
    return alpha / len(visible) >= 0.45



def _web_chunks(text: str, *, max_chars: int = 1800) -> list[str]:
    """Turn long web documents into bounded coherent training documents."""
    cleaned = str(text).replace("\r", "\n").strip()
    if not cleaned:
        return []
    paragraphs = [
        re.sub(r"\\s+", " ", part).strip()
        for part in re.split(r"\\n\\s*\\n+", cleaned)
        if part.strip()
    ]
    out: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            out.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    out.append(current)
                    current = ""
                for start in range(0, len(sentence), max_chars):
                    out.append(sentence[start:start + max_chars].strip())
                continue
            candidate = sentence if not current else current + " " + sentence
            if len(candidate) > max_chars:
                out.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            out.append(current)
    return [row for row in out if row]


def _quality_web_text(text: str) -> bool:
    text = str(text).strip()
    if len(text) < 80 or len(text) > 1800 or "\x00" in text:
        return False
    printable = sum(ch.isprintable() for ch in text)
    if printable / max(1, len(text)) < 0.98:
        return False
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return False
    alpha = sum(ch.isalpha() for ch in visible)
    if alpha / len(visible) < 0.45:
        return False
    return _normal(text) not in protected_bootstrap_texts()


def _streaming_cache_path(
    source: dict[str, Any],
    tokenizer,
    token_quota: int,
    cache_dir: Path,
) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(source["id"]))
    tok = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(tokenizer.version))
    return cache_dir / (
        f"{safe}-{tok}-v{int(tokenizer.vocab_size)}-{int(token_quota)}.jsonl.gz"
    )


def _load_or_stream_hf_documents(
    source: dict[str, Any],
    *,
    tokenizer,
    token_quota: int,
    cache_dir: Path,
) -> tuple[list[CorpusDocument], int]:
    """Stream a deterministic bounded FineWeb sample and persist it locally.

    The cache is intentionally source-text only.  Model weights are never
    imported; the live AIRI checkpoint is trained on these documents directly.
    """
    quota = max(1, int(token_quota))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _streaming_cache_path(source, tokenizer, quota, cache_dir)

    documents: list[CorpusDocument] = []
    tokens = 0
    if cache_path.is_file():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = str(row["text"])
                needed = len(tokenizer.encode(text)) + 1
                documents.append(CorpusDocument(
                    source=str(row["source"]),
                    text=text,
                    sha256=str(row["sha256"]),
                    bytes=len(text.encode("utf-8")),
                    domain=str(source.get("domain") or "general"),
                ))
                tokens += needed
        if tokens >= int(quota * 0.95):
            return documents, tokens
        documents.clear()
        tokens = 0

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "FineWeb fast-track requires the 'datasets' package"
        ) from exc

    stream = load_dataset(
        str(source["dataset"]),
        name=str(source["config"]),
        split="train",
        streaming=True,
        revision=str(source.get("revision") or "main"),
    )
    try:
        stream = stream.shuffle(
            seed=271828 if source["language"] == "it" else 314159,
            buffer_size=10_000,
        )
    except Exception:
        pass

    seen: set[str] = set()
    rows_for_cache: list[dict[str, str]] = []
    for row_index, row in enumerate(stream):
        raw_text = str((row or {}).get("text") or "").strip()
        if not raw_text:
            continue
        row_id = str((row or {}).get("id") or row_index)
        for chunk_index, chunk in enumerate(_web_chunks(raw_text)):
            if not _quality_web_text(chunk):
                continue
            digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            source_name = f"{source['id']}:{row_id}:{chunk_index}"
            documents.append(CorpusDocument(
                source=source_name,
                text=chunk,
                sha256=digest,
                bytes=len(chunk.encode("utf-8")),
                domain=str(source.get("domain") or "general"),
            ))
            rows_for_cache.append({
                "source": source_name,
                "text": chunk,
                "sha256": digest,
            })
            tokens += len(tokenizer.encode(chunk)) + 1
            if tokens >= quota:
                break
        if tokens >= quota:
            break

    if tokens < int(quota * 0.95):
        raise RuntimeError(
            f"streamed source {source['id']} supplied only {tokens} tokens "
            f"for requested quota {quota}"
        )

    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        for row in rows_for_cache:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return documents, tokens

def _parse_tatoeba(
    source: dict[str, Any],
    raw: bytes,
) -> list[tuple[str, str, str | None]]:
    detailed = str(source["format"]) == "tatoeba-detailed-bz2"
    decoded = bz2.decompress(raw).decode("utf-8", errors="replace")
    protected = protected_bootstrap_texts()
    out: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for line in decoded.splitlines():
        cols = line.split("\t")
        if len(cols) < (4 if detailed else 3):
            continue
        sentence_id = cols[0].strip()
        text = cols[2].strip()
        author = cols[3].strip() if detailed and len(cols) > 3 else None
        norm = _normal(text)
        if not _quality_sentence(text) or norm in protected:
            continue
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append((sentence_id, text, author or None))
    out.sort(key=lambda row: _stable_score(str(source["id"]), row[0]))
    return out


def _parse_oasst(raw: bytes) -> dict[str, dict[str, Any]]:
    protected = protected_bootstrap_texts()
    rows: dict[str, dict[str, Any]] = {}
    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw), mode="rb") as handle:
        for binary in handle:
            try:
                row = json.loads(binary.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if bool(row.get("deleted")) or bool(row.get("synthetic")):
                continue
            if row.get("review_result") is False:
                continue
            lang = str(row.get("lang") or "").lower()
            if lang not in {"en", "it"}:
                continue
            role = str(row.get("role") or "").lower()
            if role not in {"prompter", "assistant"}:
                continue
            text = str(row.get("text") or "").strip()
            if len(text) < 2 or len(text) > 1800 or "\x00" in text:
                continue
            if _normal(text) in protected:
                continue
            message_id = str(row.get("message_id") or "").strip()
            if not message_id:
                continue
            rows[message_id] = {
                "message_id": message_id,
                "parent_id": str(row.get("parent_id") or "").strip() or None,
                "text": text,
                "role": role,
                "lang": lang,
                "review_count": int(row.get("review_count") or 0),
                "rank": int(row.get("rank") or 0),
            }
    return rows


def _oasst_conversations(rows: dict[str, dict[str, Any]]) -> list[tuple[str, list[dict[str, str]], str]]:
    out: list[tuple[str, list[dict[str, str]], str]] = []
    for message_id, row in rows.items():
        if row["role"] != "assistant":
            continue
        chain: list[dict[str, Any]] = []
        cursor: dict[str, Any] | None = row
        visited: set[str] = set()
        while cursor is not None and len(chain) < 6:
            cid = str(cursor["message_id"])
            if cid in visited:
                chain = []
                break
            visited.add(cid)
            chain.append(cursor)
            parent = cursor.get("parent_id")
            cursor = rows.get(str(parent)) if parent else None
        if len(chain) < 2:
            continue
        chain.reverse()
        if chain[-1]["role"] != "assistant" or chain[-2]["role"] != "prompter":
            continue
        messages: list[dict[str, str]] = []
        expected = "prompter"
        valid = True
        for item in chain:
            if item["role"] != expected:
                valid = False
                break
            messages.append({
                "role": "user" if item["role"] == "prompter" else "assistant",
                "content": item["text"],
            })
            expected = "assistant" if expected == "prompter" else "prompter"
        if not valid or messages[-1]["role"] != "assistant":
            continue
        out.append((message_id, messages, str(row["lang"])))
    out.sort(key=lambda row: _stable_score("oasst1-conversation", row[0]))
    return out


def _take_documents(
    *,
    source_id: str,
    language: str,
    domain: str,
    rows: Iterable[tuple[str, str, str | None]],
    tokenizer,
    token_quota: int,
) -> tuple[list[CorpusDocument], int, list[str]]:
    documents: list[CorpusDocument] = []
    tokens = 0
    authors: set[str] = set()
    for item_id, text, author in rows:
        ids = tokenizer.encode(text)
        if not ids:
            continue
        needed = len(ids) + 1
        if documents and tokens + needed > token_quota:
            break
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        documents.append(
            CorpusDocument(
                source=f"{source_id}:{item_id}:{language}",
                text=text,
                sha256=digest,
                bytes=len(text.encode("utf-8")),
                domain=domain,
            )
        )
        tokens += needed
        if author:
            authors.add(author)
        if tokens >= token_quota:
            break
    return documents, tokens, sorted(authors)


def _split_documents(documents: list[CorpusDocument]) -> tuple[list[CorpusDocument], list[CorpusDocument]]:
    train: list[CorpusDocument] = []
    validation: list[CorpusDocument] = []
    for row in documents:
        bucket = int(hashlib.sha256(row.sha256.encode("ascii")).hexdigest()[:8], 16) % 20
        (validation if bucket == 0 else train).append(row)
    if documents and not validation:
        validation.append(train.pop())
    return train, validation


def _split_sft(rows: list[SFTExample]) -> tuple[list[SFTExample], list[SFTExample]]:
    train: list[SFTExample] = []
    validation: list[SFTExample] = []
    for row in rows:
        raw = json.dumps(row.messages, ensure_ascii=False, sort_keys=True)
        bucket = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16) % 20
        (validation if bucket == 0 else train).append(row)
    if rows and not validation:
        validation.append(train.pop())
    return train, validation


def _replay_bucket(document: CorpusDocument) -> str:
    source = str(document.source)
    if source.startswith("tatoeba-en-"):
        return "language-en"
    if source.startswith("tatoeba-it-"):
        return "language-it"
    if source.startswith("oasst1-human:"):
        return "dialogue"
    return str(getattr(document, "domain", "general") or "general")


def _bounded_replay_documents(
    documents: list[CorpusDocument],
    tokenizer,
    *,
    max_tokens: int,
) -> tuple[list[CorpusDocument], dict[str, int]]:
    """Keep a deterministic bilingual/dialogue replay without holdout rows."""
    max_tokens = max(10_000, int(max_tokens))
    target = {
        "language-en": int(max_tokens * 0.4),
        "language-it": int(max_tokens * 0.4),
        "dialogue": max_tokens - int(max_tokens * 0.8),
    }
    grouped: dict[str, list[CorpusDocument]] = {}
    for document in documents:
        grouped.setdefault(_replay_bucket(document), []).append(document)

    selected: list[CorpusDocument] = []
    counts: dict[str, int] = {}
    selected_hashes: set[str] = set()
    for bucket in ("language-en", "language-it", "dialogue"):
        quota = target[bucket]
        used = 0
        rows = sorted(
            grouped.get(bucket, []),
            key=lambda row: _stable_score("bootstrap-replay", row.sha256),
        )
        for row in rows:
            needed = len(tokenizer.encode(row.text)) + 1
            if selected and used + needed > quota:
                continue
            if row.sha256 in selected_hashes:
                continue
            selected.append(row)
            selected_hashes.add(row.sha256)
            used += needed
            if used >= quota:
                break
        counts[bucket] = used

    # If one language bucket is sparse, fill remaining budget deterministically
    # from any already-approved training document without crossing holdouts.
    used_total = sum(counts.values())
    if used_total < max_tokens:
        fallback = sorted(
            documents,
            key=lambda row: _stable_score("bootstrap-replay-fallback", row.sha256),
        )
        for row in fallback:
            if row.sha256 in selected_hashes:
                continue
            needed = len(tokenizer.encode(row.text)) + 1
            if used_total + needed > max_tokens:
                continue
            selected.append(row)
            selected_hashes.add(row.sha256)
            used_total += needed
            bucket = _replay_bucket(row)
            counts[bucket] = counts.get(bucket, 0) + needed
            if used_total >= max_tokens:
                break
    return selected, counts


def _gzip_jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ).encode("utf-8")
    if payload:
        payload += b"\n"
    return gzip.compress(payload, compresslevel=9, mtime=0)


def write_bootstrap_replay(
    bundle: BootstrapDataBundle,
    tokenizer,
    *,
    output_dir: str | Path,
    max_tokens: int = 1_000_000,
    max_sft_conversations: int = 512,
) -> dict[str, Any]:
    """Persist a bounded immutable training-only replay for future descendants."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    documents, token_counts = _bounded_replay_documents(
        list(bundle.train_documents),
        tokenizer,
        max_tokens=max_tokens,
    )
    sft_rows = list(bundle.sft_train)[: max(0, int(max_sft_conversations))]

    corpus_rows = [
        {
            "source": row.source,
            "text": row.text,
            "sha256": row.sha256,
            "bytes": int(row.bytes),
            "domain": str(row.domain),
        }
        for row in documents
    ]
    sft_payload = [{"messages": row.messages} for row in sft_rows]

    corpus_bytes = _gzip_jsonl(corpus_rows)
    sft_bytes = _gzip_jsonl(sft_payload)
    corpus_path = root / "replay-corpus.jsonl.gz"
    sft_path = root / "replay-sft.jsonl.gz"
    corpus_path.write_bytes(corpus_bytes)
    sft_path.write_bytes(sft_bytes)

    manifest = {
        "schema": 1,
        "version": BOOTSTRAP_REPLAY_VERSION,
        "source_manifest_sha256": str(
            bundle.manifest.get("manifest_content_sha256") or ""
        ),
        "held_out_phase5_suite_excluded": True,
        "validation_documents_excluded": True,
        "sft_validation_excluded": True,
        "user_private_data_used": False,
        "external_model_distillation_used": False,
        "documents": len(documents),
        "sft_conversations": len(sft_rows),
        "selected_tokens": int(sum(token_counts.values())),
        "token_counts_by_bucket": dict(sorted(token_counts.items())),
        "corpus_file": corpus_path.name,
        "corpus_sha256": _sha256_bytes(corpus_bytes),
        "sft_file": sft_path.name,
        "sft_sha256": _sha256_bytes(sft_bytes),
    }
    manifest_path = root / "replay-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_bootstrap_replay(
    replay_dir: str | Path,
) -> BootstrapReplayBundle:
    root = Path(replay_dir).expanduser().resolve()
    manifest_path = root / "replay-manifest.json"
    if not manifest_path.is_file():
        return BootstrapReplayBundle([], [], {
            "version": BOOTSTRAP_REPLAY_VERSION,
            "available": False,
        })
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("version")) != BOOTSTRAP_REPLAY_VERSION:
        raise RuntimeError("unsupported bootstrap replay version")
    for key in (
        "held_out_phase5_suite_excluded",
        "validation_documents_excluded",
        "sft_validation_excluded",
    ):
        if manifest.get(key) is not True:
            raise RuntimeError(f"bootstrap replay fail-closed invariant missing: {key}")

    corpus_path = root / str(manifest["corpus_file"])
    sft_path = root / str(manifest["sft_file"])
    corpus_raw = corpus_path.read_bytes()
    sft_raw = sft_path.read_bytes()
    if _sha256_bytes(corpus_raw) != str(manifest.get("corpus_sha256")):
        raise RuntimeError("bootstrap replay corpus digest mismatch")
    if _sha256_bytes(sft_raw) != str(manifest.get("sft_sha256")):
        raise RuntimeError("bootstrap replay SFT digest mismatch")

    documents: list[CorpusDocument] = []
    for line in gzip.decompress(corpus_raw).decode("utf-8").splitlines():
        row = json.loads(line)
        text = str(row["text"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != str(row["sha256"]):
            raise RuntimeError("bootstrap replay document digest mismatch")
        documents.append(CorpusDocument(
            source=str(row["source"]),
            text=text,
            sha256=digest,
            bytes=int(row["bytes"]),
            domain=str(row.get("domain") or "language"),
        ))

    sft_train: list[SFTExample] = []
    for line in gzip.decompress(sft_raw).decode("utf-8").splitlines():
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("bootstrap replay SFT row is malformed")
        sft_train.append(SFTExample(messages))

    out_manifest = dict(manifest)
    out_manifest["available"] = True
    return BootstrapReplayBundle(documents, sft_train, out_manifest)


def build_bootstrap_bundle(
    tokenizer,
    *,
    cache_dir: str | Path,
    target_tokens: int = 1_000_000,
    previous_manifest: dict[str, Any] | None = None,
) -> BootstrapDataBundle:
    target_tokens = max(100_000, int(target_tokens))
    cache = Path(cache_dir).expanduser().resolve()
    previous_sources = {
        str(row.get("id")): row
        for row in (previous_manifest or {}).get("sources", [])
        if isinstance(row, dict)
    }

    source_records: list[dict[str, Any]] = []
    all_documents: list[CorpusDocument] = []
    language_counts = {"en": 0, "it": 0}
    domain_counts = {
        "language": 0,
        "dialogue": 0,
        "general": 0,
        "coding": 0,
        "reasoning": 0,
        "data": 0,
        "structured": 0,
        "tools": 0,
    }
    attribution_authors: set[str] = set()

    tatoeba_sources = [row for row in SOURCES if str(row["id"]).startswith("tatoeba-")]
    per_tatoeba = int(target_tokens * 0.42)
    for source in tatoeba_sources:
        raw, download = _download(source, cache)
        previous = previous_sources.get(str(source["id"]))
        previous_sha = str((previous or {}).get("sha256") or "")
        if previous_sha and previous_sha != download["sha256"]:
            raise RuntimeError(
                f"source digest changed after manifest pin: {source['id']} "
                f"{previous_sha} != {download['sha256']}"
            )
        parsed = _parse_tatoeba(source, raw)
        docs, imported_tokens, authors = _take_documents(
            source_id=str(source["id"]),
            language=str(source["language"]),
            domain="language",
            rows=parsed,
            tokenizer=tokenizer,
            token_quota=per_tatoeba,
        )
        all_documents.extend(docs)
        language_counts[str(source["language"])] += imported_tokens
        domain_counts["language"] += imported_tokens
        attribution_authors.update(authors)
        source_records.append({
            "id": source["id"],
            "url": source["url"],
            "source_page": source["source_page"],
            "revision": source["revision"],
            "sha256": download["sha256"],
            "download_bytes": download["download_bytes"],
            "etag": download.get("etag"),
            "last_modified": download.get("last_modified"),
            "license": source["license"],
            "license_url": source["license_url"],
            "attribution": source["attribution"],
            "language": source["language"],
            "domain": source["domain"],
            "imported_documents": len(docs),
            "imported_tokens": imported_tokens,
        })

    oasst_source = next(row for row in SOURCES if row["id"] == "oasst1-human")
    raw, download = _download(oasst_source, cache)
    previous = previous_sources.get("oasst1-human")
    previous_sha = str((previous or {}).get("sha256") or "")
    if previous_sha and previous_sha != download["sha256"]:
        raise RuntimeError("pinned OASST1 digest changed unexpectedly")
    oasst_rows = _parse_oasst(raw)
    conversations = _oasst_conversations(oasst_rows)

    oasst_quota = max(1, target_tokens - sum(language_counts.values()))
    oasst_docs: list[CorpusDocument] = []
    oasst_tokens = 0
    oasst_by_language = {"en": 0, "it": 0}
    sft_rows: list[SFTExample] = []
    for message_id, messages, lang in conversations:
        target_text = messages[-1]["content"]
        ids = tokenizer.encode(target_text)
        if not ids:
            continue
        needed = len(ids) + 1
        if oasst_docs and oasst_tokens + needed > oasst_quota:
            break
        oasst_docs.append(
            CorpusDocument(
                source=f"oasst1-human:{message_id}:{lang}",
                text=target_text,
                sha256=hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
                bytes=len(target_text.encode("utf-8")),
                domain="dialogue",
            )
        )
        sft_rows.append(SFTExample(messages))
        oasst_tokens += needed
        oasst_by_language[lang] += needed
        if oasst_tokens >= oasst_quota:
            break
    all_documents.extend(oasst_docs)
    for lang, value in oasst_by_language.items():
        language_counts[lang] += value
    domain_counts["dialogue"] += oasst_tokens
    source_records.append({
        "id": oasst_source["id"],
        "url": oasst_source["url"],
        "source_page": oasst_source["source_page"],
        "revision": oasst_source["revision"],
        "sha256": download["sha256"],
        "download_bytes": download["download_bytes"],
        "etag": download.get("etag"),
        "last_modified": download.get("last_modified"),
        "license": oasst_source["license"],
        "license_url": oasst_source["license_url"],
        "attribution": oasst_source["attribution"],
        "language": oasst_source["language"],
        "domain": oasst_source["domain"],
        "synthetic_rows_excluded": True,
        "deleted_rows_excluded": True,
        "negative_review_rows_excluded": True,
        "imported_documents": len(oasst_docs),
        "imported_sft_conversations": len(sft_rows),
        "imported_tokens": oasst_tokens,
        "imported_tokens_by_language": oasst_by_language,
    })


    # The original pinned sources provide roughly five million useful tokens.
    # Only the 100M fast-track rung asks for more unique text; fill that bounded
    # deficit with already-cleaned/deduplicated FineWeb data while preserving
    # the same AIRI model lineage and keeping Italian slightly dominant.
    remaining = max(0, int(target_tokens) - int(sum(language_counts.values())))
    if target_tokens > 5_000_000 and remaining > 0:
        quotas = {
            "it": max(1, int(round(remaining * 0.60))),
            "en": max(1, remaining - int(round(remaining * 0.60))),
        }
        for source in STREAMING_SOURCES:
            language = str(source["language"])
            quota = int(quotas[language])
            documents, imported_tokens = _load_or_stream_hf_documents(
                source,
                tokenizer=tokenizer,
                token_quota=quota,
                cache_dir=cache,
            )
            all_documents.extend(documents)
            language_counts[language] += imported_tokens
            domain_counts["general"] += imported_tokens
            source_records.append({
                "id": source["id"],
                "dataset": source["dataset"],
                "config": source["config"],
                "revision": source["revision"],
                "source_page": source["source_page"],
                "license": source["license"],
                "license_url": source["license_url"],
                "attribution": source["attribution"],
                "language": source["language"],
                "domain": source["domain"],
                "streaming": True,
                "cached_locally": True,
                "imported_documents": len(documents),
                "imported_tokens": imported_tokens,
            })

    train_docs, val_docs = _split_documents(all_documents)
    sft_train, sft_validation = _split_sft(sft_rows)

    manifest = {
        "schema": 1,
        "version": BOOTSTRAP_DATA_VERSION,
        "target_tokens": target_tokens,
        "actual_selected_tokens": int(sum(language_counts.values())),
        "tokenizer_version": str(tokenizer.version),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "held_out_phase5_suite_excluded": True,
        "user_private_data_used": False,
        "external_pretrained_weights_used": False,
        "external_model_distillation_used": False,
        "sources": source_records,
        "token_counts_by_language": {
            "english": int(language_counts["en"]),
            "italian": int(language_counts["it"]),
        },
        "token_counts_by_domain": {
            key: int(value) for key, value in sorted(domain_counts.items())
        },
        "training_documents": len(train_docs),
        "validation_documents": len(val_docs),
        "sft_training_conversations": len(sft_train),
        "sft_validation_conversations": len(sft_validation),
        "attribution": {
            "tatoeba": "Some bootstrap sentences are from Tatoeba (https://tatoeba.org), CC-BY 2.0 FR unless explicitly marked CC0.",
            "italian_contributor_count": len(attribution_authors),
            "italian_contributors": sorted(attribution_authors),
            "oasst1": "OpenAssistant/OASST1, Apache-2.0, pinned revision " + OASST1_REVISION,
            "fineweb": "HuggingFaceFW FineWeb/FineWeb2, ODC-By 1.0; CommonCrawl terms apply.",
        },
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return BootstrapDataBundle(
        train_documents=train_docs,
        validation_documents=val_docs,
        sft_train=sft_train,
        sft_validation=sft_validation,
        manifest=manifest,
    )
