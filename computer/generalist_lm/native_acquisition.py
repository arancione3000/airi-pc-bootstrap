from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .native_data import NATIVE_CORPUS_DOMAINS, NATIVE_CORPUS_VERSION


NATIVE_SOURCE_CATALOG_VERSION = "native-sources-v1"
_ALLOWED_SOURCE_TYPES = {"owned", "public-domain", "permissive", "reviewed"}
_ALLOWED_SUFFIXES = {
    ".txt", ".md", ".py", ".json", ".jsonl", ".csv", ".rst", ".html", ".htm",
    ".js", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".rs", ".go",
    ".sql", ".yaml", ".yml", ".toml",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class NativeAcquisitionResult:
    catalog: str
    output_dir: str
    manifest: str
    documents: int
    downloaded: int
    reused: int
    total_bytes: int
    corpus_sources_digest: str

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.documents > 0,
            "catalog": self.catalog,
            "output_dir": self.output_dir,
            "manifest": self.manifest,
            "documents": self.documents,
            "downloaded": self.downloaded,
            "reused": self.reused,
            "total_bytes": self.total_bytes,
            "corpus_sources_digest": self.corpus_sources_digest,
            "external_pretrained": False,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def _parse_catalog(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("native source catalog must be a JSON object")
    if raw.get("version") != NATIVE_SOURCE_CATALOG_VERSION:
        raise ValueError("unsupported native source catalog version")
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("native source catalog requires a non-empty sources list")
    return raw


def _validated_source(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"native source row {index} must be an object")
    if row.get("approved_for_training") is not True:
        raise ValueError(f"native source row {index} is not approved_for_training")

    url = str(row.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"native source row {index} requires an https URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError(f"native source row {index} has an unsafe URL")

    digest = str(row.get("sha256", "")).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"native source row {index} requires a pinned sha256")

    filename = str(row.get("filename", "")).strip().replace("\\", "/")
    if not filename:
        filename = Path(parsed.path).name
    candidate = Path(filename)
    if (
        not filename
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.suffix.lower() not in _ALLOWED_SUFFIXES
    ):
        raise ValueError(f"native source row {index} has an unsafe filename")

    domain = str(row.get("domain", "")).strip().lower()
    if domain not in NATIVE_CORPUS_DOMAINS:
        raise ValueError(f"unsupported native source domain: {domain!r}")
    language = str(row.get("language", "")).strip().lower()
    if not language or len(language) > 32:
        raise ValueError(f"invalid native source language at row {index}")
    license_name = str(row.get("license", "")).strip()
    if not license_name:
        raise ValueError(f"native source row {index} requires license metadata")
    source_type = str(row.get("source_type", "")).strip().lower()
    if source_type not in _ALLOWED_SOURCE_TYPES:
        raise ValueError(f"unsupported native source_type: {source_type!r}")
    weight = float(row.get("weight", 1.0))
    if not (0.01 <= weight <= 100.0):
        raise ValueError(f"invalid native source weight at row {index}")

    return {
        "name": str(row.get("name", filename)).strip() or filename,
        "url": url,
        "host": parsed.hostname.lower(),
        "sha256": digest,
        "filename": candidate.as_posix(),
        "domain": domain,
        "language": language,
        "license": license_name,
        "source_type": source_type,
        "weight": weight,
    }


def _read_response_limited(response: Any, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(1_048_576, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("native source exceeds per-file byte budget")
        chunks.append(chunk)
    return b"".join(chunks)


def acquire_native_corpus(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_hosts: Iterable[str] | None = None,
    max_file_bytes: int = 250_000_000,
    max_total_bytes: int = 5_000_000_000,
    timeout_seconds: float = 60.0,
    user_agent: str = "AIRI-Native-Corpus/1.0",
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Materialize a pinned, provenance-governed Native corpus from HTTPS sources.

    This stage fetches data only. It never loads model weights, executes remote
    code, discovers arbitrary URLs, or relaxes the local Native corpus audit.
    Every source must be explicitly approved and pinned by SHA-256.
    """
    catalog = Path(catalog_path).expanduser().resolve(strict=True)
    raw = _parse_catalog(catalog)
    sources = [_validated_source(row, index) for index, row in enumerate(raw["sources"])]

    catalog_hosts = {
        str(host).strip().lower()
        for host in raw.get("allowed_hosts", [])
        if str(host).strip()
    }
    explicit_hosts = {
        str(host).strip().lower()
        for host in (allowed_hosts or ())
        if str(host).strip()
    }
    permitted_hosts = explicit_hosts or catalog_hosts
    if not permitted_hosts:
        permitted_hosts = {row["host"] for row in sources}

    unknown_hosts = sorted({row["host"] for row in sources} - permitted_hosts)
    if unknown_hosts:
        raise PermissionError(f"native source hosts are not allowed: {unknown_hosts}")

    root = Path(output_dir).expanduser().resolve()
    files_root = root / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    fetch = opener or urlopen
    documents: list[dict[str, Any]] = []
    downloaded = 0
    reused = 0
    total_bytes = 0
    seen_digests: set[str] = set()
    seen_destinations: dict[str, str] = {}

    for index, row in enumerate(sources):
        if row["sha256"] in seen_digests:
            continue
        destination = (files_root / row["filename"]).resolve()
        if not _inside(destination, files_root):
            raise PermissionError(f"native source row {index} escapes output root")
        destination_key = str(destination)
        previous_digest = seen_destinations.get(destination_key)
        if previous_digest is not None and previous_digest != row["sha256"]:
            raise ValueError(
                f"native sources map different content to the same filename: {row['filename']}"
            )
        seen_destinations[destination_key] = row["sha256"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        blob: bytes | None = None
        if destination.is_file():
            existing = destination.read_bytes()
            if _sha256_bytes(existing) == row["sha256"]:
                blob = existing
                reused += 1

        if blob is None:
            request = Request(
                row["url"],
                headers={
                    "User-Agent": user_agent,
                    "Accept": "text/plain,application/json,text/html,*/*;q=0.1",
                },
                method="GET",
            )
            response = fetch(request, timeout=float(timeout_seconds))
            try:
                final_url = response.geturl() if hasattr(response, "geturl") else row["url"]
                final = urlparse(final_url)
                if final.scheme.lower() != "https" or not final.hostname:
                    raise PermissionError("native source redirected away from https")
                if final.hostname.lower() not in permitted_hosts:
                    raise PermissionError(
                        f"native source redirected to unapproved host: {final.hostname}"
                    )
                blob = _read_response_limited(response, int(max_file_bytes))
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

            actual = _sha256_bytes(blob)
            if actual != row["sha256"]:
                raise ValueError(
                    f"native source sha256 mismatch for {row['name']}: "
                    f"expected {row['sha256']}, got {actual}"
                )
            temp = destination.with_name(destination.name + ".part")
            temp.write_bytes(blob)
            os.replace(temp, destination)
            downloaded += 1

        size = len(blob)
        if size <= 0:
            raise ValueError(f"native source is empty: {row['name']}")
        if size > int(max_file_bytes):
            raise ValueError(f"native source exceeds per-file byte budget: {row['name']}")
        if total_bytes + size > int(max_total_bytes):
            raise ValueError("native acquisition exceeds total byte budget")
        try:
            blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"native source is not UTF-8 text: {row['name']}") from exc

        total_bytes += size
        seen_digests.add(row["sha256"])
        documents.append({
            "path": destination.relative_to(root).as_posix(),
            "domain": row["domain"],
            "language": row["language"],
            "license": row["license"],
            "source_type": row["source_type"],
            "approved_for_training": True,
            "weight": row["weight"],
            "source_url": row["url"],
            "source_sha256": row["sha256"],
            "source_name": row["name"],
        })

    if not documents:
        raise ValueError("native acquisition produced no trainable documents")

    manifest = root / "native-corpus.json"
    manifest_payload = {
        "version": NATIVE_CORPUS_VERSION,
        "acquisition": {
            "version": NATIVE_SOURCE_CATALOG_VERSION,
            "catalog": str(catalog),
            "catalog_sha256": _sha256_bytes(catalog.read_bytes()),
            "network_discovery": False,
            "external_pretrained": False,
        },
        "documents": documents,
    }
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = NativeAcquisitionResult(
        catalog=str(catalog),
        output_dir=str(root),
        manifest=str(manifest),
        documents=len(documents),
        downloaded=downloaded,
        reused=reused,
        total_bytes=total_bytes,
        corpus_sources_digest=_sha256_bytes(_canonical_json([
            {
                "url": row["source_url"],
                "sha256": row["source_sha256"],
                "domain": row["domain"],
                "language": row["language"],
                "license": row["license"],
            }
            for row in documents
        ])),
    )
    return result.summary()
