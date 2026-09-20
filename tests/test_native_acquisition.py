from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


DOMAINS = ("general", "language-it")


class _FakeResponse:
    def __init__(self, blob: bytes, url: str):
        self._blob = blob
        self._offset = 0
        self._url = url
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._blob) - self._offset
        if self._offset >= len(self._blob):
            return b""
        chunk = self._blob[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        self.closed = True


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _write_catalog(tmp_path: Path, sources: list[dict]) -> Path:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps({
            "version": "native-sources-v1",
            "allowed_hosts": ["data.example"],
            "sources": sources,
        }),
        encoding="utf-8",
    )
    return path


def test_native_acquisition_downloads_pinned_sources_and_emits_trainable_manifest(tmp_path: Path):
    from generalist_lm.native_acquisition import acquire_native_corpus
    from generalist_lm.native_data import audit_native_corpus, load_native_corpus

    blobs = {
        "https://data.example/general.txt": b"AIRI native general language corpus.\n" * 4,
        "https://data.example/italiano.txt": "Questo testo italiano appartiene al corpus AIRI.\n".encode("utf-8") * 4,
    }
    catalog = _write_catalog(tmp_path, [
        {
            "name": "general-seed",
            "url": "https://data.example/general.txt",
            "sha256": _sha(blobs["https://data.example/general.txt"]),
            "filename": "general/general.txt",
            "domain": "general",
            "language": "en",
            "license": "CC0-1.0",
            "source_type": "public-domain",
            "approved_for_training": True,
            "weight": 1.0,
        },
        {
            "name": "italian-seed",
            "url": "https://data.example/italiano.txt",
            "sha256": _sha(blobs["https://data.example/italiano.txt"]),
            "filename": "it/italiano.txt",
            "domain": "language-it",
            "language": "it",
            "license": "CC-BY-4.0",
            "source_type": "permissive",
            "approved_for_training": True,
            "weight": 1.25,
        },
    ])

    calls: list[str] = []

    def opener(request, timeout):
        url = request.full_url
        calls.append(url)
        return _FakeResponse(blobs[url], url)

    output = tmp_path / "corpus"
    result = acquire_native_corpus(
        catalog,
        output,
        opener=opener,
        max_file_bytes=1_000_000,
        max_total_bytes=2_000_000,
    )

    assert result["ok"] is True
    assert result["documents"] == 2
    assert result["downloaded"] == 2
    assert result["reused"] == 0
    assert set(calls) == set(blobs)

    manifest = Path(result["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["acquisition"]["network_discovery"] is False
    assert payload["acquisition"]["external_pretrained"] is False
    assert {row["source_url"] for row in payload["documents"]} == set(blobs)

    report = load_native_corpus(manifest, allowed_roots=[output])
    audit = audit_native_corpus(report, required_domains=DOMAINS)
    assert audit["ok"] is True
    assert audit["coverage_ok"] is True

    # A second pass reuses byte-identical pinned files instead of downloading.
    calls.clear()
    reused = acquire_native_corpus(
        catalog,
        output,
        opener=opener,
        max_file_bytes=1_000_000,
        max_total_bytes=2_000_000,
    )
    assert reused["downloaded"] == 0
    assert reused["reused"] == 2
    assert calls == []


def test_native_acquisition_rejects_sha_mismatch(tmp_path: Path):
    from generalist_lm.native_acquisition import acquire_native_corpus

    url = "https://data.example/bad.txt"
    catalog = _write_catalog(tmp_path, [{
        "name": "bad",
        "url": url,
        "sha256": "0" * 64,
        "filename": "bad.txt",
        "domain": "general",
        "language": "en",
        "license": "CC0-1.0",
        "source_type": "public-domain",
        "approved_for_training": True,
    }])

    def opener(request, timeout):
        return _FakeResponse(b"tampered bytes", request.full_url)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        acquire_native_corpus(catalog, tmp_path / "corpus", opener=opener)


def test_native_acquisition_rejects_unapproved_redirect(tmp_path: Path):
    from generalist_lm.native_acquisition import acquire_native_corpus

    blob = b"safe corpus bytes"
    catalog = _write_catalog(tmp_path, [{
        "name": "redirect",
        "url": "https://data.example/file.txt",
        "sha256": _sha(blob),
        "filename": "file.txt",
        "domain": "general",
        "language": "en",
        "license": "CC0-1.0",
        "source_type": "public-domain",
        "approved_for_training": True,
    }])

    def opener(request, timeout):
        return _FakeResponse(blob, "https://evil.example/file.txt")

    with pytest.raises(PermissionError, match="unapproved host"):
        acquire_native_corpus(catalog, tmp_path / "corpus", opener=opener)


def test_native_acquisition_rejects_conflicting_destination_paths(tmp_path: Path):
    from generalist_lm.native_acquisition import acquire_native_corpus

    first = b"first corpus"
    second = b"second corpus"
    catalog = _write_catalog(tmp_path, [
        {
            "name": "first",
            "url": "https://data.example/first.txt",
            "sha256": _sha(first),
            "filename": "same.txt",
            "domain": "general",
            "language": "en",
            "license": "CC0-1.0",
            "source_type": "public-domain",
            "approved_for_training": True,
        },
        {
            "name": "second",
            "url": "https://data.example/second.txt",
            "sha256": _sha(second),
            "filename": "same.txt",
            "domain": "general",
            "language": "en",
            "license": "CC0-1.0",
            "source_type": "public-domain",
            "approved_for_training": True,
        },
    ])

    blobs = {
        "https://data.example/first.txt": first,
        "https://data.example/second.txt": second,
    }

    def opener(request, timeout):
        return _FakeResponse(blobs[request.full_url], request.full_url)

    with pytest.raises(ValueError, match="same filename"):
        acquire_native_corpus(catalog, tmp_path / "corpus", opener=opener)


def test_native_acquisition_catalog_is_fail_closed(tmp_path: Path):
    from generalist_lm.native_acquisition import acquire_native_corpus

    catalog = _write_catalog(tmp_path, [{
        "name": "unpinned",
        "url": "https://data.example/file.txt",
        "sha256": "",
        "filename": "file.txt",
        "domain": "general",
        "language": "en",
        "license": "CC0-1.0",
        "source_type": "public-domain",
        "approved_for_training": True,
    }])

    with pytest.raises(ValueError, match="pinned sha256"):
        acquire_native_corpus(catalog, tmp_path / "corpus", opener=lambda *_a, **_k: None)


def test_cli_exposes_native_corpus_acquire(tmp_path: Path):
    from generalist_lm.cli import parser

    args = parser().parse_args([
        "native-corpus-acquire",
        str(tmp_path / "sources.json"),
        str(tmp_path / "corpus"),
        "--allowed-host",
        "data.example",
    ])
    assert args.cmd == "native-corpus-acquire"
    assert args.allowed_host == ["data.example"]
