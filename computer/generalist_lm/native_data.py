from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .bpe_tokenizer import BPETokenizer, train_bpe
from .tokenizer import EOS, PAD


NATIVE_CORPUS_VERSION = "native-corpus-v1"
NATIVE_CORPUS_DOMAINS = (
    "general",
    "language-en",
    "language-it",
    "code",
    "math",
    "reasoning",
    "data-analysis",
    "tool-use",
)
_ALLOWED_SOURCE_TYPES = {
    "owned",
    "public-domain",
    "permissive",
    "reviewed",
}
_ALLOWED_SUFFIXES = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".jsonl",
    ".csv",
    ".rst",
    ".html",
    ".htm",
    ".js",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rs",
    ".go",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
}


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class NativeCorpusDocument:
    source: str
    domain: str
    language: str
    license: str
    source_type: str
    weight: float
    sha256: str
    bytes: int
    text: str

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "domain": self.domain,
            "language": self.language,
            "license": self.license,
            "source_type": self.source_type,
            "weight": self.weight,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class NativeCorpusReport:
    manifest: str
    version: str
    documents: tuple[NativeCorpusDocument, ...]
    skipped: tuple[dict[str, Any], ...]
    total_bytes: int
    corpus_digest: str
    domains: dict[str, int]
    languages: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "ok": bool(self.documents),
            "manifest": self.manifest,
            "version": self.version,
            "documents": len(self.documents),
            "skipped": list(self.skipped),
            "total_bytes": self.total_bytes,
            "corpus_digest": self.corpus_digest,
            "domains": dict(self.domains),
            "languages": dict(self.languages),
        }


@dataclass(frozen=True)
class NativeTokenBlock:
    ids: tuple[int, ...]
    domain: str
    weight: float
    source_sha256: str


def _parse_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("native corpus manifest must be a JSON object")
    if raw.get("version") != NATIVE_CORPUS_VERSION:
        raise ValueError("unsupported native corpus manifest version")
    documents = raw.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("native corpus manifest requires a non-empty documents list")
    return raw


def load_native_corpus(
    manifest_path: str | Path,
    *,
    allowed_roots: Iterable[str | Path],
    max_file_bytes: int = 25_000_000,
    max_total_bytes: int = 2_000_000_000,
    max_documents: int = 500_000,
) -> NativeCorpusReport:
    """Load an explicitly approved local AIRI corpus manifest.

    The loader is local-only, rejects path/symlink escapes, requires every
    document to carry licensing/provenance metadata, deduplicates exact content,
    and never downloads or invokes an external model.
    """
    roots = [Path(row).expanduser().resolve() for row in allowed_roots]
    if not roots:
        raise ValueError("at least one allowed corpus root is required")

    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if not _inside(manifest, roots):
        raise PermissionError("native corpus manifest escapes allowed roots")
    raw = _parse_manifest(manifest)

    documents: list[NativeCorpusDocument] = []
    skipped: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    total_bytes = 0

    for index, row in enumerate(raw["documents"]):
        if len(documents) >= max(1, int(max_documents)):
            skipped.append({"index": index, "reason": "document_budget"})
            break
        if not isinstance(row, dict):
            raise ValueError(f"native corpus row {index} must be an object")
        if row.get("approved_for_training") is not True:
            raise ValueError(f"native corpus row {index} is not approved_for_training")

        relative = str(row.get("path", "")).strip()
        if not relative:
            raise ValueError(f"native corpus row {index} has no path")
        if "://" in relative:
            raise ValueError("native corpus manifest paths must be local")
        candidate = Path(relative).expanduser()
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise FileNotFoundError(f"native corpus row {index} path is missing: {relative}") from exc
        if not _inside(resolved, roots):
            raise PermissionError(f"native corpus row {index} escapes allowed roots")
        if resolved.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ValueError(f"unsupported native corpus suffix: {resolved.suffix}")

        domain = str(row.get("domain", "")).strip().lower()
        if domain not in NATIVE_CORPUS_DOMAINS:
            raise ValueError(f"unsupported native corpus domain: {domain!r}")
        language = str(row.get("language", "")).strip().lower()
        if not language or len(language) > 32:
            raise ValueError(f"invalid native corpus language at row {index}")
        license_name = str(row.get("license", "")).strip()
        if not license_name:
            raise ValueError(f"native corpus row {index} requires license metadata")
        source_type = str(row.get("source_type", "")).strip().lower()
        if source_type not in _ALLOWED_SOURCE_TYPES:
            raise ValueError(f"unsupported native corpus source_type: {source_type!r}")
        weight = float(row.get("weight", 1.0))
        if not (0.01 <= weight <= 100.0):
            raise ValueError(f"invalid native corpus weight at row {index}")

        size = int(resolved.stat().st_size)
        if size <= 0:
            skipped.append({"path": str(resolved), "reason": "empty"})
            continue
        if size > int(max_file_bytes):
            skipped.append({"path": str(resolved), "reason": "file_budget"})
            continue
        if total_bytes + size > int(max_total_bytes):
            skipped.append({"path": str(resolved), "reason": "total_budget"})
            break

        blob = resolved.read_bytes()
        text = blob.decode("utf-8", errors="replace").strip()
        if not text:
            skipped.append({"path": str(resolved), "reason": "empty_text"})
            continue
        digest = _sha256_bytes(blob)
        if digest in seen_digests:
            skipped.append({"path": str(resolved), "reason": "duplicate"})
            continue
        seen_digests.add(digest)
        total_bytes += len(blob)
        documents.append(
            NativeCorpusDocument(
                source=str(resolved),
                domain=domain,
                language=language,
                license=license_name,
                source_type=source_type,
                weight=weight,
                sha256=digest,
                bytes=len(blob),
                text=text,
            )
        )

    if not documents:
        raise ValueError("native corpus contains no trainable documents")

    metadata = [row.metadata() for row in documents]
    digest = _sha256_bytes(_canonical_json({
        "version": NATIVE_CORPUS_VERSION,
        "documents": metadata,
    }))
    domains: dict[str, int] = {}
    languages: dict[str, int] = {}
    for row in documents:
        domains[row.domain] = domains.get(row.domain, 0) + 1
        languages[row.language] = languages.get(row.language, 0) + 1

    return NativeCorpusReport(
        manifest=str(manifest),
        version=NATIVE_CORPUS_VERSION,
        documents=tuple(documents),
        skipped=tuple(skipped),
        total_bytes=total_bytes,
        corpus_digest=digest,
        domains=domains,
        languages=languages,
    )


def audit_native_corpus(
    report: NativeCorpusReport,
    *,
    required_domains: Iterable[str] = NATIVE_CORPUS_DOMAINS,
) -> dict[str, Any]:
    required = tuple(dict.fromkeys(str(row).strip().lower() for row in required_domains))
    unknown = sorted(set(required) - set(NATIVE_CORPUS_DOMAINS))
    if unknown:
        raise ValueError(f"unknown required native corpus domains: {unknown}")
    missing = [domain for domain in required if report.domains.get(domain, 0) <= 0]
    return {
        **report.summary(),
        "required_domains": list(required),
        "missing_domains": missing,
        "coverage_ok": not missing,
        "local_only": True,
        "external_model_generated_required": False,
    }


def train_native_bpe(
    report: NativeCorpusReport,
    output_path: str | Path,
    *,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    max_bytes: int = 500_000_000,
) -> dict[str, Any]:
    ordered = sorted(report.documents, key=lambda row: (row.domain, row.sha256, row.source))
    tokenizer = train_bpe(
        (row.text for row in ordered),
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        max_bytes=int(max_bytes),
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    return {
        "ok": True,
        "tokenizer_version": tokenizer.version,
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "tokenizer_digest": tokenizer.digest,
        "corpus_digest": report.corpus_digest,
        "output": str(output),
        "external_pretrained": False,
    }


def split_native_corpus(
    documents: Sequence[NativeCorpusDocument],
    *,
    validation_fraction: float = 0.05,
    seed: int = 17,
) -> tuple[list[NativeCorpusDocument], list[NativeCorpusDocument]]:
    if not (0.0 < float(validation_fraction) < 0.5):
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if len(documents) < 2:
        raise ValueError("native training requires at least two unique documents")

    grouped: dict[str, list[NativeCorpusDocument]] = {}
    for row in documents:
        grouped.setdefault(row.domain, []).append(row)

    train: list[NativeCorpusDocument] = []
    validation: list[NativeCorpusDocument] = []
    for domain in sorted(grouped):
        rows = sorted(
            grouped[domain],
            key=lambda row: hashlib.sha256(
                f"{int(seed)}:{row.sha256}".encode("ascii")
            ).hexdigest(),
        )
        if len(rows) <= 1:
            train.extend(rows)
            continue
        count = max(1, int(round(len(rows) * float(validation_fraction))))
        count = min(count, len(rows) - 1)
        validation.extend(rows[:count])
        train.extend(rows[count:])

    if not validation:
        ordered = sorted(
            train,
            key=lambda row: hashlib.sha256(
                f"fallback:{int(seed)}:{row.sha256}".encode("ascii")
            ).hexdigest(),
        )
        validation.append(ordered[0])
        train = [row for row in train if row.sha256 != ordered[0].sha256]

    train_hashes = {row.sha256 for row in train}
    validation_hashes = {row.sha256 for row in validation}
    if train_hashes & validation_hashes:
        raise RuntimeError("native train/validation leakage detected")
    if not train or not validation:
        raise RuntimeError("native corpus split produced an empty partition")
    return train, validation


def pack_native_blocks(
    documents: Sequence[NativeCorpusDocument],
    tokenizer: BPETokenizer | Any,
    *,
    context_length: int,
    min_tokens: int = 8,
) -> list[NativeTokenBlock]:
    context_length = int(context_length)
    if context_length < 8:
        raise ValueError("context_length must be at least 8")
    min_tokens = max(2, int(min_tokens))

    blocks: list[NativeTokenBlock] = []
    for document in documents:
        stream = list(tokenizer.encode(document.text))
        if not stream:
            continue
        stream.append(EOS)
        for start in range(0, len(stream), context_length):
            ids = stream[start:start + context_length]
            if len(ids) < min_tokens:
                continue
            if len(ids) < context_length:
                ids.extend([PAD] * (context_length - len(ids)))
            blocks.append(
                NativeTokenBlock(
                    ids=tuple(int(token) for token in ids),
                    domain=document.domain,
                    weight=float(document.weight),
                    source_sha256=document.sha256,
                )
            )
    return blocks


def load_native_tokenizer(checkpoint_dir: str | Path, tokenizer_version: str):
    root = Path(checkpoint_dir).expanduser().resolve()
    if tokenizer_version == "bpe-v1":
        return BPETokenizer.load(root / "tokenizer.json")
    if tokenizer_version == "byte-v1":
        from .tokenizer import ByteTokenizer
        return ByteTokenizer()
    raise ValueError(f"unsupported native tokenizer: {tokenizer_version}")
