from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

_ALLOWED_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh",
    ".html", ".css", ".js", ".ts", ".java", ".kt", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".sql",
}
_BLOCKED_PARTS = {
    ".git", ".svn", ".hg", ".venv", "venv", "node_modules", "dist", "build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "generalist-state",
    "mathesis-state", "secrets", "secret", "credentials", "credential", ".ssh",
}
_BLOCKED_BASENAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "credentials.json", "secrets.json",
}
# Protected exam/governance sources must never become pretraining text.
_PROTECTED_PATHS = {
    "computer/generalist_lm/curriculum.py",
    "computer/generalist_lm/curriculum_memory.py",
    "computer/generalist_lm/benchmarks.py",
    "computer/generalist_lm/qualification.py",
    "computer/generalist_lm/production_promotion.py",
    "computer/generalist_lm/research_health.py",
}
_SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    path: str
    text: str


@dataclass(frozen=True)
class CorpusManifest:
    files: int
    chunks: int
    bytes_read: int
    chars_emitted: int
    digest: str
    excluded_protected: int
    excluded_sensitive: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "bytes_read": self.bytes_read,
            "chars_emitted": self.chars_emitted,
            "digest": self.digest,
            "excluded_protected": self.excluded_protected,
            "excluded_sensitive": self.excluded_sensitive,
        }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sensitive_path(rel: str) -> bool:
    parts = tuple(part.lower() for part in Path(rel).parts)
    if not parts:
        return True
    if Path(rel).name.lower() in _BLOCKED_BASENAMES:
        return True
    return any(part in _BLOCKED_PARTS for part in parts)


def _protected_path(rel: str) -> bool:
    if rel in _PROTECTED_PATHS:
        return True
    # Tests are an exam surface. Do not train on any of them.
    return rel == "tests" or rel.startswith("tests/")


def _redact_secrets(text: str) -> str:
    value = str(text)
    for pattern in _SECRET_PATTERNS:
        if "authorization" in pattern.pattern.lower():
            value = pattern.sub(r"\1[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED]", value)
    return value


def repository_corpus(
    root: str | Path,
    *,
    max_files: int = 256,
    max_bytes: int = 2_000_000,
    max_file_bytes: int = 256_000,
    chunk_chars: int = 1200,
    max_chunks: int = 1200,
) -> tuple[list[CorpusChunk], CorpusManifest]:
    """Build a deterministic bounded text/code corpus from a repository.

    This is a read-only data path. It excludes tests and Generalist protected
    evaluation/governance sources so pretraining cannot memorize its exams.
    """
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("repository corpus root does not exist")
    max_files = max(1, min(int(max_files), 10_000))
    max_bytes = max(1_024, min(int(max_bytes), 100_000_000))
    max_file_bytes = max(1_024, min(int(max_file_bytes), 5_000_000))
    chunk_chars = max(128, min(int(chunk_chars), 16_000))
    max_chunks = max(1, min(int(max_chunks), 20_000))

    chunks: list[CorpusChunk] = []
    bytes_read = 0
    files = 0
    excluded_protected = 0
    excluded_sensitive = 0

    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        if files >= max_files or bytes_read >= max_bytes or len(chunks) >= max_chunks:
            break
        try:
            rel = _relative(root, path)
        except Exception:
            continue
        if _protected_path(rel):
            excluded_protected += 1
            continue
        if _sensitive_path(rel):
            excluded_sensitive += 1
            continue
        if path.suffix.lower() not in _ALLOWED_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > max_file_bytes or bytes_read + size > max_bytes:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        clean = _redact_secrets(text).strip()
        if not clean:
            continue
        files += 1
        bytes_read += len(raw)
        for offset in range(0, len(clean), chunk_chars):
            if len(chunks) >= max_chunks:
                break
            piece = clean[offset: offset + chunk_chars]
            if len(piece.strip()) < 32:
                continue
            payload = f"{rel}\0{offset}\0{piece}".encode("utf-8")
            chunks.append(CorpusChunk(
                chunk_id=hashlib.sha256(payload).hexdigest()[:24],
                path=rel,
                text=piece,
            ))

    digest_input = "\n".join(f"{row.chunk_id}:{row.path}" for row in chunks).encode("utf-8")
    manifest = CorpusManifest(
        files=files,
        chunks=len(chunks),
        bytes_read=bytes_read,
        chars_emitted=sum(len(row.text) for row in chunks),
        digest=hashlib.sha256(digest_input).hexdigest(),
        excluded_protected=excluded_protected,
        excluded_sensitive=excluded_sensitive,
    )
    return chunks, manifest
