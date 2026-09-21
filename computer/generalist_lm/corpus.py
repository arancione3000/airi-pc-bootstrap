from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from .pretraining import CorpusDocument

_ALLOWED_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".sh",
    ".html", ".css", ".js", ".ts", ".java", ".kt", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".sql", ".rst", ".tex",
}
_BLOCKED_PARTS = {
    ".git", ".svn", ".hg", ".venv", "venv", "node_modules", "dist", "build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "generalist-state", "mathesis-state", "secrets", "secret", "credentials",
    "credential", ".ssh", ".ai",
}
_BLOCKED_BASENAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "credentials.json", "secrets.json",
}
_PROTECTED_PATHS = {
    "computer/generalist_lm/benchmarks.py",
    "computer/generalist_lm/foundation.py",
    "computer/generalist_lm/foundation_benchmarks.py",
    "computer/generalist_lm/foundation_probe.py",
    "computer/generalist_lm/harmony_adapter.py",
    "computer/generalist_lm/qualification.py",
    "computer/generalist_lm/production_promotion.py",
    "computer/generalist_lm/research_health.py",
    "computer/generalist_lm/research_cycle.py",
    "computer/generalist_lm/generalist_swarm.py",
    "computer/generalist_lm/generalist_data_growth.py",
    "computer/generalist_lm/evolution.py",
    "computer/generalist_lm/mathesis_bridge.py",
    "computer/control_plane/generalist_provider.py",
    "computer/control_plane/generalist_agent_bridge.py",
    "computer/control_plane/local_agent.py",
    "computer/control_plane/model_router.py",
    "computer/code_agent.py",
    ".github/workflows/generalist-lm.yml",
    ".github/workflows/generalist-continuum.yml",
    ".github/workflows/generalist-watchdog.yml",
}
_SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


@dataclass(frozen=True)
class RepositoryCorpusManifest:
    files: int
    documents: int
    bytes_read: int
    chars_emitted: int
    digest: str
    excluded_protected: int
    excluded_sensitive: int
    excluded_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "documents": self.documents,
            "bytes_read": self.bytes_read,
            "chars_emitted": self.chars_emitted,
            "digest": self.digest,
            "excluded_protected": self.excluded_protected,
            "excluded_sensitive": self.excluded_sensitive,
            "excluded_budget": self.excluded_budget,
        }


_CODE_SUFFIXES = {
    ".py", ".sh", ".js", ".ts", ".java", ".kt", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".rs", ".go",
}


def _document_domain(rel: str) -> str:
    lower = str(rel).lower()
    suffix = Path(lower).suffix
    if suffix in _CODE_SUFFIXES:
        return "code"
    if suffix in {".csv", ".json", ".jsonl", ".sql"} or any(
        token in lower for token in ("dataset", "data/", "tables/", "records/")
    ):
        return "data"
    if suffix == ".tex" or any(
        token in lower for token in ("proof", "theorem", "math/", "reasoning")
    ):
        return "reasoning"
    return "general"


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
    if rel == "tests" or rel.startswith("tests/"):
        return True
    return False


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
    chunk_chars: int = 4000,
    max_documents: int = 1200,
) -> tuple[list[CorpusDocument], RepositoryCorpusManifest]:
    """Build a deterministic bounded corpus from project code/docs.

    Tests, qualification/promotion logic and security/governance files are
    excluded so training cannot memorize its protected evaluation surface.
    """
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("repository corpus root does not exist")

    max_files = max(1, min(int(max_files), 10_000))
    max_bytes = max(1_024, min(int(max_bytes), 100_000_000))
    max_file_bytes = max(1_024, min(int(max_file_bytes), 5_000_000))
    chunk_chars = max(256, min(int(chunk_chars), 32_000))
    max_documents = max(1, min(int(max_documents), 20_000))

    docs: list[CorpusDocument] = []
    bytes_read = 0
    files = 0
    excluded_protected = 0
    excluded_sensitive = 0
    excluded_budget = 0

    stop = False
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name.lower() not in _BLOCKED_PARTS
            and not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            if files >= max_files or bytes_read >= max_bytes or len(docs) >= max_documents:
                excluded_budget += 1
                stop = True
                break
            path = current_path / filename
            try:
                rel = _relative(root, path)
            except Exception:
                # Includes symlinks resolving outside the repository.
                excluded_sensitive += 1
                continue
            if _protected_path(rel):
                excluded_protected += 1
                continue
            if _sensitive_path(rel):
                excluded_sensitive += 1
                continue
            if path.is_symlink():
                excluded_sensitive += 1
                continue
            if path.suffix.lower() not in _ALLOWED_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > max_file_bytes or bytes_read + size > max_bytes:
                excluded_budget += 1
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw:
                continue
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            text = _redact_secrets(decoded).strip()
            if not text:
                continue
            files += 1
            bytes_read += len(raw)

            for offset in range(0, len(text), chunk_chars):
                if len(docs) >= max_documents:
                    excluded_budget += 1
                    stop = True
                    break
                piece = text[offset:offset + chunk_chars].strip()
                if len(piece) < 64:
                    continue
                payload = f"{rel}\0{offset}\0{piece}".encode("utf-8")
                digest = hashlib.sha256(payload).hexdigest()
                docs.append(
                    CorpusDocument(
                        source=f"repo:{rel}#{offset}",
                        text=f"FILE: {rel}\n{piece}",
                        sha256=digest,
                        bytes=len(piece.encode("utf-8")),
                        domain=_document_domain(rel),
                    )
                )
            if stop:
                break
        if stop:
            break

    manifest_digest = hashlib.sha256(
        "\n".join(f"{d.sha256}:{d.source}" for d in docs).encode("utf-8")
    ).hexdigest()
    manifest = RepositoryCorpusManifest(
        files=files,
        documents=len(docs),
        bytes_read=bytes_read,
        chars_emitted=sum(len(d.text) for d in docs),
        digest=manifest_digest,
        excluded_protected=excluded_protected,
        excluded_sensitive=excluded_sensitive,
        excluded_budget=excluded_budget,
    )
    return docs, manifest
