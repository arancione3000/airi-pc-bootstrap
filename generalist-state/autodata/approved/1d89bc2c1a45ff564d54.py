"""Read YAML frontmatter from corpus markdown files."""

from __future__ import annotations

import re
from pathlib import Path

from .config import logger


class MissingFrontmatterError(Exception):
    """Raised when a markdown file has no opening --- frontmatter block."""


class ParseError(Exception):
    """Raised when frontmatter YAML cannot be parsed."""


_UNQUOTED_VALUE = re.compile(r"^([^:]+):\s*(.+)$")
_QUOTED_VALUE = re.compile(r'^([^:]+):\s*"(.*)"\s*$')


def read_frontmatter(path: Path) -> dict[str, str]:
    """Return frontmatter key/value pairs from a markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        raise MissingFrontmatterError(path)

    end = text.find("\n---", 3)
    if end == -1:
        raise MissingFrontmatterError(path)

    block = text[3:end].strip()
    if not block:
        return {}

    result: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        quoted = _QUOTED_VALUE.match(line)
        if quoted:
            key, value = quoted.group(1).strip(), quoted.group(2)
            result[key] = value.replace('\\"', '"').replace("\\\\", "\\")
            continue
        unquoted = _UNQUOTED_VALUE.match(line)
        if unquoted:
            result[unquoted.group(1).strip()] = unquoted.group(2).strip()
            continue
        raise ParseError(f"Invalid frontmatter line in {path}: {line!r}")

    return result


def build_urn_index(repo_root: Path) -> dict[str, str]:
    """Scan repo for .md files and map URN → path relative to repo root."""
    urn_index: dict[str, str] = {}
    for path in sorted(repo_root.rglob("*.md")):
        try:
            urn = read_frontmatter(path).get("urn")
        except (MissingFrontmatterError, ParseError):
            continue
        if urn:
            rel = path.relative_to(repo_root).as_posix()
            if urn in urn_index and urn_index[urn] != rel:
                logger.warning(
                    "Duplicate URN %s: keeping %s, ignoring %s",
                    urn,
                    urn_index[urn],
                    rel,
                )
            else:
                urn_index[urn] = rel
    logger.info("Indexed %d URN(s) under %s", len(urn_index), repo_root)
    return urn_index


def update_urn_index_from_paths(
    urn_index: dict[str, str],
    repo_root: Path,
    paths: list[Path],
) -> None:
    """Add or refresh index entries for markdown files already under repo_root."""
    for path in sorted(paths):
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        if not path.is_file() or path.suffix != ".md":
            continue
        try:
            urn = read_frontmatter(path).get("urn")
        except (MissingFrontmatterError, ParseError):
            continue
        if urn:
            urn_index[urn] = rel.as_posix()
