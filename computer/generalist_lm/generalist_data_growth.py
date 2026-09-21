from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable
from urllib.parse import quote, quote_plus, urlparse
from urllib.request import Request, urlopen


GENERALIST_DATA_GROWTH_VERSION = "generalist-data-growth-v3"

PERMISSIVE_SPDX = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "Unlicense",
    "0BSD",
    "ISC",
}

_ALLOWED_SUFFIXES = {
    ".txt", ".md", ".rst", ".json", ".jsonl", ".csv",
    ".py", ".js", ".ts", ".tsx", ".java", ".c", ".h",
    ".cpp", ".hpp", ".rs", ".go", ".sql", ".tex", ".yaml", ".yml", ".toml",
}

_CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".java", ".c", ".h",
    ".cpp", ".hpp", ".rs", ".go", ".sql",
}

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"
)


@dataclass(frozen=True)
class GrowthFile:
    repo: str
    commit: str
    path: str
    spdx: str
    sha256: str
    bytes: int
    domain: str
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _request_json(
    url: str,
    *,
    token: str | None = None,
    opener: Callable[..., Any] | None = None,
    timeout: float = 20.0,
    max_bytes: int = 4_000_000,
) -> Any:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com":
        raise PermissionError("Generalist data discovery is restricted to GitHub HTTPS API")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AIRI-Generalist-Data-Growth/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    handle = (opener or urlopen)(req, timeout=timeout)
    try:
        final = urlparse(handle.geturl())
        if final.scheme != "https" or final.hostname != "api.github.com":
            raise PermissionError("GitHub API redirected to unapproved host")
        limit = max(64_000, min(int(max_bytes), 16_000_000))
        raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(
                f"GitHub metadata response exceeded budget:{limit}"
            )
        return json.loads(raw.decode("utf-8"))
    finally:
        handle.close()


def _request_raw(
    url: str,
    *,
    max_bytes: int,
    opener: Callable[..., Any] | None = None,
    timeout: float = 20.0,
) -> bytes:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"raw.githubusercontent.com", "github.com"}
    ):
        raise PermissionError("Generalist corpus download uses approved GitHub HTTPS hosts only")
    req = Request(
        url,
        headers={"User-Agent": "AIRI-Generalist-Data-Growth/1.0"},
    )
    handle = (opener or urlopen)(req, timeout=timeout)
    try:
        final = urlparse(handle.geturl())
        if (
            final.scheme != "https"
            or final.hostname not in {"raw.githubusercontent.com", "github.com"}
        ):
            raise PermissionError("corpus download redirected to unapproved host")
        raw = handle.read(max(1, int(max_bytes)) + 1)
        if len(raw) > max_bytes:
            raise ValueError("corpus object exceeds file budget")
        return raw
    finally:
        handle.close()


def _quality_text(raw: bytes) -> tuple[bool, str, str]:
    if not raw or b"\x00" in raw:
        return False, "binary_or_empty", ""
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return False, "empty_text", ""
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    if replacement_ratio > 0.01:
        return False, "encoding_damage", ""
    printable = sum(
        1 for char in text
        if char.isprintable() or char in "\n\r\t"
    )
    if printable / max(1, len(text)) < 0.92:
        return False, "low_printable_ratio", ""
    alpha = sum(char.isalpha() for char in text)
    digit = sum(char.isdigit() for char in text)
    if (alpha + digit) / max(1, len(text)) < 0.12:
        return False, "low_information_density", ""
    if _SECRET_RE.search(text):
        return False, "secret_like_content", ""
    return True, "accepted", text


def _domain_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    lower = path.lower()
    if suffix in _CODE_SUFFIXES:
        return "code"
    if suffix == ".tex":
        return "reasoning"
    if (
        suffix in {".csv", ".json", ".jsonl", ".sql"}
        or any(token in lower for token in ("dataset", "data/", "tables/", "records/"))
    ):
        return "data"
    if any(
        token in lower
        for token in ("math", "theorem", "algebra", "proof", "reasoning", "logic")
    ):
        return "reasoning"
    if any(token in lower for token in ("italian", "italiano", "/it/", "_it.")):
        return "language-it"
    return "general"


def _low_value_path(path: str) -> bool:
    lower = "/" + path.replace("\\", "/").lower().lstrip("/")
    name = Path(path).name.lower()
    if any(
        token in lower
        for token in (
            "/node_modules/", "/vendor/", "/dist/", "/build/",
            "/coverage/", "/quotes/", "/snapshots/", "/fixtures/",
        )
    ):
        return True
    return name in {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "cargo.lock",
    }


def _desired_domains(signals: Iterable[str] | None) -> list[str]:
    signal_set = {str(row).lower() for row in (signals or ())}
    domains: list[str] = []
    if "reasoning_gap" in signal_set or "symbolic_reasoning_signal" in signal_set:
        domains.append("reasoning")
    if "data_gap" in signal_set:
        domains.append("data")
    if "coding_gap" in signal_set:
        domains.append("code")
    if "language_gap" in signal_set:
        domains.append("language-it")
    domains.extend(["general", "code", "data", "reasoning"])
    return list(dict.fromkeys(domains))


def _rank_entries(
    entries: Iterable[dict[str, Any]],
    desired_domains: Iterable[str],
) -> list[dict[str, Any]]:
    preferred_suffixes = {
        ".txt", ".md", ".rst", ".py", ".js", ".ts", ".java",
        ".c", ".cpp", ".rs", ".go", ".sql", ".tex", ".csv", ".jsonl",
    }
    buckets: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        domain = _domain_for(str(entry.get("path") or ""))
        buckets.setdefault(domain, []).append(entry)
    for rows in buckets.values():
        rows.sort(
            key=lambda row: (
                0
                if Path(str(row.get("path") or "")).suffix.lower()
                in preferred_suffixes
                else 1,
                -min(int(row.get("size", 0) or 0), 512_000),
                str(row.get("path") or ""),
            )
        )

    order = list(dict.fromkeys([*desired_domains, *sorted(buckets)]))
    ranked: list[dict[str, Any]] = []
    while any(buckets.get(domain) for domain in order):
        for domain in order:
            rows = buckets.get(domain) or []
            if rows:
                ranked.append(rows.pop(0))
    return ranked


def _candidate_queries(signals: Iterable[str] | None) -> list[str]:
    """Return short, license-qualified discovery queries.

    GitHub repository search behaves much better when topical terms are compact.
    Putting the SPDX qualifier into discovery also prevents most of the API
    budget being wasted on repositories that will later fail closed.
    """
    signal_set = {str(row).lower() for row in (signals or ())}
    queries: list[str] = []
    if "reasoning_gap" in signal_set or "symbolic_reasoning_signal" in signal_set:
        queries.extend([
            "reasoning dataset license:mit",
            "mathematics proofs license:mit",
        ])
    if "data_gap" in signal_set:
        queries.extend([
            "csv dataset license:mit",
            "json dataset license:mit",
        ])
    if "coding_gap" in signal_set:
        queries.append("algorithms license:mit")
    if "language_gap" in signal_set:
        queries.append("italian corpus license:mit")
    queries.extend([
        "text corpus license:mit",
        "educational corpus license:apache-2.0",
        "algorithms license:mit",
        "dataset license:cc0-1.0",
    ])
    return list(dict.fromkeys(queries))[:8]


def _search_repositories(
    queries: Iterable[str],
    *,
    token: str | None,
    per_query: int,
    opener=None,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for query in queries:
        qualifier = "" if "license:" in query.lower() else " license:mit"
        bounded_query = (
            f"{query}{qualifier} size:<100000 fork:false archived:false"
        )
        url = (
            "https://api.github.com/search/repositories?"
            f"q={quote_plus(bounded_query)}&sort=stars&order=desc&per_page={max(1, min(per_query, 10))}"
        )
        payload = _request_json(url, token=token, opener=opener)
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name") or "")
            if full_name:
                rows.setdefault(full_name, item)
    return list(rows.values())


def grow_generalist_data(
    state_dir: str | Path,
    *,
    signals: Iterable[str] | None = None,
    github_token: str | None = None,
    max_new_bytes: int = 8_000_000,
    max_total_bytes: int = 50_000_000,
    max_files_per_repo: int = 24,
    max_repositories: int = 12,
    max_file_bytes: int = 512_000,
    max_total_files_per_repo: int = 48,
    max_total_bytes_per_repo: int = 8_000_000,
    opener=None,
) -> dict[str, Any]:
    """Grow a persistent permissively licensed corpus without model weights.

    Discovery is automatic, but admission is fail-closed:
    GitHub repository -> SPDX allowlist -> immutable commit -> text quarantine
    -> quality/secret filter -> SHA-256 dedup -> approved corpus.
    """
    root = Path(state_dir).expanduser().resolve()
    quarantine = root / "quarantine"
    approved = root / "approved"
    quarantine.mkdir(parents=True, exist_ok=True)
    approved.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {
            "version": GENERALIST_DATA_GROWTH_VERSION,
            "files": [],
        }
    existing = {
        str(row.get("sha256")): row
        for row in manifest.get("files", [])
        if isinstance(row, dict) and row.get("sha256")
    }
    existing_sources = {
        (str(row.get("repo")), str(row.get("commit")), str(row.get("path")))
        for row in existing.values()
    }
    repo_file_counts: dict[str, int] = {}
    repo_byte_counts: dict[str, int] = {}
    for row in existing.values():
        repo_name = str(row.get("repo") or "")
        repo_file_counts[repo_name] = repo_file_counts.get(repo_name, 0) + 1
        repo_byte_counts[repo_name] = (
            repo_byte_counts.get(repo_name, 0)
            + int(row.get("bytes", 0) or 0)
        )
    current_bytes = sum(
        int(row.get("bytes", 0) or 0)
        for row in existing.values()
    )
    total_cap = max(0, int(max_total_bytes))
    cycle_cap = min(
        max(0, int(max_new_bytes)),
        max(0, total_cap - current_bytes),
    )
    if cycle_cap <= 0:
        return {
            "ok": True,
            "version": GENERALIST_DATA_GROWTH_VERSION,
            "added_files": 0,
            "added_bytes": 0,
            "total_files": len(existing),
            "total_bytes": current_bytes,
            "reason": "persistent corpus budget reached",
        }

    token = github_token or os.environ.get("GITHUB_TOKEN")
    repositories = _search_repositories(
        _candidate_queries(signals),
        token=token,
        per_query=8,
        opener=opener,
    )

    added: list[GrowthFile] = []
    rejected: list[dict[str, Any]] = []
    bytes_added = 0
    repos_used = 0

    for item in repositories:
        if repos_used >= max(1, int(max_repositories)):
            break
        full_name = str(item.get("full_name") or "")
        if not full_name or "/" not in full_name:
            continue
        if (
            repo_file_counts.get(full_name, 0) >= max(1, int(max_total_files_per_repo))
            or repo_byte_counts.get(full_name, 0) >= max(1, int(max_total_bytes_per_repo))
        ):
            rejected.append({
                "repo": full_name,
                "reason": "persistent_repo_cap",
            })
            continue

        try:
            details = _request_json(
                f"https://api.github.com/repos/{full_name}",
                token=token,
                opener=opener,
            )
        except Exception as exc:
            rejected.append({
                "repo": full_name,
                "reason": f"metadata_repo:{type(exc).__name__}",
            })
            continue
        license_obj = details.get("license") if isinstance(details, dict) else None
        spdx = (
            str(license_obj.get("spdx_id") or "")
            if isinstance(license_obj, dict)
            else ""
        )
        if spdx not in PERMISSIVE_SPDX:
            rejected.append({"repo": full_name, "reason": f"license:{spdx or 'unknown'}"})
            continue

        branch = str(details.get("default_branch") or "main")
        try:
            branch_info = _request_json(
                f"https://api.github.com/repos/{full_name}/branches/{quote_plus(branch)}",
                token=token,
                opener=opener,
            )
        except Exception as exc:
            rejected.append({
                "repo": full_name,
                "reason": f"metadata_branch:{type(exc).__name__}",
            })
            continue
        commit = str(
            ((branch_info.get("commit") or {}).get("sha"))
            if isinstance(branch_info, dict)
            else ""
        )
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            rejected.append({"repo": full_name, "reason": "unresolved_immutable_commit"})
            continue

        try:
            tree = _request_json(
                f"https://api.github.com/repos/{full_name}/git/trees/{commit}?recursive=1",
                token=token,
                opener=opener,
                max_bytes=16_000_000,
            )
        except Exception as exc:
            rejected.append({
                "repo": full_name,
                "reason": f"metadata_tree:{type(exc).__name__}",
            })
            continue
        if bool(tree.get("truncated")):
            rejected.append({
                "repo": full_name,
                "reason": "metadata_tree:truncated",
            })
            continue
        entries = [
            row for row in tree.get("tree", [])
            if isinstance(row, dict)
            and row.get("type") == "blob"
            and Path(str(row.get("path") or "")).suffix.lower() in _ALLOWED_SUFFIXES
            and 512 <= int(row.get("size", 0) or 0) <= max_file_bytes
            and not _low_value_path(str(row.get("path") or ""))
        ]
        entries = _rank_entries(entries, _desired_domains(signals))
        accepted_repo = 0
        accepted_repo_bytes = 0

        for entry in entries:
            if accepted_repo >= max(1, int(max_files_per_repo)):
                break
            if (
                repo_file_counts.get(full_name, 0) + accepted_repo
                >= max(1, int(max_total_files_per_repo))
            ):
                break
            if bytes_added >= cycle_cap:
                break
            path = str(entry.get("path") or "")
            if (full_name, commit.lower(), path) in existing_sources:
                continue
            remaining = cycle_cap - bytes_added
            repo_remaining = (
                max(1, int(max_total_bytes_per_repo))
                - repo_byte_counts.get(full_name, 0)
                - accepted_repo_bytes
            )
            budget = min(max_file_bytes, remaining, repo_remaining)
            if budget < 512:
                break
            raw_path = quote(path, safe="/")
            raw_url = f"https://raw.githubusercontent.com/{full_name}/{commit}/{raw_path}"
            try:
                raw = _request_raw(
                    raw_url,
                    max_bytes=budget,
                    opener=opener,
                )
            except Exception as exc:
                rejected.append({
                    "repo": full_name,
                    "path": path,
                    "reason": f"download:{type(exc).__name__}",
                })
                continue

            digest = hashlib.sha256(raw).hexdigest()
            if digest in existing or any(row.sha256 == digest for row in added):
                continue

            safe_name = hashlib.sha256(
                f"{full_name}\0{commit}\0{path}".encode("utf-8")
            ).hexdigest()[:20] + Path(path).suffix.lower()
            qpath = quarantine / safe_name
            qpath.write_bytes(raw)
            good, reason, _text = _quality_text(raw)
            if not good:
                qpath.unlink(missing_ok=True)
                rejected.append({"repo": full_name, "path": path, "reason": reason})
                continue

            target = approved / safe_name
            shutil.move(str(qpath), str(target))
            record = GrowthFile(
                repo=full_name,
                commit=commit.lower(),
                path=path,
                spdx=spdx,
                sha256=digest,
                bytes=len(raw),
                domain=_domain_for(path),
                source_url=raw_url,
            )
            added.append(record)
            bytes_added += len(raw)
            accepted_repo += 1
            accepted_repo_bytes += len(raw)

        if accepted_repo:
            repos_used += 1
        if bytes_added >= cycle_cap:
            break

    all_rows = list(existing.values()) + [row.to_dict() for row in added]
    all_rows.sort(key=lambda row: (str(row.get("repo")), str(row.get("path")), str(row.get("sha256"))))
    manifest = {
        "version": GENERALIST_DATA_GROWTH_VERSION,
        "files": all_rows,
        "policy": {
            "permissive_spdx": sorted(PERMISSIVE_SPDX),
            "immutable_commit_required": True,
            "sha256_dedup": True,
            "quality_filter": True,
            "secret_filter": True,
            "external_pretrained": False,
        },
    }
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(manifest_path)

    domain_files: dict[str, int] = {}
    domain_bytes: dict[str, int] = {}
    for row in all_rows:
        domain = str(row.get("domain") or "general")
        domain_files[domain] = domain_files.get(domain, 0) + 1
        domain_bytes[domain] = (
            domain_bytes.get(domain, 0) + int(row.get("bytes", 0) or 0)
        )

    return {
        "ok": True,
        "version": GENERALIST_DATA_GROWTH_VERSION,
        "added_files": len(added),
        "added_bytes": bytes_added,
        "total_files": len(all_rows),
        "total_bytes": sum(int(row.get("bytes", 0) or 0) for row in all_rows),
        "repositories_used": repos_used,
        "domain_files": domain_files,
        "domain_bytes": domain_bytes,
        "desired_domains": _desired_domains(signals),
        "rejected": rejected[:100],
        "manifest": str(manifest_path),
        "approved_dir": str(approved),
        "external_pretrained": False,
    }


def main() -> int:
    state = os.environ.get(
        "AIRI_GENERALIST_DATA_STATE",
        ".ai/generalist-data",
    )
    signals = [
        row.strip()
        for row in os.environ.get("AIRI_GENERALIST_DATA_SIGNALS", "").split(",")
        if row.strip()
    ]
    result = grow_generalist_data(
        state,
        signals=signals,
        github_token=os.environ.get("GITHUB_TOKEN"),
        max_new_bytes=int(os.environ.get("AIRI_GENERALIST_DATA_MAX_NEW_BYTES", "8000000")),
        max_total_bytes=int(os.environ.get("AIRI_GENERALIST_DATA_MAX_TOTAL_BYTES", "50000000")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
