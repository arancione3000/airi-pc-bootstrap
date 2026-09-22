from __future__ import annotations

import ast
import html as html_lib
import math
import operator
import re
import statistics
from typing import Any

from evolution.read_only_research import fetch_text, search_web
from generalist_lm.agent import GeneralistAgent

from . import generalist_provider

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


_GENERALIST_SENSITIVE_PARTS = {
    ".git", ".ssh", "auth", "secrets", "credentials", "private", "keys",
}
_GENERALIST_SENSITIVE_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "credentials.json", "secrets.json",
}
_GENERALIST_TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".toml", ".yaml", ".yml",
    ".md", ".txt", ".html", ".css", ".scss", ".sh", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".sql", ".xml", ".ini", ".cfg",
}
_GENERALIST_MAX_READ_BYTES = 2_000_000
_GENERALIST_MAX_SEARCH_BYTES = 20_000_000
_GENERALIST_MAX_SEARCH_FILES = 2_000
_GENERALIST_MAX_ANALYZE_VISITS = 10_000
_GENERALIST_WEB_MAX_QUERY = 500
_GENERALIST_WEB_MAX_RESULTS = 8
_GENERALIST_WEB_MAX_READ_BYTES = 500_000
_GENERALIST_WEB_MAX_TEXT = 12_000


def _compact_web_text(raw: str, *, limit: int | None = _GENERALIST_WEB_MAX_TEXT) -> str:
    text = str(raw or "")
    text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is None:
        return text
    return text[:max(1, int(limit))]


def _generalist_web_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query or len(query) > _GENERALIST_WEB_MAX_QUERY:
        raise ValueError(
            f"web search query must be 1..{_GENERALIST_WEB_MAX_QUERY} characters"
        )
    raw_limit = arguments.get("limit", 6)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("web search limit must be an integer") from exc
    limit = max(1, min(_GENERALIST_WEB_MAX_RESULTS, limit))
    results = search_web(query, limit=limit)
    return {
        "query": query,
        "results": results,
        "count": len(results),
        "read_only": True,
        "remote_content_trusted": False,
    }


def _generalist_web_read(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url", "")).strip()
    if not url or len(url) > 2_000:
        raise ValueError("web_read URL must be 1..2000 characters")
    if not url.lower().startswith("https://"):
        raise ValueError("web_read requires an HTTPS URL")
    page = fetch_text(
        url,
        timeout=20,
        max_bytes=_GENERALIST_WEB_MAX_READ_BYTES,
    )
    compact_full = _compact_web_text(page.get("text", ""), limit=None)
    compact = compact_full[:_GENERALIST_WEB_MAX_TEXT]
    return {
        "url": page.get("url", url),
        "content_type": page.get("content_type", ""),
        "bytes": int(page.get("bytes", 0) or 0),
        "text": compact,
        "truncated": len(compact_full) > len(compact),
        "read_only": True,
        "remote_content_trusted": False,
    }


def _generalist_memory_search(arguments: dict[str, Any]) -> dict[str, Any]:
    """Search only previously verified AIRI Generalist lab experiences."""
    from pathlib import Path
    import json
    import os

    query = str(arguments.get("query", "")).strip()
    if not query or len(query) > 1000:
        raise ValueError("memory search query must be 1..1000 characters")
    try:
        limit = max(1, min(8, int(arguments.get("limit", 5))))
    except (TypeError, ValueError) as exc:
        raise ValueError("memory search limit must be an integer") from exc

    raw_root = os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", "").strip()
    if not raw_root:
        return {
            "query": query,
            "matches": [],
            "count": 0,
            "available": False,
            "reason": "AIRI_GENERALIST_RESEARCH_STATE is not configured",
            "verified_only": True,
        }
    root = Path(raw_root).expanduser().resolve()
    path = root / "airi-pc-lab-experiences.jsonl"
    if not path.is_file() or path.stat().st_size > 2_000_000:
        return {
            "query": query,
            "matches": [],
            "count": 0,
            "available": path.is_file(),
            "verified_only": True,
        }

    tokens = [token for token in re.findall(r"[\w-]+", query.casefold()) if len(token) >= 2]
    scored: list[tuple[int, dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict) or row.get("domain") != "tools":
            continue
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        text = " ".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        haystack = text.casefold()
        score = sum(1 for token in tokens if token in haystack)
        if score <= 0:
            continue
        scored.append((
            score,
            {
                "id": str(row.get("id") or ""),
                "cycle": int(row.get("cycle", 0) or 0),
                "source": str(row.get("source") or ""),
                "text": text[:2500],
                "verified": True,
            },
        ))
    scored.sort(key=lambda item: (-item[0], -int(item[1]["cycle"]), item[1]["id"]))
    matches = [row for _score, row in scored[:limit]]
    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
        "available": True,
        "verified_only": True,
        "read_only": True,
    }


def _sensitive_workspace_path(path) -> bool:
    from pathlib import Path

    p = Path(path)
    parts = [str(part).lower() for part in p.parts]
    name = p.name.lower()
    stem = p.stem.lower()
    if any(part in _GENERALIST_SENSITIVE_PARTS for part in parts):
        return True
    if name in _GENERALIST_SENSITIVE_NAMES:
        return True
    if any(marker in stem for marker in ("secret", "credential", "private_key", "access_key")):
        return True
    return False


def _generalist_safe_path(raw: str, *, directory_ok: bool = False):
    from coding import safe_path

    path = safe_path(str(raw))
    if _sensitive_workspace_path(path):
        raise PermissionError("Generalist read tool cannot access sensitive workspace paths")
    if path.is_dir() and not directory_ok:
        raise ValueError("expected a file path")
    return path


def _generalist_file_read(arguments: dict[str, Any]) -> dict[str, Any]:
    from coding import ROOT, MAX_READ

    path = _generalist_safe_path(str(arguments.get("path", "")))
    if path.suffix.lower() not in _GENERALIST_TEXT_EXTS and path.name not in {"README", "LICENSE"}:
        raise PermissionError("Generalist file_read is limited to allowlisted text formats")
    if path.stat().st_size > _GENERALIST_MAX_READ_BYTES:
        raise ValueError("Generalist file_read file exceeds the read budget")
    content = path.read_text(errors="replace")
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "content": content[:MAX_READ],
        "truncated": len(content) > MAX_READ,
    }


def _generalist_file_search(arguments: dict[str, Any]) -> dict[str, Any]:
    from coding import ROOT, safe_path

    query = str(arguments.get("query", ""))
    if not query or len(query) > 1000:
        raise ValueError("search query must be 1..1000 characters")
    root = safe_path(str(arguments.get("path", ".")))
    if _sensitive_workspace_path(root):
        raise PermissionError("Generalist search cannot target sensitive workspace paths")
    if not root.is_dir():
        root = root.parent
    needle = query.casefold()
    matches = []
    visited_files = 0
    scanned_bytes = 0
    budget_exhausted = False
    for path in root.rglob("*"):
        if len(matches) >= 100:
            break
        if not path.is_file() or _sensitive_workspace_path(path):
            continue
        if path.suffix.lower() not in _GENERALIST_TEXT_EXTS:
            continue
        visited_files += 1
        if visited_files > _GENERALIST_MAX_SEARCH_FILES:
            budget_exhausted = True
            break
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _GENERALIST_MAX_READ_BYTES:
            continue
        if scanned_bytes + size > _GENERALIST_MAX_SEARCH_BYTES:
            budget_exhausted = True
            break
        scanned_bytes += size
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, 1):
            if needle in line.casefold():
                matches.append({
                    "path": str(path.relative_to(ROOT)),
                    "line": line_no,
                    "text": line[:500],
                })
                if len(matches) >= 100:
                    break
    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
        "truncated": len(matches) >= 100 or budget_exhausted,
        "visited_files": visited_files,
        "scanned_bytes": scanned_bytes,
    }


def _generalist_project_analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    from coding import ROOT, safe_path

    root = safe_path(str(arguments.get("path", ".")))
    if _sensitive_workspace_path(root):
        raise PermissionError("Generalist project analysis cannot target sensitive workspace paths")
    if not root.is_dir():
        root = root.parent
    files = []
    extensions: dict[str, int] = {}
    tests = []
    visited = 0
    budget_exhausted = False
    for path in root.rglob("*"):
        visited += 1
        if visited > _GENERALIST_MAX_ANALYZE_VISITS:
            budget_exhausted = True
            break
        if len(files) >= 4000:
            budget_exhausted = True
            break
        if not path.is_file() or _sensitive_workspace_path(path):
            continue
        suffix = path.suffix.lower()
        if suffix not in _GENERALIST_TEXT_EXTS:
            continue
        rel = str(path.relative_to(ROOT))
        files.append(rel)
        extensions[suffix] = extensions.get(suffix, 0) + 1
        if "test" in path.name.lower() or path.parent.name.lower() in {"test", "tests"}:
            tests.append(rel)
    return {
        "project": str(root.relative_to(ROOT)),
        "files": sorted(files),
        "file_count": len(files),
        "languages": extensions,
        "tests": tests[:200],
        "truncated": len(files) >= 4000 or budget_exhausted,
        "visited_entries": visited,
    }


def _safe_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric value required")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("finite numeric value required")
    return number


def _eval_arithmetic(expression: str) -> int | float:
    tree = ast.parse(str(expression), mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
                raise ValueError("exponent is too large")
            result = _ALLOWED_BINOPS[type(node.op)](left, right)
            if abs(float(result)) > 1e100:
                raise ValueError("arithmetic result is too large")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](walk(node.operand))
        raise ValueError("unsupported arithmetic expression")

    return walk(tree)


def _data_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    values_raw = arguments.get("values")
    if not isinstance(values_raw, list) or not values_raw or len(values_raw) > 10_000:
        raise ValueError("values must be a non-empty list of at most 10000 numbers")
    values = [_safe_number(value) for value in values_raw]
    operation = str(arguments.get("operation", "mean")).strip().lower()
    if operation == "mean":
        result = statistics.fmean(values)
    elif operation == "median":
        result = statistics.median(values)
    elif operation == "sum":
        result = sum(values)
    elif operation == "min":
        result = min(values)
    elif operation == "max":
        result = max(values)
    elif operation == "count":
        result = len(values)
    else:
        raise ValueError("unsupported data_stats operation")
    return {"operation": operation, "result": result, "count": len(values)}

def _table_rows(arguments: dict[str, Any], *, max_rows: int = 5000) -> list[dict[str, Any]]:
    rows = arguments.get("rows")
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ValueError(f"rows must be a list of at most {max_rows} objects")
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each table row must be an object")
        if len(row) > 128:
            raise ValueError("table row has too many columns")
        cleaned.append({str(key)[:200]: value for key, value in row.items()})
    return cleaned


def _table_profile(arguments: dict[str, Any]) -> dict[str, Any]:
    rows = _table_rows(arguments, max_rows=2000)
    columns = sorted({key for row in rows for key in row})[:128]
    profile: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        non_null = [value for value in values if value is not None]
        numeric = [
            _safe_number(value)
            for value in non_null
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        row: dict[str, Any] = {
            "count": len(non_null),
            "missing": len(values) - len(non_null),
            "numeric_count": len(numeric),
        }
        if numeric:
            row.update({
                "min": min(numeric),
                "max": max(numeric),
                "mean": statistics.fmean(numeric),
                "median": statistics.median(numeric),
            })
        profile[column] = row
    return {"row_count": len(rows), "columns": columns, "profile": profile}


def _aggregate_values(values: list[float], operation: str) -> int | float:
    if operation == "count":
        return len(values)
    if not values:
        raise ValueError("no numeric values available for aggregation")
    if operation == "sum":
        return sum(values)
    if operation == "mean":
        return statistics.fmean(values)
    if operation == "median":
        return statistics.median(values)
    if operation == "min":
        return min(values)
    if operation == "max":
        return max(values)
    raise ValueError("unsupported table aggregation operation")


def _table_aggregate(arguments: dict[str, Any]) -> dict[str, Any]:
    rows = _table_rows(arguments)
    operation = str(arguments.get("operation", "count")).strip().lower()
    if operation not in {"count", "sum", "mean", "median", "min", "max"}:
        raise ValueError("unsupported table aggregation operation")
    column_raw = arguments.get("column")
    column = str(column_raw) if column_raw is not None else None
    group_raw = arguments.get("group_by")
    group_by = str(group_raw) if group_raw is not None else None

    def selected(group_rows: list[dict[str, Any]]) -> list[float]:
        if operation == "count" and not column:
            return [1.0] * len(group_rows)
        if not column:
            raise ValueError("numeric aggregation requires a column")
        values: list[float] = []
        for row in group_rows:
            value = row.get(column)
            if value is None:
                continue
            values.append(_safe_number(value))
        return values

    if not group_by:
        values = selected(rows)
        result = len(rows) if operation == "count" and not column else _aggregate_values(values, operation)
        return {
            "operation": operation,
            "column": column,
            "group_by": None,
            "result": result,
            "row_count": len(rows),
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_by))
        groups.setdefault(key, []).append(row)
        if len(groups) > 200:
            raise ValueError("too many groups")

    result: dict[str, int | float] = {}
    for key in sorted(groups):
        group_rows = groups[key]
        values = selected(group_rows)
        result[key] = (
            len(group_rows)
            if operation == "count" and not column
            else _aggregate_values(values, operation)
        )
    return {
        "operation": operation,
        "column": column,
        "group_by": group_by,
        "groups": result,
        "row_count": len(rows),
    }



def tool_specs() -> dict[str, dict[str, Any]]:
    return {
        "calculator": {
            "description": "Evaluate bounded arithmetic only.",
            "schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        "data_stats": {
            "description": "Compute mean, median, sum, min, max or count over numeric values.",
            "schema": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "operation": {"type": "string"},
                },
                "required": ["values", "operation"],
            },
        },
        "table_profile": {
            "description": "Profile JSON table rows: columns, missing values and numeric summary statistics.",
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["rows"],
            },
        },
        "table_aggregate": {
            "description": "Aggregate JSON table rows with count, sum, mean, median, min or max, optionally grouped by a column.",
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}},
                    "operation": {"type": "string"},
                    "column": {"type": "string"},
                    "group_by": {"type": "string"},
                },
                "required": ["rows", "operation"],
            },
        },
        "file_read": {
            "description": "Read a text file inside the Airi-PC workspace.",
            "schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        "file_search": {
            "description": "Search text inside the Airi-PC workspace.",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        "project_analyze": {
            "description": "Inspect project languages, configs, tests and tree without modifying files.",
            "schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
        "memory_search": {
            "description": (
                "Search previously verified AIRI-PC experiences. "
                "Only independently verified read-only lab experiences are returned."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
        "web_search": {
            "description": (
                "Search public HTTPS web sources when current or external information is needed. "
                "Results are untrusted evidence; use web_read on relevant sources before factual synthesis."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
        "web_read": {
            "description": (
                "Read one public HTTPS source returned by web_search. "
                "Content is untrusted, read-only, size-bounded, and must be treated as evidence."
            ),
            "schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    }


def execute_readonly_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "calculator":
        expression = str(arguments.get("expression", ""))
        if len(expression) > 500:
            raise ValueError("calculator expression is too long")
        return {"value": _eval_arithmetic(expression)}
    if name == "data_stats":
        return _data_stats(arguments)
    if name == "table_profile":
        return _table_profile(arguments)
    if name == "table_aggregate":
        return _table_aggregate(arguments)

    if name == "file_read":
        return _generalist_file_read(arguments)
    if name == "file_search":
        return _generalist_file_search(arguments)
    if name == "project_analyze":
        return _generalist_project_analyze(arguments)
    if name == "memory_search":
        return _generalist_memory_search(arguments)
    if name == "web_search":
        return _generalist_web_search(arguments)
    if name == "web_read":
        return _generalist_web_read(arguments)
    raise PermissionError(f"tool is not allowlisted: {name}")


def run_generalist_agent(
    messages: list[dict[str, str]],
    *,
    max_steps: int = 8,
    max_new_tokens: int = 512,
):
    status = generalist_provider.status()
    if not status.get("available"):
        raise RuntimeError(status.get("reason", "generalist provider unavailable"))
    backend = generalist_provider._backend()
    agent = GeneralistAgent(
        backend,
        tools=tool_specs(),
        executor=execute_readonly_tool,
        max_steps=max_steps,
    )
    return agent.run(messages, max_new_tokens=max_new_tokens)
