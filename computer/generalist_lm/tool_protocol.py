from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_call(text: str, *, allowed_tools: set[str], max_chars: int = 12000) -> ToolCall:
    raw = str(text)
    if len(raw) > max_chars:
        raise ValueError("tool call output is too large")
    match = TOOL_RE.search(raw)
    payload = match.group(1) if match else raw.strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("model output is not a valid tool-call JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("tool call must be a JSON object")
    name = str(value.get("name", "")).strip()
    args = value.get("arguments", {})
    if name not in allowed_tools:
        raise PermissionError(f"tool is not allowlisted: {name!r}")
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return ToolCall(name=name, arguments=args)


def tool_prompt(tools: dict[str, dict[str, Any]]) -> str:
    safe = {
        str(name): {
            "description": str(spec.get("description", ""))[:1000],
            "schema": spec.get("schema", {}),
        }
        for name, spec in tools.items()
    }
    return (
        "Available tools are strictly allowlisted. If a tool is required, output exactly "
        "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>. "
        "Never invent tool names. Tool specifications: "
        + json.dumps(safe, ensure_ascii=False, sort_keys=True)
    )
