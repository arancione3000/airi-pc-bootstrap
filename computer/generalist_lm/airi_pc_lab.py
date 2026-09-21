from __future__ import annotations

"""Read-only AIRI-PC laboratory for the autonomous Generalist model.

The lab deliberately exposes a *copy of AIRI-PC's public architecture surface*,
not the host computer. The Generalist can inspect a small allowlist of Control
Plane modules through tool calls and learn deterministic supervision generated
from that same snapshot. No shell, network, filesystem mutation, credentials,
phone control, or production promotion capability is exposed.
"""

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .curriculum import ResearchRow
from .tool_protocol import ToolCall, tool_prompt


LAB_VERSION = "airi-pc-lab-v1"

_ALLOWED_MODULES = (
    "computer/control_plane/task_engine.py",
    "computer/control_plane/experience.py",
    "computer/control_plane/judge.py",
    "computer/control_plane/verification_engine.py",
    "computer/control_plane/model_router.py",
    "computer/control_plane/mcp_tasks.py",
    "computer/control_plane/generalist_agent_bridge.py",
    "computer/control_plane/local_agent.py",
)


@dataclass(frozen=True)
class LabSymbol:
    name: str
    kind: str
    doc: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "doc": self.doc}


def _safe_doc(node: ast.AST) -> str:
    value = ast.get_docstring(node) or ""
    return " ".join(value.split())[:360]


def _module_snapshot(root: Path, relative: str) -> dict[str, Any] | None:
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        return None
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    symbols: list[LabSymbol] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(LabSymbol(node.name, "class", _safe_doc(node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                symbols.append(LabSymbol(node.name, "function", _safe_doc(node)))
    return {
        "name": Path(relative).stem,
        "path": relative,
        "doc": _safe_doc(tree),
        "symbols": [row.to_dict() for row in symbols[:24]],
    }


def snapshot_airi_pc_lab(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    modules = [
        row
        for relative in _ALLOWED_MODULES
        if (row := _module_snapshot(root, relative)) is not None
    ]
    return {
        "version": LAB_VERSION,
        "mode": "read_only_sandbox",
        "modules": modules,
        "capabilities": [
            "list_capabilities",
            "inspect_module",
            "describe_task_flow",
        ],
        "denied_capabilities": [
            "shell",
            "network",
            "filesystem_write",
            "credentials",
            "phone_control",
            "production_promotion",
        ],
    }


def lab_tools(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    module_names = [
        str(row.get("name") or "")
        for row in snapshot.get("modules") or []
        if row.get("name")
    ]
    return {
        "lab_list_capabilities": {
            "description": (
                "List the read-only AIRI-PC Lab modules and allowed sandbox "
                "capabilities. It never executes host actions."
            ),
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        "lab_inspect_module": {
            "description": (
                "Inspect classes/functions/docstrings for one allowlisted AIRI-PC "
                "Control Plane module."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "module": {"type": "string", "enum": module_names},
                },
                "required": ["module"],
                "additionalProperties": False,
            },
        },
        "lab_describe_task_flow": {
            "description": (
                "Return the sandboxed AIRI-PC task lifecycle and verification flow. "
                "This is descriptive only and cannot run a real task."
            ),
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def execute_lab_tool(snapshot: dict[str, Any], call: ToolCall) -> dict[str, Any]:
    if call.name == "lab_list_capabilities":
        if call.arguments:
            raise ValueError("lab_list_capabilities takes no arguments")
        return {
            "mode": snapshot.get("mode"),
            "capabilities": snapshot.get("capabilities") or [],
            "modules": [row.get("name") for row in snapshot.get("modules") or []],
            "denied_capabilities": snapshot.get("denied_capabilities") or [],
        }

    if call.name == "lab_inspect_module":
        requested = str(call.arguments.get("module") or "").strip()
        for row in snapshot.get("modules") or []:
            if str(row.get("name") or "") == requested:
                return {
                    "name": row.get("name"),
                    "path": row.get("path"),
                    "doc": row.get("doc"),
                    "symbols": row.get("symbols") or [],
                }
        raise PermissionError(f"AIRI-PC Lab module is not allowlisted: {requested!r}")

    if call.name == "lab_describe_task_flow":
        if call.arguments:
            raise ValueError("lab_describe_task_flow takes no arguments")
        return {
            "flow": [
                "task_engine:start",
                "bounded_operation",
                "verification_engine",
                "judge",
                "experience_record",
            ],
            "mutation": False,
            "production_promotion": False,
        }

    raise PermissionError(f"AIRI-PC Lab tool is not allowlisted: {call.name!r}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_airi_pc_lab_rows(
    snapshot: dict[str, Any],
    *,
    max_rows: int = 18,
) -> list[ResearchRow]:
    """Create verified AIRI-PC tool-use curriculum from the read-only snapshot."""
    tools = lab_tools(snapshot)
    prompt = tool_prompt(tools)
    rows: list[ResearchRow] = []

    listed = execute_lab_tool(
        snapshot,
        ToolCall(name="lab_list_capabilities", arguments={}),
    )
    rows.append(
        ResearchRow(
            "tools",
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "In AIRI-PC Lab, inspect which safe capabilities are available.",
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"lab_list_capabilities","arguments":{}}</tool_call>',
                },
                {"role": "tool", "content": _json(listed)},
                {
                    "role": "assistant",
                    "content": (
                        "The lab is read-only. I can inspect allowlisted AIRI-PC "
                        "modules and task flow, but I cannot execute host actions."
                    ),
                },
            ],
        )
    )

    flow = execute_lab_tool(
        snapshot,
        ToolCall(name="lab_describe_task_flow", arguments={}),
    )
    rows.append(
        ResearchRow(
            "tools",
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Show the AIRI-PC Lab task lifecycle before learning from a result.",
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"lab_describe_task_flow","arguments":{}}</tool_call>',
                },
                {"role": "tool", "content": _json(flow)},
                {
                    "role": "assistant",
                    "content": (
                        "A task is bounded, verified, judged, and only then recorded "
                        "as experience. The lab does not mutate production."
                    ),
                },
            ],
        )
    )

    for module in snapshot.get("modules") or []:
        if len(rows) >= max(1, int(max_rows)):
            break
        name = str(module.get("name") or "")
        if not name:
            continue
        inspected = execute_lab_tool(
            snapshot,
            ToolCall(name="lab_inspect_module", arguments={"module": name}),
        )
        symbols = [
            str(row.get("name") or "")
            for row in inspected.get("symbols") or []
            if row.get("name")
        ][:6]
        rows.append(
            ResearchRow(
                "coding",
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Inspect AIRI-PC Lab module {name} before using it.",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"lab_inspect_module","arguments":'
                            + _json({"module": name})
                            + "}</tool_call>"
                        ),
                    },
                    {"role": "tool", "content": _json(inspected)},
                    {
                        "role": "assistant",
                        "content": (
                            f"{name} is available only through the read-only lab. "
                            + (
                                "Key symbols: " + ", ".join(symbols) + "."
                                if symbols
                                else "No public symbols were exposed."
                            )
                        ),
                    },
                ],
            )
        )
    return rows[: max(1, int(max_rows))]


def run_airi_pc_lab_probe(runtime, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Let a checkpoint attempt one genuine allowlisted AIRI-PC Lab interaction."""
    tools = lab_tools(snapshot)
    messages = [
        {
            "role": "user",
            "content": (
                "Use exactly one AIRI-PC Lab tool to inspect the safe task lifecycle. "
                "Do not claim that a real computer action happened."
            ),
        }
    ]
    try:
        call = runtime.request_tool(messages, tools, max_new_tokens=96)
        result = execute_lab_tool(snapshot, call)
    except Exception as exc:
        return {
            "ok": False,
            "version": LAB_VERSION,
            "mode": snapshot.get("mode"),
            "error": f"{type(exc).__name__}:{exc}",
            "tool_call_valid": False,
        }

    try:
        summary = runtime.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are inside a read-only AIRI-PC research sandbox. "
                        "Summarize only the supplied tool result."
                    ),
                },
                *messages,
                {
                    "role": "assistant",
                    "content": (
                        "<tool_call>"
                        + _json({"name": call.name, "arguments": call.arguments})
                        + "</tool_call>"
                    ),
                },
                {"role": "tool", "content": _json(result)},
                {
                    "role": "user",
                    "content": "Summarize what the tool proved in one short sentence.",
                },
            ],
            max_new_tokens=80,
        )
    except Exception as exc:
        summary = f"summary_failed:{type(exc).__name__}:{exc}"

    return {
        "ok": True,
        "version": LAB_VERSION,
        "mode": snapshot.get("mode"),
        "tool_call_valid": True,
        "tool": call.name,
        "arguments": call.arguments,
        "tool_result": result,
        "model_summary": summary[:2000],
    }


def summarize_lab_learning(rows: list[ResearchRow]) -> dict[str, Any]:
    counts = Counter(row.domain for row in rows)
    return {
        "version": LAB_VERSION,
        "training_rows": len(rows),
        "domains": dict(sorted(counts.items())),
    }
