from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any, Iterable


HARMONY_PACKAGE = "openai-harmony"
HARMONY_MODULE = "openai_harmony"
HARMONY_PINNED_VERSION = "0.0.8"


def harmony_runtime_status() -> dict[str, Any]:
    installed = importlib.util.find_spec(HARMONY_MODULE) is not None
    try:
        version = importlib.metadata.version(HARMONY_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        version = None
    compatible = bool(installed and version == HARMONY_PINNED_VERSION)
    return {
        "available": compatible,
        "installed": installed,
        "version": version,
        "required_version": HARMONY_PINNED_VERSION,
        "reason": (
            "openai-harmony runtime is pinned and available"
            if compatible
            else f"requires {HARMONY_PACKAGE}=={HARMONY_PINNED_VERSION}"
        ),
    }


def _encoding():
    status = harmony_runtime_status()
    if not status["available"]:
        raise RuntimeError(status["reason"])
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def harmony_stop_tokens() -> list[int]:
    encoding = _encoding()
    values = [int(token) for token in encoding.stop_tokens()]
    return sorted(set(values))


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(value, dict):
        direct = value.get("text")
        if isinstance(direct, str):
            return direct
        nested = value.get("content")
        if isinstance(nested, str):
            return nested
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _content_text(to_dict())
        except Exception:
            return ""
    return ""


def _message_row(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None)
    if not isinstance(content, (list, tuple)):
        content = [content] if content is not None else []
    text = "".join(_content_text(item) for item in content)
    channel = getattr(message, "channel", None)
    recipient = getattr(message, "recipient", None)
    return {
        "channel": str(channel) if channel is not None else None,
        "recipient": str(recipient) if recipient is not None else None,
        "text": text,
    }


def parse_harmony_assistant_tokens(tokens: Iterable[int]) -> dict[str, Any]:
    from openai_harmony import Role

    encoding = _encoding()
    stop_tokens = set(harmony_stop_tokens())
    values = [int(token) for token in tokens]
    while values and values[-1] in stop_tokens:
        values.pop()

    if not values:
        raise RuntimeError("Harmony completion contained no parseable tokens")

    try:
        messages = encoding.parse_messages_from_completion_tokens(
            values,
            role=Role.ASSISTANT,
            strict=True,
        )
    except Exception as exc:
        raise RuntimeError(f"invalid Harmony completion: {type(exc).__name__}") from exc

    rows = [_message_row(message) for message in messages]
    final_parts = [
        row["text"]
        for row in rows
        if row["channel"] == "final" and not row["recipient"] and row["text"]
    ]
    tool_rows = [
        row
        for row in rows
        if row["recipient"] or row["channel"] == "commentary"
    ]

    if tool_rows and not final_parts:
        return {
            "ok": False,
            "kind": "tool_action",
            "final_text": "",
            "messages": rows,
            "reason": "Harmony completion requested a tool action; tool-loop adapter is not enabled yet",
        }
    if not final_parts:
        raise RuntimeError("Harmony completion contained no final-channel message")

    return {
        "ok": True,
        "kind": "final",
        "final_text": "".join(final_parts).strip(),
        "messages": rows,
    }
