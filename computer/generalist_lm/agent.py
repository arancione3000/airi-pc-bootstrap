from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Protocol

from .tool_protocol import TOOL_RE, parse_tool_call, tool_prompt


class ChatBackend(Protocol):
    def chat(self, messages: list[dict[str, str]], *, max_new_tokens: int = 256) -> str: ...


ToolExecutor = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True)
class AgentRun:
    ok: bool
    answer: str
    steps: int
    tool_calls: list[dict[str, Any]]
    messages: list[dict[str, str]]
    reason: str


class GeneralistAgent:
    """Bounded model/tool loop.

    The language model can only *request* tools. Execution is delegated to an
    injected Control Plane executor after strict allowlist parsing.
    """

    def __init__(
        self,
        backend: ChatBackend,
        *,
        tools: dict[str, dict[str, Any]],
        executor: ToolExecutor,
        max_steps: int = 8,
        max_tool_result_chars: int = 20_000,
    ):
        self.backend = backend
        self.tools = dict(tools)
        self.executor = executor
        self.max_steps = max(1, min(int(max_steps), 16))
        self.max_tool_result_chars = max(256, min(int(max_tool_result_chars), 100_000))

    def run(self, messages: list[dict[str, str]], *, max_new_tokens: int = 512) -> AgentRun:
        transcript = [
            {"role": "system", "content": tool_prompt(self.tools)},
            *[{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
        ]
        calls: list[dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            output = self.backend.chat(transcript, max_new_tokens=max_new_tokens)
            if not TOOL_RE.search(str(output)):
                return AgentRun(
                    ok=True,
                    answer=str(output),
                    steps=step,
                    tool_calls=calls,
                    messages=transcript + [{"role": "assistant", "content": str(output)}],
                    reason="final_answer",
                )

            call = parse_tool_call(str(output), allowed_tools=set(self.tools))
            request_row = {"name": call.name, "arguments": call.arguments}
            calls.append(request_row)
            transcript.append({"role": "assistant", "content": str(output)})

            try:
                result = self.executor(call.name, call.arguments)
                payload = json.dumps(
                    {"ok": True, "tool": call.name, "result": result},
                    ensure_ascii=False,
                    default=str,
                )
            except Exception as exc:
                payload = json.dumps(
                    {
                        "ok": False,
                        "tool": call.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                )
            transcript.append({"role": "tool", "content": payload[: self.max_tool_result_chars]})

        return AgentRun(
            ok=False,
            answer="",
            steps=self.max_steps,
            tool_calls=calls,
            messages=transcript,
            reason="max_steps_exhausted",
        )
