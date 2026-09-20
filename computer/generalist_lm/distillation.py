from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

from .training import SFTExample


class TeacherBackend(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str: ...


@dataclass(frozen=True)
class DistillationPrompt:
    domain: str
    prompt: str
    system: str = ""


_ALLOWED_DOMAINS = {
    "language", "coding", "data", "reasoning", "tools", "structured",
}


def distill_prompts(
    teacher: TeacherBackend,
    prompts: Sequence[DistillationPrompt],
    *,
    max_new_tokens: int = 256,
    max_output_chars: int = 20_000,
) -> tuple[list[SFTExample], dict]:
    """Generate reviewable SFT examples from a local teacher backend.

    Teacher answers are training proposals, not trusted facts. The caller must
    still keep protected validation separate and promotion remains benchmarked.
    """
    examples: list[SFTExample] = []
    rows: list[dict] = []
    seen: set[str] = set()

    for item in prompts:
        if item.domain not in _ALLOWED_DOMAINS:
            rows.append({"domain": item.domain, "status": "rejected_domain"})
            continue
        prompt = str(item.prompt).strip()
        if not prompt:
            rows.append({"domain": item.domain, "status": "empty_prompt"})
            continue
        digest = hashlib.sha256(
            (item.domain + "\0" + item.system + "\0" + prompt).encode("utf-8")
        ).hexdigest()[:24]
        if digest in seen:
            rows.append({"domain": item.domain, "status": "duplicate", "id": digest})
            continue
        seen.add(digest)

        teacher_prompt = prompt
        if item.system:
            teacher_prompt = f"SYSTEM:\n{item.system}\n\nUSER:\n{prompt}\n\nASSISTANT:"
        try:
            answer = str(
                teacher.generate(
                    teacher_prompt,
                    max_new_tokens=max(1, min(int(max_new_tokens), 2048)),
                )
            ).strip()
        except Exception as exc:
            rows.append({
                "domain": item.domain,
                "status": "teacher_error",
                "id": digest,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        if not answer:
            rows.append({"domain": item.domain, "status": "empty_answer", "id": digest})
            continue
        if len(answer) > max_output_chars:
            rows.append({"domain": item.domain, "status": "answer_budget", "id": digest})
            continue

        messages = []
        if item.system:
            messages.append({"role": "system", "content": item.system})
        messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])
        examples.append(SFTExample(messages))
        rows.append({
            "domain": item.domain,
            "status": "accepted",
            "id": digest,
            "answer_chars": len(answer),
        })

    return examples, {
        "accepted": len(examples),
        "requested": len(prompts),
        "rows": rows,
        "policy": "teacher output is distillation data only; it does not bypass held-out qualification",
    }


def save_distilled_jsonl(
    path: str | Path,
    examples: Sequence[SFTExample],
    *,
    provenance: dict | None = None,
) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({
                "messages": example.messages,
                "provenance": dict(provenance or {}),
            }, ensure_ascii=False) + "\n")
    tmp.replace(target)
    raw = target.read_bytes()
    return {
        "path": str(target),
        "rows": len(examples),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
