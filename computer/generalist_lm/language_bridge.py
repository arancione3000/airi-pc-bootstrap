from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .curriculum import ResearchRow
from .pretraining import CorpusDocument


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    value = _SPACE_RE.sub(" ", str(text)).strip()
    return value


def _usable_sentence(text: str) -> bool:
    value = _clean(text)
    if len(value) < 32 or len(value) > 280:
        return False
    words = value.split()
    if len(words) < 6 or len(words) > 48:
        return False
    letters = sum(ch.isalpha() for ch in value)
    return letters / max(1, len(value)) >= 0.55


def _sentences(document: CorpusDocument) -> list[str]:
    text = str(document.text)
    if text.startswith("FILE: ") and "\n" in text:
        text = text.split("\n", 1)[1]
    return [
        cleaned
        for part in _SENTENCE_RE.split(text)
        if (cleaned := _clean(part)) and _usable_sentence(cleaned)
    ]


def _rank_key(seed: int, source: str, index: int, kind: str) -> str:
    raw = f"{int(seed)}\0{source}\0{index}\0{kind}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_language_bridge_rows(
    documents: Iterable[CorpusDocument],
    *,
    max_rows: int = 96,
    seed: int = 0,
) -> list[ResearchRow]:
    """Create deterministic corpus-grounded chat supervision.

    Labels are copied only from already-approved training corpus text. No model
    output is used as supervision. The bridge teaches the chat-format model to
    turn natural-language context into multi-token continuations.
    """
    candidates: list[tuple[str, ResearchRow]] = []
    seen: set[tuple[str, str]] = set()

    for document in documents:
        domain = str(getattr(document, "domain", "general") or "general")
        if domain != "general" and not domain.startswith("language"):
            continue
        sentences = _sentences(document)
        for index, sentence in enumerate(sentences):
            words = sentence.split()
            if len(words) >= 9:
                split = max(4, min(len(words) - 3, int(round(len(words) * 0.62))))
                prefix = " ".join(words[:split])
                continuation = " ".join(words[split:])
                key = (prefix, continuation)
                if key not in seen:
                    seen.add(key)
                    row = ResearchRow("language", [
                        {
                            "role": "user",
                            "content": (
                                "Continue this excerpt exactly with the missing words. "
                                f"Excerpt: {prefix}"
                            ),
                        },
                        {"role": "assistant", "content": continuation},
                    ])
                    candidates.append((
                        _rank_key(seed, document.source, index, "completion"),
                        row,
                    ))

            if index + 1 < len(sentences):
                next_sentence = sentences[index + 1]
                key = (sentence, next_sentence)
                if key not in seen:
                    seen.add(key)
                    row = ResearchRow("language", [
                        {
                            "role": "user",
                            "content": (
                                "Write the exact next sentence from this training excerpt. "
                                f"Previous sentence: {sentence}"
                            ),
                        },
                        {"role": "assistant", "content": next_sentence},
                    ])
                    candidates.append((
                        _rank_key(seed, document.source, index, "next-sentence"),
                        row,
                    ))

    candidates.sort(key=lambda item: item[0])
    return [row for _rank, row in candidates[: max(0, int(max_rows))]]
