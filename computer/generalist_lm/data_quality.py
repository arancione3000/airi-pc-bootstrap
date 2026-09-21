from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Iterable


_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class QualityAssessment:
    accepted: bool
    score: float
    sha256: str
    characters: int
    words: int
    alphabetic_ratio: float
    printable_ratio: float
    unique_word_ratio: float
    dominant_word_fraction: float
    url_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _normal(text: str) -> str:
    return " ".join(str(text).strip().split())


def exact_content_hash(text: str) -> str:
    return hashlib.sha256(_normal(text).encode("utf-8")).hexdigest()


def assess_text(
    text: str,
    *,
    min_chars: int = 24,
    max_chars: int = 20_000,
) -> QualityAssessment:
    raw = _normal(text)
    reasons: list[str] = []
    chars = len(raw)
    if chars < int(min_chars):
        reasons.append("too_short")
    if chars > int(max_chars):
        reasons.append("too_long")
    if "\x00" in raw:
        reasons.append("nul_byte")

    printable = sum(ch.isprintable() for ch in raw)
    printable_ratio = printable / max(1, chars)
    if printable_ratio < 0.985:
        reasons.append("low_printable_ratio")

    visible = [ch for ch in raw if not ch.isspace()]
    alphabetic = sum(ch.isalpha() for ch in visible)
    alphabetic_ratio = alphabetic / max(1, len(visible))
    if alphabetic_ratio < 0.35:
        reasons.append("low_alphabetic_ratio")

    words = [word.casefold() for word in _WORD_RE.findall(raw)]
    counts = Counter(words)
    unique_word_ratio = len(counts) / max(1, len(words))
    dominant_fraction = max(counts.values(), default=0) / max(1, len(words))
    if len(words) >= 12 and unique_word_ratio < 0.18:
        reasons.append("lexical_repetition")
    if len(words) >= 12 and dominant_fraction > 0.32:
        reasons.append("dominant_word")
    if len(words) < 4 and chars > 80:
        reasons.append("too_few_words")

    url_count = len(_URL_RE.findall(raw))
    if url_count >= 4:
        reasons.append("url_spam")

    # Deliberately simple and model-independent.  Quality scoring must not
    # become hidden distillation from an external LLM.
    score = 1.0
    score -= max(0.0, 0.985 - printable_ratio) * 4.0
    score -= max(0.0, 0.55 - alphabetic_ratio) * 1.5
    score -= max(0.0, 0.35 - unique_word_ratio) * 0.8
    score -= max(0.0, dominant_fraction - 0.18) * 1.2
    score -= min(0.25, url_count * 0.04)
    if reasons:
        score -= min(0.6, 0.08 * len(reasons))
    score = max(0.0, min(1.0, score))

    hard_fail = {
        "too_short",
        "too_long",
        "nul_byte",
        "low_printable_ratio",
        "low_alphabetic_ratio",
        "lexical_repetition",
        "dominant_word",
        "url_spam",
    }
    accepted = not any(reason in hard_fail for reason in reasons) and score >= 0.45
    return QualityAssessment(
        accepted=accepted,
        score=float(score),
        sha256=exact_content_hash(raw),
        characters=chars,
        words=len(words),
        alphabetic_ratio=float(alphabetic_ratio),
        printable_ratio=float(printable_ratio),
        unique_word_ratio=float(unique_word_ratio),
        dominant_word_fraction=float(dominant_fraction),
        url_count=int(url_count),
        reasons=tuple(reasons),
    )


def simhash64(text: str) -> int:
    words = [word.casefold() for word in _WORD_RE.findall(_normal(text))]
    if not words:
        return 0
    vector = [0] * 64
    for word in words:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            out |= 1 << bit
    return out


def hamming_distance(left: int, right: int) -> int:
    return int((int(left) ^ int(right)).bit_count())


def near_duplicate(
    text: str,
    existing_hashes: Iterable[int],
    *,
    max_distance: int = 3,
) -> bool:
    value = simhash64(text)
    return any(
        hamming_distance(value, existing) <= int(max_distance)
        for existing in existing_hashes
    )
