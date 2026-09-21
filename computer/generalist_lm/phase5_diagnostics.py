from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math
from typing import Any, Iterable

from .tokenizer import EOS, BYTE_OFFSET, VOCAB_SIZE as BYTE_VOCAB_SIZE
from .training import SFTExample, nll_stats_on_examples


PHASE5_PROBES: tuple[dict[str, str], ...] = (
    {"language": "it", "prompt": "Ciao", "target": "Ciao!"},
    {"language": "it", "prompt": "Come ti chiami?", "target": "Mi chiamo AIRI."},
    {"language": "it", "prompt": "Scrivi una frase su un gatto.", "target": "Il gatto dorme sul divano."},
    {"language": "it", "prompt": "Completa: Il cielo è", "target": "azzurro."},
    {"language": "it", "prompt": "Scrivi tre parole italiane.", "target": "casa sole mare"},
    {"language": "en", "prompt": "Hello", "target": "Hello!"},
    {"language": "en", "prompt": "Write one simple sentence.", "target": "The cat is sleeping."},
)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def protected_bootstrap_texts() -> set[str]:
    out: set[str] = set()
    for row in PHASE5_PROBES:
        out.add(_norm(row["prompt"]))
        out.add(_norm(row["target"]))
    return out


def _longest_run(ids: list[int]) -> int:
    if not ids:
        return 0
    longest = current = 1
    for left, right in zip(ids, ids[1:]):
        if left == right:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _distribution(ids: list[int], limit: int = 20) -> list[dict[str, Any]]:
    if not ids:
        return []
    counts = Counter(ids)
    total = len(ids)
    return [
        {"token_id": int(token), "count": int(count), "fraction": float(count / total)}
        for token, count in counts.most_common(max(1, int(limit)))
    ]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _single_greedy_trace(runtime, prompt: str, *, max_new_tokens: int) -> dict[str, Any]:
    torch = runtime.torch
    tokenizer = runtime.tokenizer
    ids = tokenizer.serialize_messages(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    context = list(ids[-runtime.config.context_length:])
    generated: list[int] = []
    step_entropy: list[float] = []
    step_top1: list[float] = []
    step_top5: list[float] = []
    step_eos: list[float] = []

    runtime.model.eval()
    with torch.no_grad():
        for _ in range(max(1, int(max_new_tokens))):
            window = context[-runtime.config.context_length:]
            tensor = torch.tensor([window], dtype=torch.long, device=runtime.device)
            logits = runtime.model(tensor)["logits"][0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
            top = torch.topk(probs, k=min(5, int(probs.numel())))
            next_id = int(torch.argmax(probs).item())

            step_entropy.append(float(entropy.cpu()))
            step_top1.append(float(top.values[0].cpu()))
            step_top5.append(float(top.values.sum().cpu()))
            step_eos.append(float(probs[int(EOS)].cpu()))

            if next_id == int(EOS):
                break
            generated.append(next_id)
            context.append(next_id)

    decoded = tokenizer.decode(generated)
    unique_ratio = (len(set(generated)) / len(generated)) if generated else 0.0
    merged = sum(int(token >= BYTE_VOCAB_SIZE) for token in generated)
    byte_tokens = sum(int(BYTE_OFFSET <= token < BYTE_VOCAB_SIZE) for token in generated)
    return {
        "prompt": prompt,
        "raw_output": decoded,
        "generated_token_ids": generated,
        "generation_length": len(generated),
        "token_entropy": _mean(step_entropy),
        "top1_probability": _mean(step_top1),
        "top5_probability_mass": _mean(step_top5),
        "eos_probability": _mean(step_eos),
        "unique_token_ratio": float(unique_ratio),
        "repetition_rate": float(1.0 - unique_ratio) if generated else 0.0,
        "longest_repeated_token_run": _longest_run(generated),
        "token_frequency_distribution": _distribution(generated),
        "bpe_token_distribution": {
            "merged_tokens": int(merged),
            "byte_tokens": int(byte_tokens),
            "merged_fraction": float(merged / len(generated)) if generated else 0.0,
        },
    }


def evaluate_phase5_language(runtime, *, max_new_tokens: int = 48) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    all_ids: list[int] = []
    similarities: list[float] = []
    exact: list[bool] = []
    nonempty: list[bool] = []

    for probe in PHASE5_PROBES:
        trace = _single_greedy_trace(
            runtime,
            probe["prompt"],
            max_new_tokens=max_new_tokens,
        )
        output = str(trace["raw_output"])
        target = probe["target"]
        similarity = SequenceMatcher(None, output, target, autojunk=False).ratio()
        is_exact = _norm(output) == _norm(target)
        trace.update({
            "language": probe["language"],
            "target": target,
            "generation_similarity": float(similarity),
            "exact": bool(is_exact),
            "nonempty": bool(output.strip()),
        })
        traces.append(trace)
        all_ids.extend(trace["generated_token_ids"])
        similarities.append(float(similarity))
        exact.append(bool(is_exact))
        nonempty.append(bool(output.strip()))

    nll_examples = [
        SFTExample([
            {"role": "user", "content": probe["prompt"]},
            {"role": "assistant", "content": probe["target"]},
        ])
        for probe in PHASE5_PROBES
    ]
    language_nll = nll_stats_on_examples(
        runtime.model,
        runtime.tokenizer,
        nll_examples,
        device=str(runtime.device),
    )

    avg_unique = _mean(row["unique_token_ratio"] for row in traces)
    avg_repetition = _mean(row["repetition_rate"] for row in traces)
    max_run = max((int(row["longest_repeated_token_run"]) for row in traces), default=0)
    top_distribution = _distribution(all_ids)

    pathological = bool(
        (len(all_ids) >= 8 and avg_repetition >= 0.65)
        or max_run >= 8
        or (len(all_ids) >= 12 and avg_unique <= 0.20)
    )
    dominant_fraction = (
        max((row["fraction"] for row in top_distribution), default=0.0)
    )
    if len(all_ids) >= 12 and dominant_fraction >= 0.55:
        pathological = True

    return {
        "schema": 1,
        "suite": "phase5-language-holdout-v1",
        "suite_training_excluded": True,
        "prompt_count": len(PHASE5_PROBES),
        "language_nll": float(language_nll["nll_per_byte"]),
        "language_bits_per_byte": float(language_nll["bits_per_byte"]),
        "generation_similarity": _mean(similarities),
        "exact_accuracy": float(sum(exact) / len(exact)) if exact else 0.0,
        "non_empty_rate": float(sum(nonempty) / len(nonempty)) if nonempty else 0.0,
        "token_entropy": _mean(row["token_entropy"] for row in traces),
        "top1_probability": _mean(row["top1_probability"] for row in traces),
        "top5_probability_mass": _mean(row["top5_probability_mass"] for row in traces),
        "repetition_rate": float(avg_repetition),
        "longest_repeated_token_run": int(max_run),
        "unique_token_ratio": float(avg_unique),
        "eos_probability": _mean(row["eos_probability"] for row in traces),
        "generation_length": _mean(row["generation_length"] for row in traces),
        "token_frequency_distribution": top_distribution,
        "bpe_token_distribution": {
            "vocab_size": int(runtime.tokenizer.vocab_size),
            "tokenizer_version": str(runtime.tokenizer.version),
            "merged_fraction": _mean(
                row["bpe_token_distribution"]["merged_fraction"] for row in traces
            ),
        },
        "dominant_token_fraction": float(dominant_fraction),
        "pathological_repetition": pathological,
        "traces": traces,
    }


def degeneration_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    max_repetition_regression: float = 0.08,
    max_entropy_collapse_fraction: float = 0.55,
) -> tuple[bool, str]:
    if bool(after.get("pathological_repetition")):
        return False, "pathological repetition detected"
    if float(after.get("repetition_rate", 1.0)) > (
        float(before.get("repetition_rate", 0.0)) + float(max_repetition_regression)
    ):
        return False, "repetition rate regressed"
    old_entropy = float(before.get("token_entropy", 0.0))
    new_entropy = float(after.get("token_entropy", 0.0))
    if old_entropy > 0 and new_entropy < old_entropy * float(max_entropy_collapse_fraction):
        return False, "token entropy collapsed"
    if int(after.get("longest_repeated_token_run", 0)) >= 8:
        return False, "repeated-token run exceeded fail-closed limit"
    return True, "degeneration checks passed"
