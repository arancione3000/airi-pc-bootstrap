from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math
import re
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


def _single_greedy_trace_messages(
    runtime,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    torch = runtime.torch
    tokenizer = runtime.tokenizer
    ids = tokenizer.serialize_messages(
        messages,
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
        "prompt": (
            str(messages[-1].get("content", ""))
            if messages
            else ""
        ),
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


def _single_greedy_trace(runtime, prompt: str, *, max_new_tokens: int) -> dict[str, Any]:
    return _single_greedy_trace_messages(
        runtime,
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
    )


def evaluate_sft_validation(
    runtime,
    examples: Iterable[SFTExample],
    *,
    max_examples: int = 24,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    selected = list(examples)[: max(1, int(max_examples))]
    if not selected:
        raise ValueError("SFT validation requires at least one held-out example")

    traces: list[dict[str, Any]] = []
    all_ids: list[int] = []
    similarities: list[float] = []
    nonempty: list[bool] = []
    word_outputs: list[bool] = []

    for example in selected:
        messages = [dict(row) for row in example.messages]
        target = str(messages[-1].get("content", ""))
        prompt_messages = messages[:-1]
        trace = _single_greedy_trace_messages(
            runtime,
            prompt_messages,
            max_new_tokens=max_new_tokens,
        )
        output = str(trace["raw_output"])
        similarity = SequenceMatcher(None, output, target, autojunk=False).ratio()
        trace.update({
            "target": target,
            "generation_similarity": float(similarity),
            "nonempty": bool(output.strip()),
        })
        traces.append(trace)
        all_ids.extend(trace["generated_token_ids"])
        similarities.append(float(similarity))
        nonempty.append(bool(output.strip()))
        word_outputs.append(bool(re.findall(r"[^\\W\\d_]+", output, flags=re.UNICODE)))

    nll = nll_stats_on_examples(
        runtime.model,
        runtime.tokenizer,
        selected,
        device=str(runtime.device),
    )
    avg_unique = _mean(row["unique_token_ratio"] for row in traces)
    avg_repetition = _mean(row["repetition_rate"] for row in traces)
    max_run = max((int(row["longest_repeated_token_run"]) for row in traces), default=0)
    distribution = _distribution(all_ids)
    dominant_fraction = max(
        (row["fraction"] for row in distribution),
        default=0.0,
    )
    pathological = bool(
        (len(all_ids) >= 8 and avg_repetition >= 0.65)
        or max_run >= 8
        or (len(all_ids) >= 12 and avg_unique <= 0.20)
        or (len(all_ids) >= 12 and dominant_fraction >= 0.55)
    )
    return {
        "schema": 1,
        "suite": "phase5-sft-heldout-v1",
        "suite_training_excluded": True,
        "prompt_count": len(selected),
        "language_nll": float(nll["nll_per_byte"]),
        "language_bits_per_byte": float(nll["bits_per_byte"]),
        "generation_similarity": _mean(similarities),
        "non_empty_rate": float(sum(nonempty) / len(nonempty)) if nonempty else 0.0,
        "word_output_rate": float(sum(word_outputs) / len(word_outputs)) if word_outputs else 0.0,
        "token_entropy": _mean(row["token_entropy"] for row in traces),
        "repetition_rate": float(avg_repetition),
        "longest_repeated_token_run": int(max_run),
        "unique_token_ratio": float(avg_unique),
        "dominant_token_fraction": float(dominant_fraction),
        "pathological_repetition": pathological,
        "traces": traces,
    }


def sft_validation_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    max_repetition_regression: float = 0.025,
    max_nll_regression: float = 0.03,
    min_entropy_fraction: float = 0.85,
    max_unique_ratio_drop: float = 0.04,
) -> tuple[bool, list[str]]:
    """Fail closed on SFT-induced degeneration without using Phase-5 canaries.

    The held-out OASST validation split may guide SFT checkpoint selection, but
    never enters training. Existing pathology is tolerated only if SFT does not
    materially worsen it.
    """
    reasons: list[str] = []
    if float(after.get("repetition_rate", 1.0)) > (
        float(before.get("repetition_rate", 0.0)) + float(max_repetition_regression)
    ):
        reasons.append("held-out repetition regressed")
    if float(after.get("language_nll", float("inf"))) > (
        float(before.get("language_nll", float("inf"))) + float(max_nll_regression)
    ):
        reasons.append("held-out SFT NLL regressed")
    old_entropy = float(before.get("token_entropy", 0.0))
    if old_entropy > 0.0 and float(after.get("token_entropy", 0.0)) < (
        old_entropy * float(min_entropy_fraction)
    ):
        reasons.append("held-out token entropy collapsed")
    if float(after.get("unique_token_ratio", 0.0)) < (
        float(before.get("unique_token_ratio", 0.0)) - float(max_unique_ratio_drop)
    ):
        reasons.append("held-out token diversity regressed")
    old_run = int(before.get("longest_repeated_token_run", 0) or 0)
    new_run = int(after.get("longest_repeated_token_run", 0) or 0)
    if new_run > max(old_run + 2, 8):
        reasons.append("held-out repeated-token run regressed")
    if float(after.get("non_empty_rate", 0.0)) + 0.10 < float(
        before.get("non_empty_rate", 0.0)
    ):
        reasons.append("held-out non-empty rate regressed")
    return (not reasons), reasons


def evaluate_phase5_language(runtime, *, max_new_tokens: int = 48) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    all_ids: list[int] = []
    similarities: list[float] = []
    exact: list[bool] = []
    nonempty: list[bool] = []
    word_outputs: list[bool] = []
    multiword_outputs: list[bool] = []

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
        words = re.findall(r"[^\\W\\d_]+", output, flags=re.UNICODE)
        word_outputs.append(bool(words))
        multiword_outputs.append(len(words) >= 2)

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
        "word_output_rate": float(sum(word_outputs) / len(word_outputs)) if word_outputs else 0.0,
        "multiword_output_rate": float(sum(multiword_outputs) / len(multiword_outputs)) if multiword_outputs else 0.0,
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
