from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Sequence

from .native_data import (
    NativeTokenBlock,
    load_native_corpus,
    load_native_tokenizer,
    pack_native_blocks,
    split_native_corpus,
)
from .native_foundation import load_native_checkpoint, native_checkpoint_status
from .tokenizer import PAD


NATIVE_EVALUATION_VERSION = "native-evaluation-v1"


@dataclass(frozen=True)
class NativeEvaluationReport:
    checkpoint_digest: str
    corpus_digest: str
    validation_seed: int
    validation_fraction: float
    loss: float
    perplexity: float
    domain_loss: dict[str, float]
    blocks: int
    parameters: int
    external_pretrained: bool
    integrity_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loss_on_blocks(
    torch,
    model,
    blocks: Sequence[NativeTokenBlock],
    *,
    device,
    batch_size: int,
    max_blocks: int,
) -> tuple[float, int]:
    selected = list(blocks[: max(1, int(max_blocks))])
    if not selected:
        raise ValueError("native verifier received no validation blocks")

    total_weighted = 0.0
    total_rows = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), max(1, int(batch_size))):
            rows = selected[start:start + max(1, int(batch_size))]
            ids = torch.tensor(
                [row.ids for row in rows],
                dtype=torch.long,
                device=device,
            )
            labels = ids.clone()
            labels[labels == PAD] = -100
            loss = model(ids, labels=labels)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Native verifier loss")
            total_weighted += float(loss.detach().float().cpu()) * len(rows)
            total_rows += len(rows)

    if total_rows <= 0:
        raise RuntimeError("Native verifier evaluated zero rows")
    return total_weighted / total_rows, total_rows


def evaluate_native_checkpoint(
    checkpoint_dir: str | Path,
    corpus_manifest: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
    validation_fraction: float = 0.2,
    seed: int = 17,
    batch_size: int = 2,
    max_eval_blocks: int = 64,
    device: str = "cpu",
    max_total_bytes: int = 2_000_000_000,
) -> dict[str, Any]:
    """Independently evaluate a Native checkpoint on deterministic held-out data.

    This verifier does not read trainer-reported validation metrics. It reloads
    the model, corpus and tokenizer and computes held-out loss itself.
    """
    import torch

    checkpoint = native_checkpoint_status(checkpoint_dir)
    if not checkpoint.get("ok"):
        raise ValueError(str(checkpoint.get("reason") or "invalid Native checkpoint"))
    if checkpoint.get("external_pretrained") is not False:
        raise ValueError("Native verifier rejects external pretrained weights")

    report = load_native_corpus(
        corpus_manifest,
        allowed_roots=allowed_roots,
        max_total_bytes=int(max_total_bytes),
    )
    _, validation_documents = split_native_corpus(
        report.documents,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )

    model, config, loaded = load_native_checkpoint(checkpoint_dir, device=device)
    tokenizer = load_native_tokenizer(checkpoint_dir, config.tokenizer_version)
    blocks = pack_native_blocks(
        validation_documents,
        tokenizer,
        context_length=config.context_length,
    )
    if not blocks:
        raise ValueError("Native verifier corpus produced no validation blocks")

    resolved_device = next(model.parameters()).device
    overall_loss, evaluated = _loss_on_blocks(
        torch,
        model,
        blocks,
        device=resolved_device,
        batch_size=batch_size,
        max_blocks=max_eval_blocks,
    )

    domain_loss: dict[str, float] = {}
    for domain in sorted({row.domain for row in blocks}):
        domain_blocks = [row for row in blocks if row.domain == domain]
        loss, _ = _loss_on_blocks(
            torch,
            model,
            domain_blocks,
            device=resolved_device,
            batch_size=batch_size,
            max_blocks=max_eval_blocks,
        )
        domain_loss[domain] = float(loss)

    if not math.isfinite(overall_loss) or not all(
        math.isfinite(value) for value in domain_loss.values()
    ):
        raise RuntimeError("Native verifier produced non-finite metrics")

    result = NativeEvaluationReport(
        checkpoint_digest=str(loaded["checkpoint_digest"]),
        corpus_digest=report.corpus_digest,
        validation_seed=int(seed),
        validation_fraction=float(validation_fraction),
        loss=float(overall_loss),
        perplexity=float(math.exp(min(20.0, overall_loss))),
        domain_loss=domain_loss,
        blocks=int(evaluated),
        parameters=int(loaded["parameters"]),
        external_pretrained=False,
        integrity_ok=bool(loaded.get("integrity_ok")),
    )
    return {
        "ok": True,
        "version": NATIVE_EVALUATION_VERSION,
        **result.to_dict(),
    }


def native_continual_decision(
    champion: dict[str, Any],
    control: dict[str, Any],
    *,
    minimum_gain: float = 0.002,
    max_domain_regression: float = 0.0,
) -> tuple[bool, str]:
    for label, report in (("champion", champion), ("control", control)):
        if not report.get("ok") or not report.get("integrity_ok"):
            return False, f"{label} evaluation is invalid"
        if report.get("external_pretrained") is not False:
            return False, f"{label} permits external pretrained weights"
    if str(champion.get("corpus_digest")) != str(control.get("corpus_digest")):
        return False, "control was not evaluated on the champion corpus"
    if float(champion["loss"]) - float(control["loss"]) < max(0.0, float(minimum_gain)):
        return False, "equal-budget continual training did not improve held-out loss"

    champion_domains = champion.get("domain_loss") or {}
    control_domains = control.get("domain_loss") or {}
    tolerance = max(0.0, float(max_domain_regression))
    for domain, old_value in champion_domains.items():
        if domain not in control_domains:
            return False, f"control lost held-out domain: {domain}"
        if float(control_domains[domain]) > float(old_value) + tolerance:
            return False, f"control regressed in held-out domain: {domain}"
    return True, "continual training improved held-out loss without domain regressions"


def native_promotion_decision(
    champion: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_gain: float = 0.005,
    max_domain_regression: float = 0.0,
    max_parameter_ratio: float = 1.5,
) -> tuple[bool, str]:
    """External champion/control/candidate gate.

    A candidate must beat both the frozen champion and an equal-budget control
    continuation, preserve every held-out domain, keep integrity/provenance,
    and remain within the declared parameter budget.
    """
    for label, report in (
        ("champion", champion),
        ("control", control),
        ("candidate", candidate),
    ):
        if not report.get("ok"):
            return False, f"{label} evaluation is not valid"
        if report.get("external_pretrained") is not False:
            return False, f"{label} permits external pretrained weights"
        if not report.get("integrity_ok"):
            return False, f"{label} checkpoint integrity failed"

    digests = {
        str(champion.get("corpus_digest")),
        str(control.get("corpus_digest")),
        str(candidate.get("corpus_digest")),
    }
    if len(digests) != 1:
        return False, "candidate/control/champion were not evaluated on the same corpus"

    old_loss = float(champion["loss"])
    control_loss = float(control["loss"])
    new_loss = float(candidate["loss"])
    margin = max(0.0, float(minimum_gain))
    if old_loss - new_loss < margin:
        return False, "candidate did not beat the frozen champion by the verifier margin"
    if control_loss - new_loss < margin:
        return False, "candidate did not beat the equal-budget control by the verifier margin"

    champion_domains = champion.get("domain_loss") or {}
    control_domains = control.get("domain_loss") or {}
    candidate_domains = candidate.get("domain_loss") or {}
    tolerance = max(0.0, float(max_domain_regression))
    for domain in sorted(set(champion_domains) | set(control_domains)):
        if domain not in candidate_domains:
            return False, f"candidate lost held-out domain: {domain}"
        reference = min(
            float(champion_domains.get(domain, float("inf"))),
            float(control_domains.get(domain, float("inf"))),
        )
        if float(candidate_domains[domain]) > reference + tolerance:
            return False, f"candidate regressed in held-out domain: {domain}"

    champion_params = max(1, int(champion.get("parameters", 1)))
    candidate_params = int(candidate.get("parameters", 0))
    if candidate_params <= 0:
        return False, "candidate parameter count is invalid"
    if candidate_params > int(math.ceil(champion_params * max(1.0, float(max_parameter_ratio)))):
        return False, "candidate exceeds verifier parameter-growth budget"

    return True, "candidate beat champion and equal-budget control without held-out regressions"
