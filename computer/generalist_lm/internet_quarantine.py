from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .data_quality import assess_text, near_duplicate, simhash64


QUARANTINE_VERSION = "airi-internet-quarantine-v1"

# Explicitly conservative.  A source with an unknown/ambiguous license may
# still be used as research reading, but never as training material.
_ALLOWED_TRAINING_LICENSES = {
    "CC0-1.0",
    "CC-BY-2.0-FR",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
}


@dataclass(frozen=True)
class SourceProvenance:
    source_id: str
    url: str
    revision: str
    license: str
    license_url: str
    language: str
    domain: str
    attribution: str = ""
    research_only: bool = False

    def validate(self) -> "SourceProvenance":
        if not self.source_id or not self.url.startswith("https://"):
            raise ValueError("source provenance requires a stable https source")
        if not self.revision:
            raise ValueError("source revision is required")
        if not self.license or not self.license_url.startswith("https://"):
            raise ValueError("license and license URL are required")
        return self


def training_license_allowed(provenance: SourceProvenance) -> bool:
    provenance.validate()
    return (
        not provenance.research_only
        and provenance.license in _ALLOWED_TRAINING_LICENSES
    )


def quarantine_records(
    records: Iterable[str],
    provenance: SourceProvenance,
    *,
    output_dir: str | Path,
    protected_hashes: set[str] | None = None,
    max_records: int = 100_000,
) -> dict[str, Any]:
    provenance.validate()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    protected_hashes = set(protected_hashes or set())

    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    near_seen: list[int] = []

    license_ok = training_license_allowed(provenance)
    for index, raw in enumerate(records):
        if index >= max(1, int(max_records)):
            break
        text = str(raw).strip()
        quality = assess_text(text)
        reasons = list(quality.reasons)
        if quality.sha256 in protected_hashes:
            reasons.append("protected_benchmark_contamination")
        if quality.sha256 in exact_seen:
            reasons.append("exact_duplicate")
        if quality.accepted and near_duplicate(text, near_seen):
            reasons.append("near_duplicate")
        if not license_ok:
            reasons.append("license_not_training_approved")

        accepted = (
            quality.accepted
            and not reasons
            and license_ok
        )
        row = {
            "index": index,
            "sha256": quality.sha256,
            "quality": quality.to_dict(),
            "accepted": accepted,
            "reasons": reasons,
        }
        if accepted:
            admitted.append({
                **row,
                "text": text,
            })
            exact_seen.add(quality.sha256)
            near_seen.append(simhash64(text))
        else:
            rejected.append(row)

    manifest = {
        "schema": 1,
        "version": QUARANTINE_VERSION,
        "source": asdict(provenance),
        "training_license_allowed": license_ok,
        "input_records": len(admitted) + len(rejected),
        "admitted_records": len(admitted),
        "rejected_records": len(rejected),
        "admitted_sha256": [row["sha256"] for row in admitted],
        "rejection_reasons": {},
        "user_private_data_used": False,
        "external_model_quality_judge_used": False,
    }
    counts: dict[str, int] = {}
    for row in rejected:
        for reason in row["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    manifest["rejection_reasons"] = dict(sorted(counts.items()))

    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in admitted
    )
    corpus_path = root / "admitted.jsonl"
    corpus_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    manifest["corpus_sha256"] = hashlib.sha256(
        corpus_path.read_bytes()
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "rejected.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in rejected
        ) + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    return manifest
