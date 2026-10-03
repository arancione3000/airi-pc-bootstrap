"""Reuse reviewed text splits without repeating corpus selection/tokenization.

Plain JSONL only; no pickle. Source bytes, preparation code, exact tokenizer,
quota and persisted source pins bind a cache to the original preparation.
"""
from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import zlib


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(tokenizer, cache: Path, target: int, manifest: dict) -> dict:
    code_root = Path(__file__).parent
    return {
        "version": "reviewed-bootstrap-bundle-v1",
        "target_tokens": int(target),
        "tokenizer": tokenizer.to_dict() if hasattr(tokenizer, "to_dict") else {
            "version": tokenizer.version, "vocab_size": tokenizer.vocab_size,
        },
        "code": {
            name: _digest_file(code_root / name)
            for name in ("bootstrap_data.py", "bootstrap_bundle_cache.py", "phase5_diagnostics.py",
                         "pretraining.py", "training.py", "tokenizer.py", "bpe_tokenizer.py")
        },
        "source_pins": sorted(
            [str(row["id"]), str(row.get("sha256") or "")]
            for row in manifest.get("sources", [])
        ),
        "source_files": {
            path.name: _digest_file(path)
            for path in sorted(cache.iterdir())
            if path.is_file() and (path.name.endswith((".gz", ".bz2", ".download.json")))
        },
    }


def _paths(tokenizer, cache: Path, target: int) -> tuple[Path, Path]:
    key = hashlib.sha256(json.dumps({
        "target": int(target),
        "tokenizer": tokenizer.to_dict() if hasattr(tokenizer, "to_dict") else {
            "version": tokenizer.version, "vocab_size": tokenizer.vocab_size,
        },
    }, sort_keys=True).encode()).hexdigest()[:24]
    root = cache / "prepared-bootstrap-v1"
    return root / f"{key}.jsonl.gz", root / f"{key}.index.json"


def load_reviewed_bundle(tokenizer, cache: Path, target: int, previous_manifest: dict | None):
    from .bootstrap_data import BootstrapDataBundle
    from .pretraining import CorpusDocument
    from .training import SFTExample

    # The normal preparation path establishes pins on the first run.
    if not previous_manifest or not previous_manifest.get("manifest_content_sha256"):
        return None
    data, index = _paths(tokenizer, cache, target)
    if not data.is_file() or not index.is_file():
        return None
    try:
        metadata = json.loads(index.read_text())
        if metadata["identity"] != _identity(tokenizer, cache, target, previous_manifest):
            return None
        if metadata["data_sha256"] != _digest_file(data):
            return None
        documents = {"train": [], "validation": []}
        conversations = {"sft_train": [], "sft_validation": []}
        with gzip.open(data, "rt", encoding="utf-8") as handle:
            manifest = json.loads(next(handle))["manifest"]
            expected_digest = manifest["manifest_content_sha256"]
            unsigned = {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
            actual_digest = hashlib.sha256(json.dumps(
                unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            if expected_digest != actual_digest or expected_digest != previous_manifest["manifest_content_sha256"]:
                return None
            for line in handle:
                row = json.loads(line)
                split = row["split"]
                if split in documents:
                    doc = CorpusDocument(**row["document"])
                    raw = doc.text.encode("utf-8")
                    if hashlib.sha256(raw).hexdigest() != doc.sha256 or len(raw) != doc.bytes:
                        return None
                    documents[split].append(doc)
                elif split in conversations:
                    conversations[split].append(SFTExample(row["messages"]))
                else:
                    return None
        counts = (len(documents["train"]), len(documents["validation"]),
                  len(conversations["sft_train"]), len(conversations["sft_validation"]))
        if list(counts) != metadata["counts"]:
            return None
        manifest_counts = [manifest[k] for k in ("training_documents", "validation_documents",
            "sft_training_conversations", "sft_validation_conversations")]
        if list(counts) != manifest_counts:
            return None
        return BootstrapDataBundle(documents["train"], documents["validation"],
            conversations["sft_train"], conversations["sft_validation"], manifest,
            cache_hit=True)
    except (OSError, ValueError, KeyError, TypeError, EOFError, StopIteration, zlib.error):
        return None


def save_reviewed_bundle(bundle, tokenizer, cache: Path, target: int) -> None:
    data, index = _paths(tokenizer, cache, target)
    data.parent.mkdir(parents=True, exist_ok=True)
    temporary = data.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
        handle.write(json.dumps({"manifest": bundle.manifest}, ensure_ascii=False) + "\n")
        for split, rows in (("train", bundle.train_documents), ("validation", bundle.validation_documents)):
            for row in rows:
                handle.write(json.dumps({"split": split, "document": asdict(row)}, ensure_ascii=False) + "\n")
        for split, rows in (("sft_train", bundle.sft_train), ("sft_validation", bundle.sft_validation)):
            for row in rows:
                handle.write(json.dumps({"split": split, "messages": row.messages}, ensure_ascii=False) + "\n")
    temporary.replace(data)
    metadata = {
        "identity": _identity(tokenizer, cache, target, bundle.manifest),
        "data_sha256": _digest_file(data),
        "counts": [len(bundle.train_documents), len(bundle.validation_documents),
                   len(bundle.sft_train), len(bundle.sft_validation)],
    }
    temporary_index = index.with_suffix(".tmp")
    temporary_index.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    temporary_index.replace(index)
