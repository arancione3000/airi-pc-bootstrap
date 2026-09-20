from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))


DOMAINS = (
    "general",
    "language-en",
    "language-it",
    "code",
    "math",
    "reasoning",
    "data-analysis",
    "tool-use",
)


def _write_manifest(root: Path, *, documents_per_domain: int = 1) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in DOMAINS:
        for index in range(documents_per_domain):
            path = corpus / f"{domain}-{index}.txt"
            text = (
                "AIRI learns language, code, mathematics, reasoning and tools from reviewed data. "
                "orange systems reason step by step and verify the final answer. "
                f"domain {domain} sample {index}. "
            ) * 20
            path.write_text(text, encoding="utf-8")
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "domain": domain,
                "language": "it" if domain == "language-it" else "en",
                "license": "project-owned-test-data",
                "source_type": "owned",
                "approved_for_training": True,
                "weight": 1.0,
            })
    manifest = root / "native-corpus.json"
    manifest.write_text(
        json.dumps({"version": "native-corpus-v1", "documents": rows}, indent=2),
        encoding="utf-8",
    )
    return manifest


def _tiny_native_config():
    from generalist_lm.native_foundation import NativeFoundationConfig

    return NativeFoundationConfig(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=96,
        tokenizer_version="byte-v1",
    ).validate()


def test_native_corpus_audit_covers_required_domains_and_is_deterministic(tmp_path: Path):
    from generalist_lm.native_data import (
        audit_native_corpus,
        load_native_corpus,
        split_native_corpus,
    )

    manifest = _write_manifest(tmp_path, documents_per_domain=2)
    report = load_native_corpus(manifest, allowed_roots=[tmp_path])
    audit = audit_native_corpus(report)

    assert audit["ok"] is True
    assert audit["coverage_ok"] is True
    assert audit["missing_domains"] == []
    assert set(audit["domains"]) == set(DOMAINS)
    assert audit["local_only"] is True

    train_a, val_a = split_native_corpus(
        report.documents,
        validation_fraction=0.25,
        seed=77,
    )
    train_b, val_b = split_native_corpus(
        report.documents,
        validation_fraction=0.25,
        seed=77,
    )
    assert [row.sha256 for row in train_a] == [row.sha256 for row in train_b]
    assert [row.sha256 for row in val_a] == [row.sha256 for row in val_b]
    assert {row.sha256 for row in train_a}.isdisjoint(
        {row.sha256 for row in val_a}
    )


def test_native_corpus_rejects_unapproved_and_external_paths(tmp_path: Path):
    from generalist_lm.native_data import load_native_corpus

    local = tmp_path / "text.txt"
    local.write_text("reviewed local text", encoding="utf-8")

    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({
        "version": "native-corpus-v1",
        "documents": [{
            "path": "text.txt",
            "domain": "general",
            "language": "en",
            "license": "owned",
            "source_type": "owned",
            "approved_for_training": False,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not approved_for_training"):
        load_native_corpus(manifest, allowed_roots=[tmp_path])

    manifest.write_text(json.dumps({
        "version": "native-corpus-v1",
        "documents": [{
            "path": "https://example.com/data.txt",
            "domain": "general",
            "language": "en",
            "license": "unknown",
            "source_type": "owned",
            "approved_for_training": True,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="must be local"):
        load_native_corpus(manifest, allowed_roots=[tmp_path])


def test_native_bpe_is_trained_from_airi_corpus_and_saved(tmp_path: Path):
    from generalist_lm.bpe_tokenizer import BPETokenizer
    from generalist_lm.native_data import load_native_corpus, train_native_bpe

    manifest = _write_manifest(tmp_path, documents_per_domain=1)
    report = load_native_corpus(manifest, allowed_roots=[tmp_path])
    tokenizer_path = tmp_path / "tokenizer.json"
    result = train_native_bpe(
        report,
        tokenizer_path,
        vocab_size=320,
        min_frequency=2,
        max_bytes=1_000_000,
    )

    tokenizer = BPETokenizer.load(tokenizer_path)
    assert result["ok"] is True
    assert result["external_pretrained"] is False
    assert result["tokenizer_digest"] == tokenizer.digest
    assert result["corpus_digest"] == report.corpus_digest
    assert 264 < tokenizer.vocab_size <= 320
    text = "ciao AIRI, hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_native_training_reduces_heldout_loss_and_saves_descendant(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_data import load_native_corpus
    from generalist_lm.native_foundation import (
        create_native_root_checkpoint,
        native_checkpoint_status,
    )
    from generalist_lm.native_training import (
        NativeTrainConfig,
        native_training_status,
        train_native_foundation,
    )

    manifest = _write_manifest(tmp_path, documents_per_domain=2)
    report = load_native_corpus(manifest, allowed_roots=[tmp_path])
    assert report.corpus_digest

    root = tmp_path / "root"
    root_status = create_native_root_checkpoint(
        root,
        _tiny_native_config(),
        root_seed=123,
    )
    output = tmp_path / "trained"
    result = train_native_foundation(
        root,
        manifest,
        allowed_roots=[tmp_path],
        output_dir=output,
        config=NativeTrainConfig(
            max_steps=18,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-2,
            min_learning_rate=1e-3,
            warmup_steps=1,
            weight_decay=0.0,
            grad_clip=1.0,
            validation_fraction=0.25,
            max_eval_blocks=8,
            seed=31,
            device="cpu",
            precision="fp32",
        ),
    )

    assert result["ok"] is True
    assert result["final_validation_loss"] < result["initial_validation_loss"]
    assert result["external_pretrained"] is False
    assert result["world_size"] == 1
    assert result["effective_batch_size"] == 2

    checkpoint = native_checkpoint_status(output)
    assert checkpoint["ok"] is True
    assert checkpoint["weights_origin"] == "airi-native-descendant"
    assert checkpoint["external_pretrained"] is False

    status = native_training_status(output)
    assert status["ok"] is True
    assert status["training"]["parent_checkpoint_digest"] == root_status["checkpoint_digest"]
    assert status["training"]["corpus_digest"] == report.corpus_digest
    assert status["training"]["global_step"] == 18


def test_native_training_resumes_optimizer_only_for_same_corpus(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import create_native_root_checkpoint
    from generalist_lm.native_training import NativeTrainConfig, train_native_foundation

    manifest = _write_manifest(tmp_path, documents_per_domain=2)
    root = tmp_path / "root"
    create_native_root_checkpoint(root, _tiny_native_config(), root_seed=7)

    first = tmp_path / "first"
    common = dict(
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=8e-3,
        min_learning_rate=1e-3,
        warmup_steps=1,
        weight_decay=0.0,
        validation_fraction=0.25,
        max_eval_blocks=4,
        seed=9,
        device="cpu",
        precision="fp32",
    )
    train_native_foundation(
        root,
        manifest,
        allowed_roots=[tmp_path],
        output_dir=first,
        config=NativeTrainConfig(max_steps=5, **common),
    )

    second = tmp_path / "second"
    resumed = train_native_foundation(
        first,
        manifest,
        allowed_roots=[tmp_path],
        output_dir=second,
        config=NativeTrainConfig(max_steps=8, **common),
    )
    assert resumed["optimizer_resumed"] is True
    assert resumed["start_step"] == 5
    assert resumed["global_step"] == 8
    assert resumed["steps_run"] == 3


def test_native_training_status_rejects_tampered_trainer_state(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import create_native_root_checkpoint
    from generalist_lm.native_training import (
        NATIVE_TRAINER_STATE_FILENAME,
        NativeTrainConfig,
        native_training_status,
        train_native_foundation,
    )

    manifest = _write_manifest(tmp_path, documents_per_domain=1)
    root = tmp_path / "root"
    create_native_root_checkpoint(root, _tiny_native_config(), root_seed=44)
    output = tmp_path / "trained"
    train_native_foundation(
        root,
        manifest,
        allowed_roots=[tmp_path],
        output_dir=output,
        config=NativeTrainConfig(
            max_steps=3,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=8e-3,
            min_learning_rate=1e-3,
            warmup_steps=1,
            weight_decay=0.0,
            validation_fraction=0.2,
            max_eval_blocks=2,
            seed=2,
            device="cpu",
            precision="fp32",
        ),
    )
    trainer = output / NATIVE_TRAINER_STATE_FILENAME
    trainer.write_bytes(trainer.read_bytes() + b"tamper")
    status = native_training_status(output)
    assert status["ok"] is False
    assert "trainer state digest mismatch" in status["reason"]


def test_native_training_modules_do_not_load_external_pretrained_models():
    import inspect
    import generalist_lm.native_data as native_data
    import generalist_lm.native_training as native_training

    source = (inspect.getsource(native_data) + inspect.getsource(native_training)).lower()
    assert "from_pretrained" not in source
    assert "transformers" not in source
    assert "huggingface" not in source


def test_cli_exposes_native_phase2_commands(tmp_path: Path):
    from generalist_lm.cli import parser

    p = parser()
    args = p.parse_args([
        "native-corpus-audit",
        str(tmp_path / "manifest.json"),
        "--allowed-root",
        str(tmp_path),
    ])
    assert args.cmd == "native-corpus-audit"

    args = p.parse_args([
        "native-tokenizer-train",
        str(tmp_path / "manifest.json"),
        str(tmp_path / "tokenizer.json"),
        "--allowed-root",
        str(tmp_path),
    ])
    assert args.cmd == "native-tokenizer-train"

    args = p.parse_args([
        "native-train",
        str(tmp_path / "root"),
        str(tmp_path / "manifest.json"),
        "--allowed-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "trained"),
    ])
    assert args.cmd == "native-train"

    args = p.parse_args([
        "native-training-status",
        str(tmp_path / "trained"),
    ])
    assert args.cmd == "native-training-status"
