from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.foundation import (
    FOUNDATION_DOMAINS,
    FoundationManifest,
    foundation_identity,
    foundation_manifest_digest,
    write_foundation_manifest,
)
from generalist_lm.foundation_benchmarks import (
    foundation_suite,
    foundation_suite_digest,
    foundation_suite_manifest,
)


def _fake_model(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"gpt2","n_positions":4096}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"foundation-weights")
    return root


def _manifest(**overrides):
    values = {
        "model_id": "local/test-foundation",
        "source_revision": "reviewed-revision-001",
        "license": "test-license",
        "architecture": "decoder-only",
        "context_length": 4096,
        "parameter_count": 1_000_000,
        "dtype": "float32",
        "quantization": "none",
    }
    values.update(overrides)
    return FoundationManifest(**values)


def test_foundation_manifest_binds_local_inventory(tmp_path: Path):
    model = _fake_model(tmp_path / "model")
    written = write_foundation_manifest(model, _manifest())
    assert written["ok"] is True
    identity = foundation_identity(model)
    assert identity["manifest"]["backend"] == "transformers"
    assert identity["manifest"]["local_files_only"] is True
    assert identity["manifest"]["trust_remote_code"] is False
    assert identity["inventory"]["weight_files"] == ["model.safetensors"]
    assert identity["inventory"]["config_context_limit"] == 4096
    assert written["inventory"]["files"] == identity["inventory"]["files"]
    assert identity["manifest_digest"] == foundation_manifest_digest(model)


def test_foundation_manifest_rejects_remote_code_and_missing_domains():
    with pytest.raises(ValueError, match="trust_remote_code"):
        _manifest(trust_remote_code=True).validate()
    with pytest.raises(ValueError, match="missing protected domains"):
        _manifest(intended_domains=("language", "coding")).validate()
    with pytest.raises(ValueError, match="context_length must be an integer"):
        _manifest(context_length="4096").validate()
    with pytest.raises(ValueError, match="field license must be a string"):
        _manifest(license=123).validate()


def test_foundation_manifest_cannot_overstate_local_config_context(tmp_path: Path):
    model = _fake_model(tmp_path / "model")
    with pytest.raises(ValueError, match="exceeds the local config.json limit"):
        write_foundation_manifest(model, _manifest(context_length=8192))
    assert not (model / "airi-foundation-manifest.json").exists()


def test_foundation_manifest_rejects_symlinked_model_tree(tmp_path: Path):
    model = _fake_model(tmp_path / "model")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"external")
    link = model / "linked.bin"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        write_foundation_manifest(model, _manifest())


def test_foundation_suite_has_critical_gate_for_every_domain():
    manifest = foundation_suite_manifest()
    assert set(manifest["critical_domains"]) == set(FOUNDATION_DOMAINS)
    assert len(foundation_suite()) >= 16
    assert len(foundation_suite_digest()) == 64
    long_prompts = [task.prompt for task in foundation_suite() if task.domain == "long_context"]
    assert long_prompts
    assert max(len(prompt) for prompt in long_prompts) > 5000


def test_foundation_qualification_is_bound_to_model_manifest_and_suite(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt: str, *, max_new_tokens: int = 192) -> str:
            return ""

    monkeypatch.setattr("generalist_lm.hf_backend.LocalTransformersBackend", Backend)
    monkeypatch.setattr(
        qualification,
        "run_benchmark",
        lambda backend, tasks: {
            "ok": True,
            "score": 100.0,
            "domain_scores": {domain: 1.0 for domain in FOUNDATION_DOMAINS},
            "critical_failures": [],
            "tasks": [],
        },
    )

    result = qualification.qualify_foundation_model(model, minimum_score=90.0)
    assert result["qualified"] is True
    assert result["foundation_qualification_version"] == 1
    assert result["manifest_digest"] == foundation_manifest_digest(model)
    assert result["suite_digest"] == foundation_suite_digest()
    assert qualification.foundation_qualification_status(model)["qualified"] is True

    (model / "model.safetensors").write_bytes(b"mutated-foundation-weights")
    status = qualification.foundation_qualification_status(model)
    assert status["qualified"] is False
    assert status["integrity_ok"] is False


def test_transformers_digest_ignores_attestations_but_not_foundation_manifest(tmp_path: Path):
    from generalist_lm.qualification import transformers_model_digest

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())
    before = transformers_model_digest(model)

    (model / ".airi-qualification.json").write_text("{}", encoding="utf-8")
    (model / ".airi-foundation-qualification.json").write_text("{}", encoding="utf-8")
    assert transformers_model_digest(model) == before

    raw = json.loads((model / "airi-foundation-manifest.json").read_text(encoding="utf-8"))
    raw["source_revision"] = "reviewed-revision-002"
    (model / "airi-foundation-manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    assert transformers_model_digest(model) != before


def test_cli_exposes_foundation_track_commands(tmp_path: Path):
    from generalist_lm.cli import parser

    p = parser()
    args = p.parse_args([
        "foundation-init",
        str(tmp_path / "model"),
        "--model-id", "local/model",
        "--revision", "abc123",
        "--license", "test",
        "--architecture", "decoder-only",
        "--context-length", "4096",
    ])
    assert args.cmd == "foundation-init"

    args = p.parse_args(["qualify-foundation", str(tmp_path / "model")])
    assert args.cmd == "qualify-foundation"

    args = p.parse_args(["foundation-status", str(tmp_path / "model")])
    assert args.cmd == "foundation-status"
