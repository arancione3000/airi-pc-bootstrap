from __future__ import annotations

import json
import os
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

    _stub_transformers_preflight(monkeypatch)

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
    assert result["foundation_qualification_version"] == 3
    assert result["manifest_digest"] == foundation_manifest_digest(model)
    assert result["suite_digest"] == foundation_suite_digest()
    assert qualification.foundation_qualification_status(model)["qualified"] is True

    (model / "model.safetensors").write_bytes(b"mutated-foundation-weights")
    status = qualification.foundation_qualification_status(model)
    assert status["qualified"] is False
    assert status["integrity_ok"] is False


def test_foundation_manifest_blocks_generic_transformers_qualification_downgrade(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())

    class Backend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("generic backend must not load for a declared foundation model")

    monkeypatch.setattr("generalist_lm.hf_backend.LocalTransformersBackend", Backend)

    with pytest.raises(ValueError, match="use qualify_foundation_model"):
        qualification.qualify_transformers_model(model)

    generic = model / ".airi-qualification.json"
    generic.write_text("{}", encoding="utf-8")
    status = qualification.transformers_qualification_status(model)
    assert status["qualified"] is False
    assert status["reason"] == "foundation_model_requires_foundation_qualification"


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


def test_transformers_digest_cache_detects_same_size_tamper_with_restored_mtime(tmp_path: Path):
    from generalist_lm.qualification import transformers_model_digest

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())
    weight = model / "model.safetensors"
    original = weight.stat()
    before = transformers_model_digest(model)

    replacement = b"FOUNDATION-WEIGHTS"
    assert len(replacement) == weight.stat().st_size
    weight.write_bytes(replacement)
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))

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

def test_foundation_qualification_forwards_hardware_policy(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    _stub_transformers_preflight(monkeypatch)

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())
    seen = {}

    class Backend:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

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

    result = qualification.qualify_foundation_model(
        model,
        device="cpu",
        device_map="auto",
        torch_dtype="bfloat16",
        max_memory={"cpu": "8GiB"},
        offload_folder=tmp_path / "offload",
    )
    assert result["qualified"] is True
    assert seen["device"] == "cpu"
    assert seen["device_map"] == "auto"
    assert seen["torch_dtype"] == "bfloat16"
    assert seen["max_memory"] == {"cpu": "8GiB"}
    assert seen["offload_folder"] == tmp_path / "offload"
    assert result["load_policy"]["device_map"] == "auto"
    assert result["load_policy"]["torch_dtype"] == "bfloat16"
    assert result["inference_profile"]["torch_dtype"] == "bfloat16"


def test_cli_foundation_qualification_exposes_hardware_options(tmp_path: Path):
    from generalist_lm.cli import parser

    args = parser().parse_args([
        "qualify-foundation",
        str(tmp_path / "model"),
        "--device", "cuda:0",
        "--device-map", "balanced",
        "--torch-dtype", "bfloat16",
        "--max-memory-json", '{"0":"12GiB","cpu":"24GiB"}',
        "--offload-folder", str(tmp_path / "offload"),
    ])
    assert args.device == "cuda:0"
    assert args.device_map == "balanced"
    assert args.torch_dtype == "bfloat16"
    assert args.max_memory_json == '{"0":"12GiB","cpu":"24GiB"}'

def test_foundation_minimum_score_cannot_be_weakened(tmp_path: Path):
    from generalist_lm import qualification

    with pytest.raises(ValueError, match="between 90.0 and 100.0"):
        qualification.qualify_foundation_model(
            tmp_path / "unused",
            minimum_score=89.999,
        )


def test_foundation_attestation_v3_requires_inference_profile(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    _stub_transformers_preflight(monkeypatch)

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
    result = qualification.qualify_foundation_model(model)
    assert result["foundation_qualification_version"] == 3
    assert result["inference_profile"] == {"torch_dtype": "auto"}
    assert qualification.foundation_qualification_status(model)["qualified"] is True

    path = model / ".airi-foundation-qualification.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("inference_profile")
    path.write_text(json.dumps(raw), encoding="utf-8")
    status = qualification.foundation_qualification_status(model)
    assert status["qualified"] is False
    assert status["integrity_ok"] is False

def test_foundation_status_rejects_tampered_qualified_boolean(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    _stub_transformers_preflight(monkeypatch)

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
    qualification.qualify_foundation_model(model)
    path = model / ".airi-foundation-qualification.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    raw["qualified"] = True
    raw["report"]["score"] = 10.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    status = qualification.foundation_qualification_status(model)
    assert status["qualified"] is False
    assert status["integrity_ok"] is False
    assert status["qualification_semantics_ok"] is False


def test_foundation_status_requires_all_protected_domain_scores(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    _stub_transformers_preflight(monkeypatch)

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
    qualification.qualify_foundation_model(model)
    path = model / ".airi-foundation-qualification.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["report"]["domain_scores"].pop("robustness")
    path.write_text(json.dumps(raw), encoding="utf-8")
    status = qualification.foundation_qualification_status(model)
    assert status["qualified"] is False
    assert status["qualification_semantics_ok"] is False

def _stub_transformers_preflight(monkeypatch):
    from generalist_lm import foundation_probe

    monkeypatch.setattr(
        foundation_probe,
        "_transformers_config_probe",
        lambda root: {
            "installed": True,
            "version": "test",
            "config_loadable_without_remote_code": True,
            "resolved_config_class": "GPT2Config",
        },
    )


def test_foundation_preflight_accepts_metadata_only_candidate(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is True
    assert result["preflight_version"] == 1
    assert result["config"]["model_type"] == "gpt2"
    assert result["config"]["context_limit"] == 4096
    assert result["weights"]["weight_files"] == ["model.safetensors"]
    assert result["weights"]["weight_bytes"] == len(b"foundation-weights")
    assert result["hardware"] is None
    assert "chat_template_not_declared" in result["warnings"]


def test_foundation_preflight_validates_sharded_weight_index(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "model")
    (model / "model.safetensors").unlink()
    (model / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (model / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 18},
        "weight_map": {
            "model.embed.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
    }), encoding="utf-8")
    write_foundation_manifest(model, _manifest())
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is True
    assert result["weights"]["sharded"] is True
    assert result["weights"]["index"] == "model.safetensors.index.json"
    assert result["weights"]["weight_files"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_foundation_preflight_rejects_missing_indexed_shard(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "model")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        },
    }), encoding="utf-8")
    write_foundation_manifest(model, _manifest())
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is False
    assert any("missing shard" in row for row in result["blockers"])


def test_foundation_preflight_rejects_weight_index_escape(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    model = _fake_model(tmp_path / "model")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "safe": "model.safetensors",
            "escape": "../outside.safetensors",
        },
    }), encoding="utf-8")
    write_foundation_manifest(model, _manifest())
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is False
    assert any("escapes model directory" in row for row in result["blockers"])


def test_foundation_preflight_rejects_manifest_quantization_drift(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "model")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {"quant_method": "mxfp4"}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_foundation_manifest(model, _manifest(quantization="none"))
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is False
    assert "manifest_quantization_mismatch" in result["blockers"]


def test_foundation_preflight_flags_gpt_oss_harmony_requirement(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "gpt-oss")
    config = {
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "max_position_embeddings": 131072,
        "quantization_config": {"quant_method": "mxfp4"},
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "chat_template.jinja").write_text(
        "<|start|>{{ messages[0]['content'] }}<|end|>",
        encoding="utf-8",
    )
    write_foundation_manifest(
        model,
        _manifest(
            model_id="openai/gpt-oss-20b",
            architecture="decoder-only-moe",
            context_length=131072,
            parameter_count=21_000_000_000,
            dtype="bfloat16",
            quantization="mxfp4",
        ),
    )
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is False
    assert result["protocol_requirements"] == ["harmony"]
    assert result["chat_template"]["available"] is True
    assert "gpt_oss_harmony_adapter_required" in result["blockers"]
    assert result["quantization"]["detected"] == "mxfp4"
    assert "mxfp4" in result["special_runtime"]


def test_foundation_preflight_parses_gpu_cpu_memory_budget(tmp_path: Path, monkeypatch):
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())
    _stub_transformers_preflight(monkeypatch)

    result = foundation_preflight(
        model,
        max_memory={0: "14GiB", "cpu": "32GiB"},
        probe_hardware=False,
    )
    assert result["ok"] is True
    assert result["memory_budget"]["declared"] is True
    assert result["memory_budget"]["devices"]["0"] == 14 * 1024**3
    assert result["memory_budget"]["devices"]["cpu"] == 32 * 1024**3


def test_foundation_preflight_never_loads_model_tensors(tmp_path: Path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from generalist_lm.foundation_probe import foundation_preflight

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must never load model tensors")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    result = foundation_preflight(model, probe_hardware=False)
    assert result["ok"] is True


def test_foundation_qualification_refuses_preflight_blocker_before_backend(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    model = _fake_model(tmp_path / "gpt-oss")
    (model / "config.json").write_text(json.dumps({
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "max_position_embeddings": 4096,
        "quantization_config": {"quant_method": "mxfp4"},
    }), encoding="utf-8")
    (model / "chat_template.jinja").write_text("template", encoding="utf-8")
    write_foundation_manifest(
        model,
        _manifest(
            model_id="openai/gpt-oss-test",
            quantization="mxfp4",
        ),
    )
    _stub_transformers_preflight(monkeypatch)

    class Backend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("blocked preflight must prevent backend construction")

    monkeypatch.setattr("generalist_lm.hf_backend.LocalTransformersBackend", Backend)
    with pytest.raises(ValueError, match="gpt_oss_harmony_adapter_required"):
        qualification.qualify_foundation_model(model)


def test_foundation_attestation_v3_requires_preflight_record(tmp_path: Path, monkeypatch):
    from generalist_lm import qualification

    model = _fake_model(tmp_path / "model")
    write_foundation_manifest(model, _manifest())
    _stub_transformers_preflight(monkeypatch)

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
    result = qualification.qualify_foundation_model(model)
    assert result["foundation_qualification_version"] == 3
    assert result["preflight"]["ok"] is True
    assert qualification.foundation_qualification_status(model)["qualified"] is True

    path = model / ".airi-foundation-qualification.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("preflight")
    path.write_text(json.dumps(raw), encoding="utf-8")
    status = qualification.foundation_qualification_status(model)
    assert status["qualified"] is False
    assert status["preflight_ok"] is False


def test_cli_exposes_foundation_preflight_command(tmp_path: Path):
    from generalist_lm.cli import parser

    args = parser().parse_args([
        "foundation-preflight",
        str(tmp_path / "model"),
        "--max-memory-json", '{"0":"12GiB","cpu":"24GiB"}',
        "--no-hardware",
    ])
    assert args.cmd == "foundation-preflight"
    assert args.no_hardware is True
    assert args.max_memory_json == '{"0":"12GiB","cpu":"24GiB"}'

