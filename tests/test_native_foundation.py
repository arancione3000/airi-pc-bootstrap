from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


def test_native_config_validates_gqa_and_rope():
    from generalist_lm.native_foundation import NativeFoundationConfig

    cfg = NativeFoundationConfig(
        vocab_size=512,
        context_length=512,
        d_model=128,
        n_heads=8,
        n_kv_heads=2,
        n_layers=4,
        d_ff=384,
    ).validate()
    assert cfg.d_model // cfg.n_heads == 16
    assert cfg.n_heads // cfg.n_kv_heads == 4

    with pytest.raises(ValueError, match="n_heads must be divisible"):
        NativeFoundationConfig(d_model=120, n_heads=6, n_kv_heads=4).validate()


def test_native_profiles_scale_without_instantiating_large_models():
    from generalist_lm.native_foundation import native_parameter_count, native_scale_profile

    one = native_scale_profile("1b")
    three = native_scale_profile("3b")
    seven = native_scale_profile("7b")
    counts = [native_parameter_count(row) for row in (one, three, seven)]
    assert counts[0] < counts[1] < counts[2]
    assert 1_000_000_000 <= counts[0] <= 1_300_000_000
    assert 3_000_000_000 <= counts[1] <= 3_700_000_000
    assert 6_800_000_000 <= counts[2] <= 7_400_000_000
    assert one.tokenizer_version == "bpe-v1"
    assert seven.context_length == 32768


def test_native_parameter_estimate_matches_real_model():
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import (
        NativeFoundationConfig,
        NativeFoundationLM,
        native_parameter_count,
    )

    for bias in (False, True):
        cfg = NativeFoundationConfig(
            vocab_size=128,
            context_length=32,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            d_ff=96,
            bias=bias,
        ).validate()
        model = NativeFoundationLM(cfg)
        actual = sum(int(parameter.numel()) for parameter in model.parameters())
        assert native_parameter_count(cfg) == actual


def test_native_model_forward_loss_and_compact_kv_cache():
    torch = pytest.importorskip("torch")
    from generalist_lm.native_foundation import NativeFoundationConfig, NativeFoundationLM

    cfg = NativeFoundationConfig(
        vocab_size=300,
        context_length=64,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=192,
    ).validate()
    model = NativeFoundationLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12), dtype=torch.long)
    out = model(ids, labels=ids, use_cache=True)
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    assert len(out["past_key_values"]) == cfg.n_layers
    key, value = out["past_key_values"][0]
    assert key.shape == (2, cfg.n_kv_heads, 12, cfg.d_model // cfg.n_heads)
    assert value.shape == key.shape

    next_ids = torch.randint(0, cfg.vocab_size, (2, 1), dtype=torch.long)
    cached = model(next_ids, past_key_values=out["past_key_values"], use_cache=True)
    assert cached["logits"].shape == (2, 1, cfg.vocab_size)
    assert cached["past_key_values"][0][0].shape[-2] == 13


def test_native_generate_uses_own_model_without_transformers_dependency():
    torch = pytest.importorskip("torch")
    from generalist_lm.native_foundation import NativeFoundationConfig, NativeFoundationLM

    cfg = NativeFoundationConfig(
        vocab_size=128,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=96,
    ).validate()
    model = NativeFoundationLM(cfg)
    ids = torch.tensor([[1, 10, 11]], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=3, eos_token_id=None)
    assert out.shape == (1, 6)


def test_native_module_has_no_pretrained_loader_dependency():
    import inspect
    import generalist_lm.native_foundation as native

    source = inspect.getsource(native).lower()
    assert "from_pretrained" not in source
    assert "huggingface" not in source
    assert "transformers" not in source


def test_native_root_api_has_no_pretrained_or_source_model_input():
    from generalist_lm.native_foundation import create_native_root_checkpoint

    names = set(inspect.signature(create_native_root_checkpoint).parameters)
    assert "pretrained" not in names
    assert "source_model" not in names
    assert "model_path" not in names
    assert names == {"state_dir", "config", "root_seed", "tokenizer_path"}


def test_native_root_checkpoint_is_random_init_and_loadable(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from generalist_lm.native_foundation import (
        NativeFoundationConfig,
        create_native_root_checkpoint,
        load_native_checkpoint,
        native_checkpoint_status,
    )

    cfg = NativeFoundationConfig(
        vocab_size=128,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=96,
        tokenizer_version="byte-v1",
    ).validate()
    root = tmp_path / "native"
    created = create_native_root_checkpoint(root, cfg, root_seed=1234)
    assert created["ok"] is True
    assert created["weights_origin"] == "random-init"
    assert created["external_pretrained"] is False
    assert created["root_seed"] == 1234
    assert created["tokenizer_version"] == "byte-v1"
    assert created["tokenizer_digest"] is None

    model, loaded_cfg, status = load_native_checkpoint(root)
    assert status["ok"] is True
    assert loaded_cfg.to_dict() == cfg.to_dict()
    ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    assert model(ids)["logits"].shape == (1, 4, cfg.vocab_size)
    assert native_checkpoint_status(root)["checkpoint_digest"] == status["checkpoint_digest"]


def test_native_root_seed_is_reproducible(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from generalist_lm.native_foundation import (
        NativeFoundationConfig,
        create_native_root_checkpoint,
        load_native_checkpoint,
    )

    cfg = NativeFoundationConfig(
        vocab_size=96,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=96,
    ).validate()
    create_native_root_checkpoint(tmp_path / "a", cfg, root_seed=77)
    create_native_root_checkpoint(tmp_path / "b", cfg, root_seed=77)
    create_native_root_checkpoint(tmp_path / "c", cfg, root_seed=78)

    a, _, _ = load_native_checkpoint(tmp_path / "a")
    b, _, _ = load_native_checkpoint(tmp_path / "b")
    c, _, _ = load_native_checkpoint(tmp_path / "c")

    a_state = a.state_dict()
    b_state = b.state_dict()
    c_state = c.state_dict()
    assert all(torch.equal(a_state[name], b_state[name]) for name in a_state)
    assert any(not torch.equal(a_state[name], c_state[name]) for name in a_state)


def test_native_status_rejects_tampered_weights(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import (
        NATIVE_MODEL_FILENAME,
        NativeFoundationConfig,
        create_native_root_checkpoint,
        native_checkpoint_status,
    )

    root = tmp_path / "native"
    create_native_root_checkpoint(
        root,
        NativeFoundationConfig(
            vocab_size=96,
            context_length=32,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=96,
        ),
        root_seed=5,
    )
    model_path = root / NATIVE_MODEL_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    status = native_checkpoint_status(root)
    assert status["ok"] is False


def test_native_status_rejects_forged_external_pretrained_manifest(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import (
        NATIVE_MANIFEST_FILENAME,
        NativeFoundationConfig,
        create_native_root_checkpoint,
        native_checkpoint_status,
    )

    root = tmp_path / "native"
    create_native_root_checkpoint(
        root,
        NativeFoundationConfig(
            vocab_size=96,
            context_length=32,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=96,
        ),
        root_seed=5,
    )
    manifest = root / NATIVE_MANIFEST_FILENAME
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["external_pretrained"] = True
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    status = native_checkpoint_status(root)
    assert status["ok"] is False
    assert "external pretrained weights are forbidden" in status["reason"]


def test_native_bpe_root_binds_tokenizer_digest(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import (
        NATIVE_TOKENIZER_FILENAME,
        NativeFoundationConfig,
        create_native_root_checkpoint,
        native_checkpoint_status,
    )

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version":"bpe-v1","merges":[]}', encoding="utf-8")
    root = tmp_path / "native"
    cfg = NativeFoundationConfig(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=96,
        tokenizer_version="bpe-v1",
    )
    created = create_native_root_checkpoint(
        root,
        cfg,
        root_seed=9,
        tokenizer_path=tokenizer,
    )
    assert created["ok"] is True
    assert len(created["tokenizer_digest"]) == 64

    (root / NATIVE_TOKENIZER_FILENAME).write_text('{"tampered":true}', encoding="utf-8")
    assert native_checkpoint_status(root)["ok"] is False


def test_native_bpe_root_requires_own_tokenizer_artifact(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.native_foundation import NativeFoundationConfig, create_native_root_checkpoint

    cfg = NativeFoundationConfig(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=96,
        tokenizer_version="bpe-v1",
    )
    with pytest.raises(ValueError, match="requires an AIRI tokenizer"):
        create_native_root_checkpoint(tmp_path / "native", cfg, root_seed=1)


def test_native_cli_plans_without_allocating_weights():
    from generalist_lm.cli import parser

    args = parser().parse_args([
        "native-foundation-plan",
        "7b",
        "--vocab-size",
        "32768",
    ])
    assert args.cmd == "native-foundation-plan"
    assert args.profile == "7b"
    assert args.vocab_size == 32768


def test_native_cli_init_has_large_allocation_guard(tmp_path: Path):
    from generalist_lm.cli import parser

    args = parser().parse_args([
        "native-foundation-init",
        str(tmp_path / "native"),
        "--profile",
        "1b",
        "--seed",
        "42",
    ])
    assert args.cmd == "native-foundation-init"
    assert args.profile == "1b"
    assert args.max_init_parameters == 100_000_000
    assert args.allow_large_init is False
