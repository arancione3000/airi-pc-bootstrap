from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.benchmarks import default_suite, run_benchmark
from generalist_lm.evolution import GeneralistGenome, generate_challengers, promotion_decision, weakness_signals
from generalist_lm.hf_backend import LocalTransformersBackend
from generalist_lm.mathesis_bridge import mathesis_signals
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig, parameter_count
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import ASSISTANT, BOS, ByteTokenizer
from generalist_lm.tool_protocol import parse_tool_call
from generalist_lm.training import SFTExample, train_sft


def tiny_config():
    return GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()


def test_byte_tokenizer_roundtrip_and_chat_roles():
    tok = ByteTokenizer()
    text = "Ciao 🌸 — hello!"
    assert tok.decode(tok.encode(text)) == text
    ids = tok.serialize_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "ciao"},
    ])
    assert ids[0] == BOS
    assert ids[-1] == ASSISTANT
    with pytest.raises(ValueError):
        tok.serialize_messages([{"role": "root", "content": "x"}])


def test_causal_lm_forward_loss_and_generation_shape():
    torch = pytest.importorskip("torch")
    cfg = tiny_config()
    model = CausalTransformerLM(cfg)
    ids = torch.randint(8, cfg.vocab_size, (2, 16))
    labels = ids.clone()
    result = model(ids, labels=labels)
    assert tuple(result["logits"].shape) == (2, 16, cfg.vocab_size)
    assert torch.isfinite(result["loss"])
    generated = model.generate(ids[:, :4], max_new_tokens=3, eos_token_id=None)
    assert tuple(generated.shape) == (2, 7)
    assert parameter_count(model) > 10_000


def test_sft_actually_reduces_language_model_loss():
    pytest.importorskip("torch")
    cfg = tiny_config()
    model = CausalTransformerLM(cfg)
    tok = ByteTokenizer()
    examples = [
        SFTExample([
            {"role": "user", "content": "Reply OK"},
            {"role": "assistant", "content": "OK"},
        ]),
        SFTExample([
            {"role": "user", "content": "Reply HI"},
            {"role": "assistant", "content": "HI"},
        ]),
    ]
    report = train_sft(
        model,
        tok,
        examples,
        steps=40,
        batch_size=2,
        learning_rate=8e-3,
        weight_decay=0.0,
        seed=11,
    )
    assert report["ok"] is True
    assert report["final_loss"] < report["initial_loss"]
    assert report["loss_improvement"] > 0


def test_checkpoint_roundtrip(tmp_path: Path):
    torch = pytest.importorskip("torch")
    runtime = GeneralistRuntime.fresh(tiny_config())
    metadata = runtime.save_checkpoint(tmp_path, metadata={"purpose": "test"})
    assert metadata["format"] == "airi-generalist-lm-v1"
    restored = GeneralistRuntime.from_checkpoint(tmp_path)
    assert restored.config.to_dict() == runtime.config.to_dict()
    assert parameter_count(restored.model) == parameter_count(runtime.model)
    first = next(runtime.model.parameters()).detach().cpu()
    second = next(restored.model.parameters()).detach().cpu()
    assert torch.equal(first, second)


def test_tool_protocol_is_strictly_allowlisted():
    call = parse_tool_call(
        '<tool_call>{"name":"calculator","arguments":{"expression":"17*19"}}</tool_call>',
        allowed_tools={"calculator"},
    )
    assert call.name == "calculator"
    assert call.arguments["expression"] == "17*19"
    with pytest.raises(PermissionError):
        parse_tool_call(
            '{"name":"shell","arguments":{"command":"rm -rf /"}}',
            allowed_tools={"calculator"},
        )
    with pytest.raises(ValueError):
        parse_tool_call("not-json", allowed_tools={"calculator"})


class ScriptedBackend:
    def generate(self, prompt: str, *, max_new_tokens: int = 192) -> str:
        del max_new_tokens
        if "ORANGE" in prompt:
            return "ORANGE"
        if "CIAO" in prompt:
            return "CIAO"
        if "defining add" in prompt:
            return "def add(a, b):\n    return a + b"
        if "arithmetic mean" in prompt:
            return "4"
        if "Compute 17*19" in prompt:
            return "323"
        if "calculator" in prompt:
            return '<tool_call>{"name":"calculator","arguments":{"expression":"17*19"}}</tool_call>'
        if "JSON object" in prompt:
            return '{"ok":true,"items":3}'
        raise AssertionError(prompt)


def test_generalist_benchmark_covers_language_code_data_reasoning_tools():
    report = run_benchmark(ScriptedBackend())
    assert report["ok"] is True
    assert report["score"] == 100.0
    assert set(report["domain_scores"]) == {
        "language", "coding", "data", "reasoning", "tools", "structured"
    }
    assert report["critical_failures"] == []


def test_generalist_promotion_is_domain_regression_gated():
    champion = {
        "score": 80.0,
        "domain_scores": {"language": 1.0, "coding": 0.5, "data": 0.5},
        "critical_failures": [],
    }
    candidate = {
        "score": 84.0,
        "domain_scores": {"language": 1.0, "coding": 1.0, "data": 0.5},
        "critical_failures": [],
    }
    ok, reason = promotion_decision(champion, candidate, minimum_gain=2.0)
    assert ok is True and "gain" in reason

    regressed = {
        "score": 90.0,
        "domain_scores": {"language": 0.0, "coding": 1.0, "data": 1.0},
        "critical_failures": [],
    }
    ok, reason = promotion_decision(champion, regressed, minimum_gain=2.0)
    assert ok is False and "regressed" in reason


def test_generalist_genome_is_bounded_and_generates_distinct_challengers():
    champion = GeneralistGenome()
    challengers = generate_challengers(
        champion,
        signals=["coding_gap", "data_gap", "symbolic_reasoning_signal"],
        count=4,
    )
    assert len(challengers) == 4
    assert all(row.generation == 1 for row in challengers)
    assert all(row.parent_id == champion.genome_id for row in challengers)
    assert len({json.dumps(row.to_dict(), sort_keys=True) for row in challengers}) == 4
    with pytest.raises(ValueError):
        GeneralistGenome(tokenizer_version="download-executable-tokenizer").validate()


def test_weakness_signals_are_semantic():
    report = {
        "domain_scores": {"coding": 0.0, "data": 1.0, "tools": 0.0, "reasoning": 0.5}
    }
    assert weakness_signals(report) == ["coding_gap", "tool_gap", "reasoning_gap"]


def test_mathesis_bridge_accepts_only_current_proof_gated_items(tmp_path: Path):
    (tmp_path / "discoveries.json").write_text(json.dumps({
        "theorems": {
            "valid": {
                "verified": True,
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": True},
            },
            "metadata_only": {
                "verified": True,
                "certificate": {"ok": True},
            },
            "false_cert": {
                "verified": True,
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": False},
            },
        }
    }), encoding="utf-8")
    (tmp_path / "curriculum.json").write_text('{"cursor":12}', encoding="utf-8")
    (tmp_path / "champion.json").write_text('{"generation":15,"symbolic_depth":12}', encoding="utf-8")
    result = mathesis_signals(tmp_path)
    assert result["verified_math_items"] == 1
    assert result["mathesis_generation"] == 15
    assert "symbolic_reasoning_signal" in result["signals"]
    assert "research_curriculum_signal" in result["signals"]


def test_local_transformers_backend_refuses_missing_or_remote_model_paths(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LocalTransformersBackend(tmp_path / "not-downloaded")
