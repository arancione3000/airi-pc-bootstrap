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


def test_mathesis_bridge_reproves_persisted_math_instead_of_trusting_metadata(tmp_path: Path):
    (tmp_path / "discoveries.json").write_text(json.dumps({
        "theorems": {
            "valid": {
                "id": "valid",
                "statement": "x^2-y^2 = (x-y)*(x+y)",
                "verified": True,
                "strategy": "difference_of_powers",
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": True, "status": "verified"},
            },
            "forged_verified": {
                "id": "forged_verified",
                "statement": "x+1=x+2",
                "verified": True,
                "strategy": "difference_of_powers",
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": True, "status": "verified"},
            },
            "metadata_only": {
                "id": "metadata_only",
                "statement": "x^2=x*x",
                "verified": True,
                "certificate": {"ok": True},
            },
            "false_cert": {
                "id": "false_cert",
                "statement": "x^3=x*x*x",
                "verified": True,
                "strategy": "difference_of_powers",
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": False},
            },
        }
    }), encoding="utf-8")
    (tmp_path / "curriculum.json").write_text('{"cursor":12}', encoding="utf-8")
    (tmp_path / "champion.json").write_text('{"generation":15,"symbolic_depth":12}', encoding="utf-8")
    result = mathesis_signals(tmp_path)
    assert result["ok"] is True
    assert result["verifier_available"] is True
    assert result["verified_math_items"] == 1
    assert result["rejected_math_items"]["forged_verified"]
    assert result["mathesis_generation"] == 15
    assert "symbolic_reasoning_signal" in result["signals"]
    assert "deep_symbolic_signal" in result["signals"]
    assert "research_curriculum_signal" in result["signals"]


def test_mathesis_bridge_fails_closed_when_verifier_is_unavailable(tmp_path: Path, monkeypatch):
    import generalist_lm.mathesis_bridge as bridge

    (tmp_path / "discoveries.json").write_text(
        json.dumps({"theorems": {}}),
        encoding="utf-8",
    )
    (tmp_path / "curriculum.json").write_text('{"cursor":99}', encoding="utf-8")
    (tmp_path / "champion.json").write_text(
        '{"generation":15,"symbolic_depth":12}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge,
        "_current_verified_math_items",
        lambda theorems: ([], {}, False),
    )
    result = bridge.mathesis_signals(tmp_path)
    assert result["ok"] is False
    assert result["verifier_available"] is False
    assert result["signals"] == []


def test_local_transformers_backend_refuses_missing_or_remote_model_paths(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LocalTransformersBackend(tmp_path / "not-downloaded")


def test_generalist_agent_runs_allowlisted_tool_loop():
    from generalist_lm.agent import GeneralistAgent

    class Backend:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, *, max_new_tokens=256):
            del messages, max_new_tokens
            self.calls += 1
            if self.calls == 1:
                return '<tool_call>{"name":"calculator","arguments":{"expression":"17*19"}}</tool_call>'
            return "323"

    executed = []

    def executor(name, arguments):
        executed.append((name, arguments))
        assert name == "calculator"
        assert arguments == {"expression": "17*19"}
        return {"value": 323}

    agent = GeneralistAgent(
        Backend(),
        tools={
            "calculator": {
                "description": "Evaluate arithmetic",
                "schema": {"type": "object"},
            }
        },
        executor=executor,
        max_steps=4,
    )
    run = agent.run([{"role": "user", "content": "What is 17*19?"}])
    assert run.ok is True
    assert run.answer == "323"
    assert run.steps == 2
    assert len(run.tool_calls) == 1
    assert executed == [("calculator", {"expression": "17*19"})]


def test_generalist_agent_never_executes_unknown_tool():
    from generalist_lm.agent import GeneralistAgent

    class Backend:
        def chat(self, messages, *, max_new_tokens=256):
            del messages, max_new_tokens
            return '<tool_call>{"name":"shell","arguments":{"command":"danger"}}</tool_call>'

    executed = []
    agent = GeneralistAgent(
        Backend(),
        tools={"calculator": {"description": "math", "schema": {}}},
        executor=lambda name, args: executed.append((name, args)),
    )
    with pytest.raises(PermissionError):
        agent.run([{"role": "user", "content": "do something"}])
    assert executed == []


def test_generalist_agent_stops_after_bounded_steps():
    from generalist_lm.agent import GeneralistAgent

    class Backend:
        def chat(self, messages, *, max_new_tokens=256):
            del messages, max_new_tokens
            return '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>'

    agent = GeneralistAgent(
        Backend(),
        tools={"calculator": {"description": "math", "schema": {}}},
        executor=lambda name, args: {"value": 2},
        max_steps=3,
    )
    run = agent.run([{"role": "user", "content": "loop"}])
    assert run.ok is False
    assert run.reason == "max_steps_exhausted"
    assert run.steps == 3
    assert len(run.tool_calls) == 3


def test_research_promotion_rejects_hidden_domain_regression():
    from generalist_lm.research_cycle import _research_eligible

    champion = {
        "loss": 1.00,
        "domain_loss": {"language": 0.9, "coding": 1.1, "data": 1.0},
        "finite": True,
    }
    candidate = {
        "loss": 0.80,
        "domain_loss": {"language": 1.3, "coding": 0.6, "data": 0.5},
        "finite": True,
    }
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.02,
        max_domain_regression=0.10,
    )
    assert ok is False
    assert "regressed" in reason


def test_research_cycle_persists_loadable_research_champion(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle
    from generalist_lm.research_health import research_health

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STATE", str(tmp_path))
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "3")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    first = run_research_cycle(tmp_path)
    assert first["ok"] is True
    assert first["cycle"] == 1
    assert first["bootstrapped"] is True
    assert first["policy"]["research_only"] is True
    assert (tmp_path / "champion" / "model.pt").exists()
    assert (tmp_path / "champion-genome.json").exists()
    assert research_health(tmp_path)["ok"] is True

    second = run_research_cycle(tmp_path)
    assert second["ok"] is True
    assert second["cycle"] == 2
    assert second["bootstrapped"] is False
    assert second["promoted"] is False
    assert research_health(tmp_path)["ok"] is True


def test_research_health_rejects_genome_checkpoint_mismatch(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle
    from generalist_lm.research_health import research_health

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    run_research_cycle(tmp_path)

    genome_path = tmp_path / "champion-genome.json"
    genome = json.loads(genome_path.read_text(encoding="utf-8"))
    genome["context_length"] *= 2
    genome_path.write_text(json.dumps(genome), encoding="utf-8")

    report = research_health(tmp_path)
    assert report["ok"] is False
    assert any(row["name"] == "checkpoint:context_match" for row in report["failed"])


def test_local_transformers_backend_loads_real_local_causal_model(tmp_path: Path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from generalist_lm.hf_backend import LocalTransformersBackend

    vocab = {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.save_pretrained(tmp_path)

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=1,
    )
    GPT2LMHeadModel(config).save_pretrained(tmp_path)

    backend = LocalTransformersBackend(tmp_path, device="cpu", local_files_only=True)
    output = backend.generate("hello", max_new_tokens=2)
    assert isinstance(output, str)


def test_parameter_estimator_matches_real_native_model():
    pytest.importorskip("torch")
    from generalist_lm.model import estimate_parameter_count

    cfg = tiny_config()
    model = CausalTransformerLM(cfg)
    assert estimate_parameter_count(cfg) == parameter_count(model)


def test_research_budget_blocks_runaway_architecture_before_training():
    from generalist_lm.research_cycle import _research_budget_reason

    huge = GeneralistGenome(
        context_length=8192,
        d_model=4096,
        n_heads=64,
        n_layers=96,
        d_ff=16384,
    ).validate()
    reason = _research_budget_reason(
        huge,
        max_params=5_000_000,
        max_context=512,
        max_width=256,
        max_layers=6,
    )
    assert reason is not None
    assert "exceeds research max" in reason


def test_qualification_suite_is_broader_than_research_smoke_and_has_core_per_domain():
    from generalist_lm.benchmarks import qualification_suite

    tasks = qualification_suite()
    assert len(tasks) >= 20
    domains = {task.domain for task in tasks}
    assert domains == {"language", "coding", "data", "reasoning", "tools", "structured"}
    for domain in domains:
        assert any(task.domain == domain and task.critical for task in tasks)


def test_strategy_genes_change_actual_training_curriculum():
    from dataclasses import replace
    from generalist_lm.research_cycle import _training_rows_for_genome, research_seed

    seed = research_seed()
    base = _training_rows_for_genome(seed)

    coding = replace(seed, code_adapter=True)
    coding_rows = _training_rows_for_genome(coding)
    assert len(coding_rows) > len(base)
    assert sum(row.domain == "coding" for row in coding_rows) > sum(row.domain == "coding" for row in base)

    symbolic = replace(seed, symbolic_adapter=True, reasoning_depth=3)
    symbolic_rows = _training_rows_for_genome(symbolic)
    assert sum(row.domain == "reasoning" for row in symbolic_rows) > sum(row.domain == "reasoning" for row in base)


def test_curriculum_train_and_validation_are_disjoint_and_multi_domain():
    from generalist_lm.curriculum import curriculum_manifest
    manifest = curriculum_manifest()
    assert manifest["train_rows"] >= 40
    assert manifest["validation_rows"] >= 30
    assert manifest["prompt_overlap"] == []
    assert manifest["mechanically_labeled"] is True
    assert all(count >= 5 for count in manifest["train_domains"].values())
    assert all(count >= 5 for count in manifest["validation_domains"].values())


def test_research_promotion_rejects_generalist_forgetting():
    from generalist_lm.research_cycle import _research_eligible

    champion = {
        "loss": 1.0,
        "domain_loss": {"language": 1.0, "coding": 1.0},
        "finite": True,
        "target_token_accuracy": 0.40,
        "domain_token_accuracy": {"language": 0.4, "coding": 0.4},
        "solved_items": ["language:aaa", "coding:bbb"],
    }
    candidate = {
        "loss": 0.7,
        "domain_loss": {"language": 0.8, "coding": 0.8},
        "finite": True,
        "target_token_accuracy": 0.45,
        "domain_token_accuracy": {"language": 0.45, "coding": 0.45},
        "solved_items": ["language:aaa"],
    }
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.02,
        max_domain_regression=0.10,
    )
    assert ok is False
    assert "forgot" in reason


def test_research_promotion_accepts_loss_gain_with_retained_solutions():
    from generalist_lm.research_cycle import _research_eligible

    champion = {
        "loss": 1.0,
        "domain_loss": {"language": 1.0, "coding": 1.0},
        "finite": True,
        "target_token_accuracy": 0.40,
        "domain_token_accuracy": {"language": 0.4, "coding": 0.4},
        "solved_items": ["language:aaa"],
    }
    candidate = {
        "loss": 0.7,
        "domain_loss": {"language": 0.8, "coding": 0.8},
        "finite": True,
        "target_token_accuracy": 0.50,
        "domain_token_accuracy": {"language": 0.5, "coding": 0.5},
        "solved_items": ["language:aaa", "coding:bbb"],
    }
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.02,
        max_domain_regression=0.10,
    )
    assert ok is True
    assert "anti-forgetting" in reason


def test_generalist_continuum_has_parallel_free_speed_swarm_and_nonforce_state_push():
    workflow = (ROOT / ".github" / "workflows" / "generalist-continuum.yml").read_text(encoding="utf-8")
    assert "group: airi-generalist-free-speed" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "max-parallel: 8" in workflow
    assert "max-parallel: 4" in workflow
    assert "max-parallel: 2" in workflow
    assert "--survivors 4" in workflow
    assert "--survivors 2" in workflow
    assert "--repeat-seeds 2" in workflow
    assert "python -m generalist_lm.generalist_swarm finalize" in workflow
    assert "candidate_can_self_promote == false" in workflow
    assert "git push --quiet origin HEAD:generalist-state" in workflow
    assert "git push --force" not in workflow
    assert "git push -f" not in workflow


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("position_encoding", ["learned", "sinusoidal"])
@pytest.mark.parametrize("ff_variant", ["swiglu", "gelu"])
def test_evolvable_transformer_variants_are_real_and_parameter_estimator_exact(
    norm_type,
    position_encoding,
    ff_variant,
):
    pytest.importorskip("torch")
    from generalist_lm.model import estimate_parameter_count

    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        norm_type=norm_type,
        position_encoding=position_encoding,
        ff_variant=ff_variant,
    ).validate()
    model = CausalTransformerLM(cfg)
    assert parameter_count(model) == estimate_parameter_count(cfg)


def test_generalist_exploration_rotates_without_changing_weakness_priority():
    champion = GeneralistGenome(
        code_adapter=False,
        data_adapter=False,
        retrieval_adapter=False,
        symbolic_adapter=False,
        reasoning_depth=1,
    ).validate()
    first = generate_challengers(champion, signals=[], count=2, exploration_offset=0)
    second = generate_challengers(champion, signals=[], count=2, exploration_offset=2)
    first_shapes = {(row.norm_type, row.position_encoding, row.ff_variant, row.d_model, row.n_layers) for row in first}
    second_shapes = {(row.norm_type, row.position_encoding, row.ff_variant, row.d_model, row.n_layers) for row in second}
    assert first_shapes != second_shapes

    directed = generate_challengers(
        champion,
        signals=["coding_gap"],
        count=1,
        exploration_offset=7,
    )[0]
    assert directed.code_adapter is True
    assert directed.reasoning_depth > champion.reasoning_depth


def test_genome_training_seed_is_stable_for_same_effective_architecture():
    from dataclasses import replace
    from generalist_lm.research_cycle import _genome_training_seed

    genome = GeneralistGenome().validate()
    renamed = replace(genome, genome_id="another-id", generation=99, parent_id="different-parent")
    assert _genome_training_seed(genome) == _genome_training_seed(renamed)

    changed = replace(genome, norm_type="rmsnorm")
    assert _genome_training_seed(genome) != _genome_training_seed(changed)


def test_benchmark_prefers_chat_interface_for_instruction_models():
    from generalist_lm.benchmarks import BenchmarkTask, exact, run_benchmark

    class ChatAwareBackend:
        def __init__(self):
            self.chat_calls = 0
            self.generate_calls = 0

        def chat(self, messages, *, max_new_tokens=192):
            self.chat_calls += 1
            assert messages == [{"role": "user", "content": "say READY"}]
            return "READY"

        def generate(self, prompt, *, max_new_tokens=192):
            self.generate_calls += 1
            return "WRONG"

    backend = ChatAwareBackend()
    report = run_benchmark(
        backend,
        [BenchmarkTask("language:chat-path", "language", "say READY", exact("READY"))],
    )
    assert report["ok"] is True
    assert backend.chat_calls == 1
    assert backend.generate_calls == 0


def _fake_qualified_attestation(path: Path, *, score: float, domain_scores: dict[str, float] | None = None):
    from generalist_lm.qualification import QUALIFICATION_VERSION, checkpoint_digest

    domains = domain_scores or {
        "language": 1.0,
        "coding": 1.0,
        "data": 1.0,
        "reasoning": 1.0,
        "tools": 1.0,
        "structured": 1.0,
    }
    payload = {
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-qualification-v2",
        "checkpoint_digest": checkpoint_digest(path),
        "qualified": True,
        "minimum_score": 85.0,
        "report": {
            "ok": True,
            "score": float(score),
            "critical_failures": [],
            "domain_scores": domains,
            "tasks": [],
        },
    }
    (path / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_production_promotion_is_digest_cached_and_transactional(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    import generalist_lm.production_promotion as promotion
    from generalist_lm.qualification import qualification_status

    research = tmp_path / "research"
    production = tmp_path / "production"
    runtime = GeneralistRuntime.fresh(tiny_config())
    runtime.save_checkpoint(research, metadata={"revision": 1})

    calls = {"count": 0}

    def fake_qualify(path, minimum_score=85.0):
        calls["count"] += 1
        return _fake_qualified_attestation(Path(path), score=90.0)

    monkeypatch.setattr(promotion, "qualify_checkpoint", fake_qualify)

    first = promotion.attempt_production_promotion(
        research,
        production,
        minimum_score=85.0,
        minimum_gain=2.0,
    )
    assert first["promoted"] is True
    assert first["qualified"] is True
    assert qualification_status(production)["qualified"] is True
    production_metadata = json.loads((production / "metadata.json").read_text(encoding="utf-8"))
    from generalist_lm.qualification import checkpoint_digest
    assert production_metadata["role"] == "production_champion"
    assert production_metadata["source_research_checkpoint_digest"] == checkpoint_digest(research)
    production_digest = qualification_status(production)["current_checkpoint_digest"]

    second = promotion.attempt_production_promotion(
        research,
        production,
        minimum_score=85.0,
        minimum_gain=2.0,
    )
    assert second["cached"] is True
    assert calls["count"] == 1
    assert qualification_status(production)["current_checkpoint_digest"] == production_digest


def test_production_promotion_rejects_domain_regression_and_preserves_old_champion(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    import generalist_lm.production_promotion as promotion
    from generalist_lm.qualification import qualification_status

    research = tmp_path / "research"
    production = tmp_path / "production"
    runtime = GeneralistRuntime.fresh(tiny_config())
    runtime.save_checkpoint(research, metadata={"revision": 1})

    scores = [
        (90.0, {
            "language": 1.0, "coding": 1.0, "data": 1.0,
            "reasoning": 1.0, "tools": 1.0, "structured": 1.0,
        }),
        (95.0, {
            "language": 0.5, "coding": 1.0, "data": 1.0,
            "reasoning": 1.0, "tools": 1.0, "structured": 1.0,
        }),
    ]

    def fake_qualify(path, minimum_score=85.0):
        score, domains = scores.pop(0)
        return _fake_qualified_attestation(Path(path), score=score, domain_scores=domains)

    monkeypatch.setattr(promotion, "qualify_checkpoint", fake_qualify)

    first = promotion.attempt_production_promotion(research, production)
    assert first["promoted"] is True
    old_digest = qualification_status(production)["current_checkpoint_digest"]

    # Change exact research checkpoint identity so the qualification cache cannot hide the second attempt.
    runtime.save_checkpoint(research, metadata={"revision": 2})
    second = promotion.attempt_production_promotion(
        research,
        production,
        minimum_gain=2.0,
        max_domain_regression=0.0,
    )
    assert second["qualified"] is True
    assert second["promoted"] is False
    assert second["eligible"] is False
    assert "regressed" in second["reason"]
    assert qualification_status(production)["current_checkpoint_digest"] == old_digest


def test_production_promotion_requires_meaningful_gain_over_existing_champion(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    import generalist_lm.production_promotion as promotion
    from generalist_lm.qualification import qualification_status

    research = tmp_path / "research"
    production = tmp_path / "production"
    runtime = GeneralistRuntime.fresh(tiny_config())
    runtime.save_checkpoint(research, metadata={"revision": 1})

    next_score = {"value": 90.0}

    def fake_qualify(path, minimum_score=85.0):
        return _fake_qualified_attestation(Path(path), score=next_score["value"])

    monkeypatch.setattr(promotion, "qualify_checkpoint", fake_qualify)
    assert promotion.attempt_production_promotion(research, production)["promoted"] is True
    old_digest = qualification_status(production)["current_checkpoint_digest"]

    runtime.save_checkpoint(research, metadata={"revision": 2})
    next_score["value"] = 91.0
    result = promotion.attempt_production_promotion(
        research,
        production,
        minimum_gain=2.0,
    )
    assert result["qualified"] is True
    assert result["promoted"] is False
    assert "promotion margin" in result["reason"]
    assert qualification_status(production)["current_checkpoint_digest"] == old_digest


def test_research_health_rejects_tampered_production_checkpoint(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    import generalist_lm.production_promotion as promotion
    from generalist_lm.research_cycle import run_research_cycle
    from generalist_lm.research_health import research_health

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")
    run_research_cycle(tmp_path)

    def fake_qualify(path, minimum_score=85.0):
        return _fake_qualified_attestation(Path(path), score=100.0)

    monkeypatch.setattr(promotion, "qualify_checkpoint", fake_qualify)
    promoted = promotion.attempt_production_promotion(
        tmp_path / "champion",
        tmp_path / "production",
    )
    assert promoted["promoted"] is True
    assert research_health(tmp_path)["ok"] is True

    metadata = json.loads((tmp_path / "production" / "metadata.json").read_text(encoding="utf-8"))
    metadata["tampered"] = True
    (tmp_path / "production" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    report = research_health(tmp_path)
    assert report["ok"] is False
    assert any(row["name"] == "production:qualified_integrity" for row in report["failed"])


def test_generalist_continuum_attempts_production_qualification_before_persist():
    workflow = (ROOT / ".github" / "workflows" / "generalist-continuum.yml").read_text(encoding="utf-8")
    assert "python -m generalist_lm.production_promotion" in workflow
    assert "AIRI_GENERALIST_PRODUCTION_MIN_SCORE: '85'" in workflow
    assert "AIRI_GENERALIST_PRODUCTION_MIN_GAIN: '2'" in workflow
    assert workflow.index("python -m generalist_lm.production_promotion") < workflow.index("Persist health-gated Free-Speed state")


def test_cached_promoted_digest_retries_when_production_integrity_is_lost(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    import generalist_lm.production_promotion as promotion
    from generalist_lm.qualification import qualification_status

    research = tmp_path / "research"
    production = tmp_path / "production"
    runtime = GeneralistRuntime.fresh(tiny_config())
    runtime.save_checkpoint(research, metadata={"revision": 1})

    calls = {"count": 0}

    def fake_qualify(path, minimum_score=85.0):
        calls["count"] += 1
        return _fake_qualified_attestation(Path(path), score=100.0)

    monkeypatch.setattr(promotion, "qualify_checkpoint", fake_qualify)
    first = promotion.attempt_production_promotion(research, production)
    assert first["promoted"] is True
    assert calls["count"] == 1

    metadata = json.loads((production / "metadata.json").read_text(encoding="utf-8"))
    metadata["corrupt"] = True
    (production / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert qualification_status(production)["qualified"] is False

    recovered = promotion.attempt_production_promotion(research, production)
    assert recovered["promoted"] is True
    assert recovered["cached"] is False
    assert calls["count"] == 2
    assert qualification_status(production)["qualified"] is True


def test_curriculum_memory_grows_idempotently_and_never_overlaps_validation(tmp_path: Path):
    from generalist_lm.curriculum_memory import CurriculumMemory

    memory = CurriculumMemory(tmp_path, max_rows=120)
    first = memory.expand(1, signals=["coding_gap", "symbolic_reasoning_signal"])
    assert first["ok"] is True
    assert first["added"] >= 6
    assert first["stored"] == first["added"]
    assert first["validation_overlap"] == []

    again = memory.expand(1, signals=["coding_gap", "symbolic_reasoning_signal"])
    assert again["added"] == 0
    assert again["stored"] == first["stored"]
    assert again["validation_overlap"] == []

    second = memory.expand(2, signals=["data_gap", "tool_gap"])
    assert second["stored"] > first["stored"]
    assert second["validation_overlap"] == []
    manifest = memory.manifest()
    assert manifest["stored"] == second["stored"]
    assert set(manifest["domains"]) == {"language", "coding", "data", "reasoning", "tools", "structured"}


def test_curriculum_memory_is_bounded_and_deduplicated(tmp_path: Path):
    from generalist_lm.curriculum_memory import CurriculumMemory

    memory = CurriculumMemory(tmp_path, max_rows=60)
    for cycle in range(1, 25):
        memory.expand(cycle, signals=["coding_gap", "data_gap", "tool_gap", "reasoning_gap"])
    rows = memory.rows()
    assert len(rows) <= 60
    fingerprints = {
        json.dumps({"domain": row.domain, "messages": row.messages}, sort_keys=True)
        for row in rows
    }
    assert len(fingerprints) == len(rows)
    assert memory.manifest()["validation_overlap"] == []


def test_training_curriculum_includes_persistent_replay_rows():
    from generalist_lm.curriculum import ResearchRow
    from generalist_lm.research_cycle import _training_rows_for_genome, research_seed

    replay = [
        ResearchRow(
            "language",
            [
                {"role": "user", "content": "Reply exactly PERSISTENT_REPLAY"},
                {"role": "assistant", "content": "PERSISTENT_REPLAY"},
            ],
        )
    ]
    rows = _training_rows_for_genome(research_seed(), replay)
    assert any(
        row.messages[0]["content"] == "Reply exactly PERSISTENT_REPLAY"
        for row in rows
    )


def test_continual_candidate_preserves_architecture_and_advances_lineage():
    from generalist_lm.research_cycle import _continual_candidate_genome, research_seed

    champion = research_seed()
    candidate = _continual_candidate_genome(champion, 7)
    assert candidate.generation == champion.generation + 1
    assert candidate.parent_id == champion.genome_id
    assert "continual" in candidate.genome_id
    for field in (
        "context_length", "d_model", "n_heads", "n_layers", "d_ff",
        "norm_type", "position_encoding", "ff_variant",
    ):
        assert getattr(candidate, field) == getattr(champion, field)


def test_research_cycle_records_persistent_curriculum_and_continual_trial(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    first = run_research_cycle(tmp_path)
    assert first["curriculum_memory"]["stored"] >= 6
    assert first["curriculum_memory"]["validation_overlap"] == []
    assert any(row.get("kind") == "continual_learning" for row in first["trials"])
    stored_first = first["curriculum_memory"]["stored"]

    second = run_research_cycle(tmp_path)
    assert second["curriculum_memory"]["stored"] > stored_first
    assert second["curriculum_memory"]["validation_overlap"] == []
    assert any(row.get("kind") == "continual_learning" for row in second["trials"])
    assert second["policy"]["continual_learning"]["full_replay"] is True


def test_research_promotion_rejects_autoregressive_generation_regression():
    from generalist_lm.research_cycle import _research_eligible

    champion = {
        "loss": 1.0,
        "domain_loss": {"language": 1.0},
        "finite": True,
        "target_token_accuracy": 0.40,
        "domain_token_accuracy": {"language": 0.40},
        "solved_items": [],
        "generation_exact_accuracy": 1.0,
        "domain_generation_accuracy": {"language": 1.0},
        "generated_solved_items": ["language:kept"],
    }
    candidate = {
        "loss": 0.7,
        "domain_loss": {"language": 0.7},
        "finite": True,
        "target_token_accuracy": 0.60,
        "domain_token_accuracy": {"language": 0.60},
        "solved_items": [],
        "generation_exact_accuracy": 0.0,
        "domain_generation_accuracy": {"language": 0.0},
        "generated_solved_items": [],
    }
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.02,
        max_domain_regression=0.10,
    )
    assert ok is False
    assert "autoregressive generation" in reason or "generated held-out" in reason


def test_research_promotion_retains_generated_solutions_when_loss_improves():
    from generalist_lm.research_cycle import _research_eligible

    champion = {
        "loss": 1.0,
        "domain_loss": {"language": 1.0},
        "finite": True,
        "target_token_accuracy": 0.40,
        "domain_token_accuracy": {"language": 0.40},
        "solved_items": ["language:teacher"],
        "generation_exact_accuracy": 1.0,
        "domain_generation_accuracy": {"language": 1.0},
        "generated_solved_items": ["language:generated"],
    }
    candidate = {
        "loss": 0.6,
        "domain_loss": {"language": 0.6},
        "finite": True,
        "target_token_accuracy": 0.60,
        "domain_token_accuracy": {"language": 0.60},
        "solved_items": ["language:teacher"],
        "generation_exact_accuracy": 1.0,
        "domain_generation_accuracy": {"language": 1.0},
        "generated_solved_items": ["language:generated"],
    }
    ok, reason = _research_eligible(
        champion,
        candidate,
        minimum_loss_gain=0.02,
        max_domain_regression=0.10,
    )
    assert ok is True
    assert "generation" in reason


def test_research_cycle_reports_real_generation_probe(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.curriculum import DOMAINS
    from generalist_lm.research_cycle import run_research_cycle

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    result = run_research_cycle(tmp_path)
    report = result["champion_report"]
    assert 0.0 <= report["generation_exact_accuracy"] <= 1.0
    assert set(report["domain_generation_accuracy"]) == set(DOMAINS)
    assert len(report["generation_probe"]) == len(DOMAINS)
    assert isinstance(report["generated_solved_items"], list)


def test_rotating_canary_is_multi_domain_and_excluded_from_persistent_training(tmp_path: Path):
    from generalist_lm.curriculum import DOMAINS
    from generalist_lm.curriculum_memory import CurriculumMemory, canary_rows

    memory = CurriculumMemory(tmp_path)
    memory.expand(9, signals=["coding_gap", "data_gap", "tool_gap", "reasoning_gap"])
    training_prompts = {row.messages[0]["content"] for row in memory.rows()}
    canary = canary_rows(9)
    canary_prompts = {row.messages[0]["content"] for row in canary}

    assert {row.domain for row in canary} == set(DOMAINS)
    assert len(canary) == len(DOMAINS)
    assert canary_prompts.isdisjoint(training_prompts)


def test_research_promotion_rejects_rotating_canary_regression():
    from generalist_lm.research_cycle import _research_eligible

    base_report = {
        "loss": 1.0,
        "domain_loss": {"language": 1.0},
        "finite": True,
        "target_token_accuracy": 0.5,
        "domain_token_accuracy": {"language": 0.5},
        "solved_items": [],
        "generation_exact_accuracy": 0.0,
        "domain_generation_accuracy": {"language": 0.0},
        "generated_solved_items": [],
        "canary": {
            "loss": 1.0,
            "domain_loss": {"language": 1.0},
            "finite": True,
            "generation_exact_accuracy": 0.0,
            "domain_generation_accuracy": {"language": 0.0},
            "generated_solved_items": [],
        },
    }
    candidate = {
        **base_report,
        "loss": 0.7,
        "domain_loss": {"language": 0.7},
        "target_token_accuracy": 0.6,
        "domain_token_accuracy": {"language": 0.6},
        "canary": {
            "loss": 1.5,
            "domain_loss": {"language": 1.5},
            "finite": True,
            "generation_exact_accuracy": 0.0,
            "domain_generation_accuracy": {"language": 0.0},
            "generated_solved_items": [],
        },
    }
    ok, reason = _research_eligible(
        base_report,
        candidate,
        minimum_loss_gain=0.02,
        max_domain_regression=0.10,
    )
    assert ok is False
    assert "rotating canary" in reason


def test_research_cycle_persists_rotating_canary_contract(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.curriculum import DOMAINS
    from generalist_lm.research_cycle import run_research_cycle
    from generalist_lm.research_health import research_health

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    result = run_research_cycle(tmp_path)
    assert result["rotating_canary"]["cycle"] == 1
    assert set(result["rotating_canary"]["domains"]) == set(DOMAINS)
    assert result["rotating_canary"]["training_overlap"] == []
    assert result["policy"]["rotating_canary"]["training_excluded"] is True
    assert result["champion_report"]["canary_cycle"] == 1
    assert set(result["champion_report"]["canary"]["domain_loss"]) == set(DOMAINS)
    assert research_health(tmp_path)["ok"] is True


def test_rope_position_encoding_runs_real_forward_and_generation():
    torch = pytest.importorskip("torch")
    from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig

    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        position_encoding="rope",
    ).validate()
    model = CausalTransformerLM(cfg)
    ids = torch.randint(8, cfg.vocab_size, (2, 12))
    result = model(ids, labels=ids.clone())
    assert tuple(result["logits"].shape) == (2, 12, cfg.vocab_size)
    assert torch.isfinite(result["loss"])
    generated = model.generate(ids[:, :4], max_new_tokens=3, eos_token_id=None)
    assert tuple(generated.shape) == (2, 7)


def test_rope_rejects_odd_attention_head_dimension():
    from generalist_lm.model import GeneralistLMConfig
    with pytest.raises(ValueError, match="even attention head dimension"):
        GeneralistLMConfig(
            vocab_size=264,
            context_length=64,
            d_model=36,
            n_heads=4,
            n_layers=1,
            d_ff=72,
            position_encoding="rope",
        ).validate()


def test_evolution_can_propose_rope_without_forcing_it():
    from dataclasses import replace
    from generalist_lm.evolution import generate_challengers, GeneralistGenome

    champion = GeneralistGenome(
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        position_encoding="learned",
    ).validate()
    challengers = generate_challengers(champion, count=4, exploration_offset=0)
    assert any(row.position_encoding == "rope" for row in challengers)
    assert all(row.position_encoding in {"learned", "sinusoidal", "rope"} for row in challengers)

    rope = replace(champion, position_encoding="rope").validate()
    assert rope.position_encoding == "rope"


def test_compatible_weight_transfer_preserves_identical_model_state():
    torch = pytest.importorskip("torch")
    from generalist_lm.model import CausalTransformerLM
    from generalist_lm.research_cycle import _transfer_compatible_weights

    cfg = tiny_config()
    source = CausalTransformerLM(cfg)
    target = CausalTransformerLM(cfg)
    report = _transfer_compatible_weights(source, target)
    assert report["copied_tensors"] == report["target_tensors"]
    assert report["parameter_fraction"] == pytest.approx(1.0)

    ids = torch.randint(8, cfg.vocab_size, (1, 12))
    source.eval()
    target.eval()
    with torch.no_grad():
        source_logits = source(ids)["logits"]
        target_logits = target(ids)["logits"]
    assert torch.equal(source_logits, target_logits)


def test_architecture_weight_transfer_copies_only_shape_compatible_tensors():
    torch = pytest.importorskip("torch")
    from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
    from generalist_lm.research_cycle import _transfer_compatible_weights

    learned = GeneralistLMConfig(
        vocab_size=264, context_length=64, d_model=32, n_heads=4,
        n_layers=1, d_ff=64, dropout=0.0, position_encoding="learned",
    ).validate()
    rope = GeneralistLMConfig(
        vocab_size=264, context_length=64, d_model=32, n_heads=4,
        n_layers=1, d_ff=64, dropout=0.0, position_encoding="rope",
    ).validate()
    source = CausalTransformerLM(learned)
    target = CausalTransformerLM(rope)
    report = _transfer_compatible_weights(source, target)

    assert report["parameter_fraction"] == pytest.approx(1.0)
    assert report["source_tensors"] > report["target_tensors"]
    assert "position_embedding.weight" in report["source_unmatched_tensors"]
    assert torch.equal(
        source.token_embedding.weight.detach(),
        target.token_embedding.weight.detach(),
    )


def test_research_cycle_architecture_trial_records_weight_transfer(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    result = run_research_cycle(tmp_path)
    architecture_trials = [
        row for row in result["trials"]
        if row.get("kind") == "architecture" and row.get("report")
    ]
    assert architecture_trials
    trial = architecture_trials[0]
    transfer = trial["report"]["weight_transfer"]
    assert transfer["copied_tensors"] > 0
    assert 0.0 < transfer["parameter_fraction"] <= 1.0
    if trial["genome"]["tokenizer_version"] == result["champion"]["tokenizer_version"]:
        assert transfer["policy"] == "exact tensors plus safe prefix inheritance for expansions"
        assert transfer["tokenizer_identical"] is True
        assert isinstance(transfer["partial_prefix_tensors"], list)
    else:
        assert transfer["policy"] == "exact/prefix-compatible tensors plus deterministic byte-compatible vocabulary migration"
        assert transfer["vocabulary_migrated"] is True
        assert transfer["shared_token_rows"] > 0


def test_curriculum_memory_fails_closed_on_digest_corruption(tmp_path: Path):
    from generalist_lm.curriculum_memory import CurriculumMemory

    memory = CurriculumMemory(tmp_path)
    memory.expand(1, signals=[])
    raw = json.loads((tmp_path / "curriculum-memory.json").read_text(encoding="utf-8"))
    raw["rows"][0]["id"] = "tampered"
    (tmp_path / "curriculum-memory.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        memory.rows()


def test_curriculum_memory_fails_closed_on_invalid_domain_and_validation_overlap(tmp_path: Path):
    from generalist_lm.curriculum import validation_rows
    from generalist_lm.curriculum_memory import CurriculumMemory, _row_id
    from generalist_lm.curriculum import ResearchRow

    memory = CurriculumMemory(tmp_path)
    memory.expand(1, signals=[])
    path = tmp_path / "curriculum-memory.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"][0]["domain"] = "unbounded_web_truth"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported curriculum domain|digest mismatch"):
        memory.rows()

    protected = validation_rows()[0]
    malicious = ResearchRow(protected.domain, protected.messages)
    path.write_text(json.dumps({
        "version": 1,
        "last_cycle": 1,
        "rows": [{
            "id": _row_id(malicious),
            "domain": malicious.domain,
            "messages": malicious.messages,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="protected validation"):
        memory.rows()


def test_research_health_rejects_corrupted_persistent_curriculum(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle
    from generalist_lm.research_health import research_health

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    run_research_cycle(tmp_path)
    path = tmp_path / "curriculum-memory.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"][0]["id"] = "corrupted"
    path.write_text(json.dumps(raw), encoding="utf-8")

    report = research_health(tmp_path)
    assert report["ok"] is False
    assert any(row["name"] == "curriculum_memory:loadable" for row in report["failed"])


def test_production_qualification_is_disjoint_from_all_research_curricula():
    from generalist_lm.benchmarks import qualification_manifest, qualification_suite
    from generalist_lm.curriculum import DOMAINS, train_rows, validation_rows
    from generalist_lm.curriculum_memory import _mechanical_rows, canary_rows

    protected = qualification_suite()
    protected_prompts = {task.prompt for task in protected}
    assert protected_prompts
    assert set(qualification_manifest()["critical_domains"]) == set(DOMAINS)

    fixed_research_prompts = {
        row.messages[0]["content"]
        for row in [*train_rows(), *validation_rows()]
    }
    assert protected_prompts.isdisjoint(fixed_research_prompts)

    # Sample a long horizon of the deterministic autonomous replay/canary
    # generator. Production qualification must never become training replay.
    generated_prompts = set()
    for cycle in range(1, 257):
        for row in _mechanical_rows(
            cycle,
            ["coding_gap", "data_gap", "reasoning_gap", "tool_gap", "symbolic_reasoning_signal"],
        ):
            generated_prompts.add(row.messages[0]["content"])
        for row in canary_rows(cycle):
            generated_prompts.add(row.messages[0]["content"])
    assert protected_prompts.isdisjoint(generated_prompts)


def test_qualification_suite_has_critical_gate_for_every_generalist_domain():
    from generalist_lm.benchmarks import qualification_suite
    from generalist_lm.curriculum import DOMAINS

    tasks = qualification_suite()
    for domain in DOMAINS:
        domain_tasks = [task for task in tasks if task.domain == domain]
        assert domain_tasks, domain
        assert any(task.critical for task in domain_tasks), domain


def test_legacy_v1_qualification_attestation_is_rejected(tmp_path: Path):
    import json
    pytest.importorskip("torch")
    from generalist_lm.qualification import checkpoint_digest, qualification_status
    from generalist_lm.runtime import GeneralistRuntime

    runtime = GeneralistRuntime.fresh(tiny_config())
    runtime.save_checkpoint(tmp_path)
    digest = checkpoint_digest(tmp_path)
    (tmp_path / "benchmark.json").write_text(json.dumps({
        "qualification_version": 1,
        "attested_by": "airi-generalist-qualification-v1",
        "checkpoint_digest": digest,
        "qualified": True,
        "minimum_score": 0.0,
        "report": {"ok": True, "score": 100.0, "critical_failures": []},
    }), encoding="utf-8")

    status = qualification_status(tmp_path)
    assert status["qualified"] is False
    assert status["integrity_ok"] is False


def test_untrained_causal_checkpoint_fails_real_production_qualification(tmp_path: Path):
    pytest.importorskip("torch")
    from generalist_lm.qualification import qualify_checkpoint
    from generalist_lm.runtime import GeneralistRuntime

    runtime = GeneralistRuntime.fresh(tiny_config())
    runtime.save_checkpoint(tmp_path, metadata={"purpose": "untrained-negative-control"})
    report = qualify_checkpoint(tmp_path, minimum_score=85.0)

    assert report["qualified"] is False
    assert report["qualification_version"] == 2
    assert report["report"]["score"] < 85.0 or report["report"]["critical_failures"]


def test_generalist_research_health_rejects_oversized_persisted_checkpoint(tmp_path: Path, monkeypatch):
    import json
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import research_seed
    from generalist_lm.research_health import research_health
    from generalist_lm.runtime import GeneralistRuntime

    genome = research_seed()
    runtime = GeneralistRuntime.fresh(genome.model_config())
    champion = tmp_path / "champion"
    runtime.save_checkpoint(champion, metadata={"role": "research_champion"})
    (tmp_path / "champion-genome.json").write_text(
        json.dumps(genome.to_dict()),
        encoding="utf-8",
    )

    # A tiny limit must fail closed before such a checkpoint could be persisted.
    monkeypatch.setenv("AIRI_GENERALIST_MAX_PERSISTED_CHECKPOINT_BYTES", "65536")
    report = research_health(tmp_path)
    failed = {row["name"] for row in report["failed"]}
    assert "checkpoint:persistence_size" in failed


def _git(cwd: Path, *args: str):
    import subprocess
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        text=True,
        capture_output=True,
        check=True,
    )


def _seed_generalist_state_repo(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Test")
    _git(source, "config", "user.email", "test@example.com")
    production = source / "generalist-state" / "production"
    production.mkdir(parents=True)
    (production / "config.json").write_text('{"demo":true}', encoding="utf-8")
    (production / "model.pt").write_bytes(b"new-model-bytes")
    (production / "metadata.json").write_text('{"role":"production_champion"}', encoding="utf-8")
    (production / "benchmark.json").write_text('{"qualified":false}', encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "seed generalist production")
    _git(source, "branch", "-M", "generalist-state")
    return source


def _fake_live_qualify(path: Path, *, minimum_score: float = 85.0):
    from generalist_lm.qualification import QUALIFICATION_VERSION, checkpoint_digest

    root = Path(path)
    digest = checkpoint_digest(root)
    result = {
        "qualification_version": QUALIFICATION_VERSION,
        "attested_by": "airi-generalist-qualification-v2",
        "checkpoint_digest": digest,
        "qualified": True,
        "minimum_score": float(minimum_score),
        "report": {"ok": True, "score": 100.0, "critical_failures": []},
    }
    (root / "benchmark.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def test_generalist_state_sync_transactionally_installs_locally_requalified_model(tmp_path: Path, monkeypatch):
    from generalist_lm import state_sync
    from generalist_lm.qualification import qualification_status

    source = _seed_generalist_state_repo(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _git(runtime, "init", "-q")
    _git(runtime, "config", "user.name", "Test")
    _git(runtime, "config", "user.email", "test@example.com")
    (runtime / "README.md").write_text("runtime", encoding="utf-8")
    _git(runtime, "add", ".")
    _git(runtime, "commit", "-qm", "runtime")
    _git(runtime, "remote", "add", "origin", str(source))

    target = runtime / ".ai" / "generalist-lm" / "champion"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(state_sync, "qualify_checkpoint", _fake_live_qualify)
    result = state_sync.sync_production_checkpoint(runtime, target)
    assert result["ok"] is True
    assert result["updated"] is True
    assert (target / "model.pt").read_bytes() == b"new-model-bytes"
    assert not (target / "old.txt").exists()
    assert qualification_status(target)["qualified"] is True


def test_generalist_state_sync_keeps_previous_model_when_requalification_fails(tmp_path: Path, monkeypatch):
    from generalist_lm import state_sync

    source = _seed_generalist_state_repo(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _git(runtime, "init", "-q")
    _git(runtime, "config", "user.name", "Test")
    _git(runtime, "config", "user.email", "test@example.com")
    (runtime / "README.md").write_text("runtime", encoding="utf-8")
    _git(runtime, "add", ".")
    _git(runtime, "commit", "-qm", "runtime")
    _git(runtime, "remote", "add", "origin", str(source))

    target = runtime / ".ai" / "generalist-lm" / "champion"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("keep-me", encoding="utf-8")

    monkeypatch.setattr(
        state_sync,
        "qualify_checkpoint",
        lambda path, minimum_score=85.0: {"qualified": False},
    )
    with pytest.raises(RuntimeError, match="failed local requalification"):
        state_sync.sync_production_checkpoint(runtime, target)
    assert (target / "old.txt").read_text(encoding="utf-8") == "keep-me"
    assert not (target / "model.pt").exists()


def test_startup_generalist_sync_is_explicit_opt_in():
    start = (ROOT / "computer" / "start.sh").read_text(encoding="utf-8")
    assert "AIRI_GENERALIST_SYNC_FROM_GITHUB" in start
    assert "generalist_lm.state_sync" in start
    assert "AIRI_GENERALIST_SYNC_FAILED_KEEPING_LOCAL_CHECKPOINT" in start


@pytest.mark.parametrize("position_encoding", ["learned", "sinusoidal", "rope"])
def test_kv_cache_greedy_generation_matches_full_recompute(position_encoding):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        position_encoding=position_encoding,
    ).validate()
    model = CausalTransformerLM(cfg)
    model.eval()
    ids = torch.randint(8, cfg.vocab_size, (2, 9))

    uncached = model.generate(
        ids,
        max_new_tokens=7,
        eos_token_id=None,
        temperature=0.0,
        use_cache=False,
    )
    cached = model.generate(
        ids,
        max_new_tokens=7,
        eos_token_id=None,
        temperature=0.0,
        use_cache=True,
    )
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("position_encoding", ["learned", "sinusoidal", "rope"])
def test_incremental_kv_logits_match_full_forward(position_encoding):
    torch = pytest.importorskip("torch")
    torch.manual_seed(77)
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        position_encoding=position_encoding,
    ).validate()
    model = CausalTransformerLM(cfg)
    model.eval()
    ids = torch.randint(8, cfg.vocab_size, (1, 11))

    with torch.no_grad():
        full = model(ids)["logits"][:, -1, :]
        prefix = model(ids[:, :-1], use_cache=True)
        incremental = model(
            ids[:, -1:],
            past_key_values=prefix["past_key_values"],
            use_cache=True,
        )["logits"][:, -1, :]

    assert torch.allclose(full, incremental, atol=1e-5, rtol=1e-5)


def test_kv_cache_falls_back_safely_at_context_boundary():
    torch = pytest.importorskip("torch")
    torch.manual_seed(9)
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        position_encoding="learned",
    ).validate()
    model = CausalTransformerLM(cfg)
    model.eval()
    ids = torch.randint(8, cfg.vocab_size, (1, 15))
    cached = model.generate(
        ids,
        max_new_tokens=5,
        eos_token_id=None,
        temperature=0.0,
        use_cache=True,
    )
    uncached = model.generate(
        ids,
        max_new_tokens=5,
        eos_token_id=None,
        temperature=0.0,
        use_cache=False,
    )
    assert torch.equal(cached, uncached)


def test_training_gradient_accumulation_reports_effective_batch_and_learns():
    pytest.importorskip("torch")
    cfg = tiny_config()
    model = CausalTransformerLM(cfg)
    tok = ByteTokenizer()
    examples = [
        SFTExample([
            {"role": "user", "content": "Say A"},
            {"role": "assistant", "content": "A"},
        ]),
        SFTExample([
            {"role": "user", "content": "Say B"},
            {"role": "assistant", "content": "B"},
        ]),
    ]
    report = train_sft(
        model,
        tok,
        examples,
        steps=20,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=8e-3,
        weight_decay=0.0,
        seed=5,
        precision="fp32",
    )
    assert report["ok"] is True
    assert report["effective_batch_size"] == 2
    assert report["gradient_accumulation_steps"] == 2
    assert report["final_loss"] < report["initial_loss"]


def test_training_rejects_fp16_on_cpu():
    pytest.importorskip("torch")
    model = CausalTransformerLM(tiny_config())
    with pytest.raises(ValueError, match="fp16 training requires CUDA"):
        train_sft(
            model,
            ByteTokenizer(),
            [SFTExample([
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
            ])],
            steps=1,
            precision="fp16",
            device="cpu",
        )


def test_research_cycle_uses_configured_gradient_accumulation(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.research_cycle import run_research_cycle

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_BOOTSTRAP_STEPS", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_CHALLENGERS", "1")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_GRADIENT_ACCUMULATION", "2")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_PRECISION", "fp32")
    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_MIN_LOSS_GAIN", "999")

    result = run_research_cycle(tmp_path)
    training = result["champion_report"]["training"]
    assert training["gradient_accumulation_steps"] == 2
    assert training["effective_batch_size"] == 8
    assert training["precision"] == "fp32"


def test_research_cycle_rejects_unknown_precision_before_training(tmp_path: Path, monkeypatch):
    from generalist_lm.research_cycle import run_research_cycle

    monkeypatch.setenv("AIRI_GENERALIST_RESEARCH_PRECISION", "int4-magic")
    with pytest.raises(ValueError, match="must be fp32, bf16, or fp16"):
        run_research_cycle(tmp_path)
