from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest

from generalist_lm.bootstrap_data import (
    BootstrapDataBundle,
    OASST1_REVISION,
    SOURCES,
    STREAMING_SOURCES,
    _oasst_conversations,
    _parse_oasst,
    _quality_web_text,
    _read_streaming_cache,
    _web_chunks,
    load_bootstrap_replay,
    write_bootstrap_replay,
)
from generalist_lm.bootstrap_training import (
    _anti_collapse_rescue_gate,
    _anti_collapse_weights,
    _bootstrap_capacity_target,
    _bootstrap_corpus_target,
    _effective_bootstrap_target,
    _filter_protected_replay,
    _grow_bootstrap_runtime,
    _phase5_success,
)
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.pretraining import (
    CorpusDocument,
    load_packed_block_cache,
    save_packed_block_cache,
)
from generalist_lm.phase5_diagnostics import (
    PHASE5_PROBES,
    degeneration_gate,
    evaluate_phase5_language,
    evaluate_sft_validation,
    protected_bootstrap_texts,
    sft_validation_gate,
)
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import ByteTokenizer
from generalist_lm.training import SFTExample, causal_training_objective




def test_phase5_fasttrack_handoff_preserves_live_app_lineage():
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(encoding="utf-8")
    assert "Dispatch next in-place language rung" in workflow
    assert "same persisted AIRI Phase-5 lineage" in workflow
    assert "next=100000000" in workflow
    assert '"handoff":"converged_swarm_first"' not in workflow
    assert "Generalist swarm must run once before it is dispatched" not in workflow

def test_phase5_packed_block_cache_roundtrips_exact_tokens(tmp_path):
    path = tmp_path / "blocks.bin"
    blocks = [
        [1, 8, 9, 10, 2, 0, 0, 0],
        [1, 11, 12, 13, 14, 2, 0, 0],
    ]
    identity = {
        "version": "fixture-v1",
        "manifest_content_sha256": "abc123",
        "tokenizer_version": "bpe-v1",
        "tokenizer_vocab_size": 384,
        "context_length": 8,
        "split": "C_causal_next_sentence",
    }
    meta = save_packed_block_cache(
        path,
        blocks,
        block_size=8,
        identity=identity,
    )
    loaded = load_packed_block_cache(path, expected_identity=identity)

    assert meta["block_count"] == 2
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0] == blocks[0]
    assert loaded[1] == blocks[1]


def test_phase5_packed_block_cache_rejects_stale_or_corrupt_data(tmp_path):
    path = tmp_path / "blocks.bin"
    blocks = [[1, 8, 9, 10, 2, 0, 0, 0]]
    identity = {
        "version": "fixture-v1",
        "manifest_content_sha256": "abc123",
        "tokenizer_version": "bpe-v1",
        "tokenizer_vocab_size": 384,
        "context_length": 8,
        "split": "validation",
    }
    save_packed_block_cache(
        path,
        blocks,
        block_size=8,
        identity=identity,
    )

    stale = dict(identity)
    stale["manifest_content_sha256"] = "different"
    assert load_packed_block_cache(path, expected_identity=stale) is None

    path.write_bytes(path.read_bytes()[:-1])
    assert load_packed_block_cache(path, expected_identity=identity) is None


def test_phase5_cumulative_target_never_shrinks_on_maintenance_run():
    assert _effective_bootstrap_target(1_000_000, {"target_tokens": 5_000_000}) == 5_000_000
    assert _effective_bootstrap_target(20_000_000, {"target_tokens": 5_000_000}) == 20_000_000
    assert _effective_bootstrap_target(1_000_000, {}) == 1_000_000


def test_phase5_conversation_rescue_separates_unique_corpus_from_training_budget():
    assert _bootstrap_corpus_target(1_000_000) == 1_000_000
    assert _bootstrap_corpus_target(5_000_000) == 5_000_000
    assert _bootstrap_corpus_target(20_000_000) == 5_000_000
    assert _bootstrap_corpus_target(50_000_000) == 5_000_000
    assert _bootstrap_corpus_target(100_000_000) == 20_000_000
    assert _bootstrap_corpus_target(500_000_000) == 20_000_000


def test_phase5_conversation_rescue_has_explicit_capacity_rungs():
    assert _bootstrap_capacity_target(5_000_000) is None
    assert _bootstrap_capacity_target(19_999_999) is None
    assert _bootstrap_capacity_target(20_000_000) == 1_250_000
    assert _bootstrap_capacity_target(50_000_000) == 3_000_000
    assert _bootstrap_capacity_target(100_000_000) == 7_000_000
    assert _bootstrap_capacity_target(250_000_000) == 12_000_000
    assert _bootstrap_capacity_target(500_000_000) == 20_000_000


def test_phase5_capacity_growth_builds_a_larger_compatible_runtime():
    pytest.importorskip("torch")
    from generalist_lm.model import parameter_count
    from generalist_lm.research_cycle import research_seed

    genome = research_seed()
    tokenizer = ByteTokenizer()
    config = genome.model_config(tokenizer.vocab_size)
    runtime = GeneralistRuntime(
        CausalTransformerLM(config),
        config,
        tokenizer=tokenizer,
        device="cpu",
    )
    before = parameter_count(runtime.model)

    grown_genome, grown, report = _grow_bootstrap_runtime(
        genome,
        runtime,
        target_parameters=max(250_000, before + 1),
    )

    assert parameter_count(grown.model) > before
    assert report["source_parameters"] == before
    assert report["parameters"] == parameter_count(grown.model)
    assert report["weight_transfer"]["copied_parameters"] > 0
    assert report["weight_transfer"]["function_preserving_growth"] is True
    assert grown.config.d_model == runtime.config.d_model
    assert grown.config.to_dict() == grown_genome.model_config(
        grown.tokenizer.vocab_size
    ).to_dict()

    # Capacity growth itself must not erase the function already learned.
    torch = __import__("torch")
    ids = torch.tensor(
        [[1, 40, 41, 42, 43, 44, 45, 46]],
        dtype=torch.long,
    )
    runtime.model.eval()
    grown.model.eval()
    with torch.no_grad():
        before_logits = runtime.model(ids)["logits"]
        grown_logits = grown.model(ids)["logits"]
    assert torch.allclose(before_logits, grown_logits, atol=1e-6, rtol=1e-6)


def test_phase5_success_requires_multiword_output():
    before = {
        "language_nll": 4.0,
        "repetition_rate": 0.50,
    }
    almost = {
        "language_nll": 3.5,
        "pathological_repetition": False,
        "repetition_rate": 0.40,
        "non_empty_rate": 1.0,
        "word_output_rate": 1.0,
        "multiword_output_rate": 0.20,
    }
    ok, reasons = _phase5_success(before, almost)
    assert not ok
    assert any("multi-word" in reason for reason in reasons)

    conversational = dict(almost)
    conversational["multiword_output_rate"] = 0.60
    ok, reasons = _phase5_success(before, conversational)
    assert ok, reasons

def test_phase5_holdout_suite_is_explicit_and_protected():
    assert len(PHASE5_PROBES) == 7
    protected = protected_bootstrap_texts()
    assert "ciao" in protected
    assert "hello" in protected
    assert "write one simple sentence." in protected


def test_bootstrap_sources_are_explicitly_licensed_and_pinned():
    by_id = {row["id"]: row for row in SOURCES}
    assert by_id["tatoeba-en-cc0"]["license"] == "CC0-1.0"
    assert by_id["tatoeba-it-ccby"]["license"] == "CC-BY-2.0-FR"
    assert by_id["oasst1-human"]["license"] == "Apache-2.0"
    assert by_id["oasst1-human"]["revision"] == OASST1_REVISION
    assert len(OASST1_REVISION) == 40
    assert all(row["url"].startswith("https://") for row in SOURCES)
    assert all(row["license_url"].startswith("https://") for row in SOURCES)


def test_fasttrack_streaming_sources_are_explicit_and_bilingual():
    by_id = {row["id"]: row for row in STREAMING_SOURCES}
    assert by_id["fineweb2-it"]["dataset"] == "HuggingFaceFW/fineweb-2"
    assert by_id["fineweb2-it"]["config"] == "ita_Latn"
    assert by_id["fineweb-en"]["dataset"] == "HuggingFaceFW/fineweb"
    assert by_id["fineweb-en"]["config"] == "sample-10BT"
    assert {row["language"] for row in STREAMING_SOURCES} == {"it", "en"}
    assert all(row["license"] == "ODC-By-1.0" for row in STREAMING_SOURCES)
    assert all(row["source_page"].startswith("https://") for row in STREAMING_SOURCES)


def test_fasttrack_web_chunking_is_bounded_and_normalizes_whitespace():
    raw = (
        "Questa è una frase italiana abbastanza lunga da essere utile al modello e contiene parole naturali per un buon esempio di addestramento.\n\n"
        "Seconda frase con   spazi multipli e altro testo naturale sufficientemente lungo per verificare la pulizia dei documenti web."
    )
    chunks = _web_chunks(raw, max_chars=180)
    assert len(chunks) >= 2
    assert all(len(row) <= 180 for row in chunks)
    assert all("   " not in row for row in chunks)
    assert all(_quality_web_text(row) for row in chunks)


def test_fasttrack_stream_cache_roundtrips_with_digest_validation(tmp_path):
    tokenizer = ByteTokenizer()
    source = {"id": "fixture", "domain": "general"}
    text = "Una frase naturale abbastanza lunga per verificare la cache del corpus AIRI."
    digest = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    cache = tmp_path / "fixture.jsonl.gz"
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "source": "fixture:1:0",
            "text": text,
            "sha256": digest,
        }, sort_keys=True) + "\n")
    quota = len(tokenizer.encode(text)) + 1
    documents, tokens = _read_streaming_cache(
        source, tokenizer, quota=quota, cache_path=cache
    )
    assert tokens == quota
    assert len(documents) == 1
    assert documents[0].text == text


def test_oasst_parser_excludes_synthetic_and_builds_human_dialogue():
    rows = [
        {
            "message_id": "u1",
            "parent_id": None,
            "text": "Tell me something simple.",
            "role": "prompter",
            "lang": "en",
            "deleted": False,
            "synthetic": False,
            "review_result": True,
        },
        {
            "message_id": "a1",
            "parent_id": "u1",
            "text": "A cat sleeps.",
            "role": "assistant",
            "lang": "en",
            "deleted": False,
            "synthetic": False,
            "review_result": True,
        },
        {
            "message_id": "a2",
            "parent_id": "u1",
            "text": "Synthetic answer.",
            "role": "assistant",
            "lang": "en",
            "deleted": False,
            "synthetic": True,
            "review_result": True,
        },
    ]
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb") as handle:
        for row in rows:
            handle.write((json.dumps(row) + "\n").encode("utf-8"))
    parsed = _parse_oasst(raw.getvalue())
    assert "a1" in parsed
    assert "a2" not in parsed
    conversations = _oasst_conversations(parsed)
    assert len(conversations) == 1
    assert conversations[0][1][-1]["content"] == "A cat sleeps."


def test_bootstrap_replay_is_bounded_training_only_and_roundtrips(tmp_path):
    tokenizer = ByteTokenizer()
    documents = [
        CorpusDocument(
            source="tatoeba-en-cc0:1:en",
            text="A calm cat sleeps beside the warm window in the morning.",
            sha256=__import__("hashlib").sha256(
                b"A calm cat sleeps beside the warm window in the morning."
            ).hexdigest(),
            bytes=len(b"A calm cat sleeps beside the warm window in the morning."),
            domain="language",
        ),
        CorpusDocument(
            source="tatoeba-it-ccby:2:it",
            text="Un gatto tranquillo dorme accanto alla finestra durante la mattina.",
            sha256=__import__("hashlib").sha256(
                "Un gatto tranquillo dorme accanto alla finestra durante la mattina.".encode()
            ).hexdigest(),
            bytes=len(
                "Un gatto tranquillo dorme accanto alla finestra durante la mattina.".encode()
            ),
            domain="language",
        ),
        CorpusDocument(
            source="oasst1-human:3:en",
            text="I can explain that idea with a short and clear example.",
            sha256=__import__("hashlib").sha256(
                b"I can explain that idea with a short and clear example."
            ).hexdigest(),
            bytes=len(b"I can explain that idea with a short and clear example."),
            domain="dialogue",
        ),
    ]
    heldout = CorpusDocument(
        source="heldout:never",
        text="THIS MUST NOT ENTER TRAINING REPLAY.",
        sha256=__import__("hashlib").sha256(
            b"THIS MUST NOT ENTER TRAINING REPLAY."
        ).hexdigest(),
        bytes=len(b"THIS MUST NOT ENTER TRAINING REPLAY."),
        domain="language",
    )
    bundle = BootstrapDataBundle(
        train_documents=documents,
        validation_documents=[heldout],
        sft_train=[SFTExample([
            {"role": "user", "content": "Say hello naturally."},
            {"role": "assistant", "content": "Hello! Nice to meet you."},
        ])],
        sft_validation=[SFTExample([
            {"role": "user", "content": "Held out question"},
            {"role": "assistant", "content": "Held out answer"},
        ])],
        manifest={"manifest_content_sha256": "abc123"},
    )

    first = write_bootstrap_replay(
        bundle,
        tokenizer,
        output_dir=tmp_path,
        max_tokens=100_000,
        max_sft_conversations=8,
    )
    loaded = load_bootstrap_replay(tmp_path)

    assert loaded.manifest["available"] is True
    assert first["held_out_phase5_suite_excluded"] is True
    assert first["validation_documents_excluded"] is True
    assert first["sft_validation_excluded"] is True
    assert {row.source for row in loaded.documents} == {
        "tatoeba-en-cc0:1:en",
        "tatoeba-it-ccby:2:it",
        "oasst1-human:3:en",
    }
    assert all("THIS MUST NOT ENTER" not in row.text for row in loaded.documents)
    assert len(loaded.sft_train) == 1
    assert loaded.sft_train[0].messages[-1]["content"] == "Hello! Nice to meet you."

    # gzip output is deterministic (mtime=0), making the replay immutable/hashable.
    second = write_bootstrap_replay(
        bundle,
        tokenizer,
        output_dir=tmp_path,
        max_tokens=100_000,
        max_sft_conversations=8,
    )
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["sft_sha256"] == second["sft_sha256"]


def test_bootstrap_replay_fails_closed_on_digest_tampering(tmp_path):
    tokenizer = ByteTokenizer()
    text = "A sufficiently natural training sentence for replay integrity checks."
    bundle = BootstrapDataBundle(
        train_documents=[CorpusDocument(
            source="tatoeba-en-cc0:1:en",
            text=text,
            sha256=__import__("hashlib").sha256(text.encode()).hexdigest(),
            bytes=len(text.encode()),
            domain="language",
        )],
        validation_documents=[],
        sft_train=[],
        sft_validation=[],
        manifest={"manifest_content_sha256": "abc123"},
    )
    write_bootstrap_replay(bundle, tokenizer, output_dir=tmp_path)
    corpus = tmp_path / "replay-corpus.jsonl.gz"
    corpus.write_bytes(corpus.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_bootstrap_replay(tmp_path)


def test_anti_collapse_objective_never_penalizes_correct_repetition():
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[8, 8, 8, 8]], dtype=torch.long)
    labels = input_ids.clone()
    logits = torch.zeros((1, 4, 32), dtype=torch.float32, requires_grad=True)
    logits.data[:, :, 8] = 5.0

    total, stats = causal_training_objective(
        logits,
        labels,
        input_ids,
        repetition_unlikelihood_weight=0.2,
        eos_loss_weight=1.0,
        repetition_window=4,
    )
    assert torch.isfinite(total)
    assert stats["repetition_negative_count"] == 0
    assert stats["repetition_unlikelihood_loss"] == pytest.approx(0.0)


def test_anti_collapse_objective_penalizes_wrong_recent_token_mass():
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[8, 9, 10, 11]], dtype=torch.long)
    labels = input_ids.clone()
    logits = torch.zeros((1, 4, 32), dtype=torch.float32, requires_grad=True)
    # At each prediction position put excessive mass on the token just seen,
    # while the target is the following token.
    logits.data[0, 0, 8] = 6.0
    logits.data[0, 1, 9] = 6.0
    logits.data[0, 2, 10] = 6.0

    base, _ = causal_training_objective(
        logits,
        labels,
        input_ids,
        repetition_unlikelihood_weight=0.0,
    )
    guarded, stats = causal_training_objective(
        logits,
        labels,
        input_ids,
        repetition_unlikelihood_weight=0.1,
        repetition_window=4,
    )
    assert stats["repetition_negative_count"] > 0
    assert stats["repetition_unlikelihood_loss"] > 0.0
    assert float(guarded.detach()) > float(base.detach())
    guarded.backward()
    assert torch.isfinite(logits.grad).all()


def test_anti_collapse_schedule_only_activates_for_measured_collapse():
    clean = {"pathological_repetition": False, "repetition_rate": 0.20}
    assert _anti_collapse_weights("C_causal_next_sentence", clean) == (0.0, 1.0)

    collapsed = {"pathological_repetition": True, "repetition_rate": 0.80}
    a = _anti_collapse_weights("A_frequent_word_contexts", collapsed)
    b = _anti_collapse_weights("B_short_sentence_completion", collapsed)
    c_stage = _anti_collapse_weights("C_causal_next_sentence", collapsed)
    assert a[0] == 0.0
    assert 0.0 < b[0] < c_stage[0] <= 0.10
    assert 1.0 < a[1] < b[1] < c_stage[1] <= 2.0


def test_anti_collapse_rescue_gate_requires_real_repetition_improvement():
    before = {
        "pathological_repetition": True,
        "repetition_rate": 0.80,
        "language_nll": 3.0,
        "non_empty_rate": 1.0,
        "token_entropy": 3.0,
        "generation_similarity": 0.10,
        "longest_repeated_token_run": 12,
    }
    better = {
        **before,
        "repetition_rate": 0.70,
        "language_nll": 3.02,
        "token_entropy": 3.1,
        "generation_similarity": 0.12,
        "longest_repeated_token_run": 8,
    }
    ok, reasons = _anti_collapse_rescue_gate(before, better)
    assert ok, reasons

    fake_gain = dict(better)
    fake_gain["repetition_rate"] = 0.79
    ok, reasons = _anti_collapse_rescue_gate(before, fake_gain)
    assert not ok
    assert any("repetition" in reason for reason in reasons)

    nll_regression = dict(better)
    nll_regression["language_nll"] = 3.2
    ok, reasons = _anti_collapse_rescue_gate(before, nll_regression)
    assert not ok
    assert any("NLL" in reason for reason in reasons)


def test_degeneration_gate_rejects_repeated_token_collapse():
    before = {
        "repetition_rate": 0.20,
        "token_entropy": 3.0,
    }
    after = {
        "pathological_repetition": True,
        "repetition_rate": 0.90,
        "token_entropy": 0.2,
        "longest_repeated_token_run": 12,
    }
    ok, reason = degeneration_gate(before, after)
    assert not ok
    assert "repetition" in reason


def test_sft_gate_compares_against_pre_sft_baseline_instead_of_rejecting_existing_pathology():
    before = {
        "pathological_repetition": True,
        "repetition_rate": 0.70,
        "token_entropy": 3.0,
        "unique_token_ratio": 0.25,
        "longest_repeated_token_run": 9,
        "language_nll": 2.0,
        "non_empty_rate": 1.0,
    }
    stable = {
        "pathological_repetition": True,
        "repetition_rate": 0.71,
        "token_entropy": 2.9,
        "unique_token_ratio": 0.24,
        "longest_repeated_token_run": 9,
        "language_nll": 1.95,
        "non_empty_rate": 1.0,
    }
    ok, reasons = sft_validation_gate(before, stable)
    assert ok, reasons

    regressed = dict(stable)
    regressed["repetition_rate"] = 0.80
    ok, reasons = sft_validation_gate(before, regressed)
    assert not ok
    assert any("repetition" in reason for reason in reasons)


def test_replay_filter_structurally_excludes_phase5_and_sft_holdouts():
    normal = SFTExample([
        {"role": "user", "content": "Tell me about rain."},
        {"role": "assistant", "content": "Rain falls from clouds."},
    ])
    heldout = SFTExample([
        {"role": "user", "content": "A held out prompt."},
        {"role": "assistant", "content": "A held out answer."},
    ])
    phase5 = SFTExample([
        {"role": "user", "content": "Ciao"},
        {"role": "assistant", "content": "Ciao!"},
    ])
    rows, filtered = _filter_protected_replay(
        [normal, heldout, phase5, normal],
        heldout_sft=[heldout],
    )
    assert filtered == 2
    assert len(rows) == 1
    assert rows[0].messages == normal.messages


def test_sft_validation_metrics_use_separate_heldout_examples():
    pytest.importorskip("torch")
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
    ).validate()
    runtime = GeneralistRuntime(
        CausalTransformerLM(cfg),
        cfg,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )
    examples = [
        SFTExample([
            {"role": "user", "content": "Say a short greeting."},
            {"role": "assistant", "content": "Hi there."},
        ]),
        SFTExample([
            {"role": "user", "content": "Name one fruit."},
            {"role": "assistant", "content": "Apple."},
        ]),
    ]
    report = evaluate_sft_validation(
        runtime,
        examples,
        max_examples=2,
        max_new_tokens=4,
    )
    assert report["suite"] == "phase5-sft-heldout-v1"
    assert report["suite_training_excluded"] is True
    assert report["prompt_count"] == 2
    assert "repetition_rate" in report
    assert "language_nll" in report


def test_phase5_diagnostics_run_on_real_local_generalist_model():
    pytest.importorskip("torch")
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=96,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
    ).validate()
    runtime = GeneralistRuntime(
        CausalTransformerLM(cfg),
        cfg,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )
    report = evaluate_phase5_language(runtime, max_new_tokens=4)
    required = {
        "token_entropy",
        "top1_probability",
        "top5_probability_mass",
        "repetition_rate",
        "longest_repeated_token_run",
        "unique_token_ratio",
        "eos_probability",
        "token_frequency_distribution",
        "generation_length",
        "bpe_token_distribution",
        "language_nll",
        "generation_similarity",
        "exact_accuracy",
        "non_empty_rate",
        "word_output_rate",
    }
    assert required <= set(report)
    assert report["suite_training_excluded"] is True
    assert len(report["traces"]) == len(PHASE5_PROBES)


def test_sampling_controls_preserve_raw_greedy_default():
    torch = pytest.importorskip("torch")

    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
    ).validate()
    model = CausalTransformerLM(cfg)
    tok = ByteTokenizer()
    prompt = torch.tensor([tok.encode("Hello", bos=True)], dtype=torch.long)

    first = model.generate(prompt, max_new_tokens=6, eos_token_id=None)
    second = model.generate(
        prompt,
        max_new_tokens=6,
        eos_token_id=None,
        temperature=0.0,
        top_k=1,
        top_p=0.01,
        repetition_penalty=2.0,
    )
    # Greedy intentionally ignores sampling knobs so canonical evaluation is stable.
    assert torch.equal(first, second)

    torch.manual_seed(123)
    sampled = model.generate(
        prompt,
        max_new_tokens=6,
        eos_token_id=None,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.08,
    )
    assert sampled.shape == first.shape
