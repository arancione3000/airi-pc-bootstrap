from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest
import generalist_lm.bootstrap_training as bootstrap_training

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
    PHASE5_LANGUAGE_REHABILITATION_STAGES,
    _anti_collapse_rescue_gate,
    _anti_collapse_weights,
    _assisted_capacity_target,
    _bootstrap_capacity_target,
    _bootstrap_corpus_target,
    _effective_bootstrap_target,
    _elementary_rehabilitation_rows,
    _filter_protected_replay,
    _grow_bootstrap_runtime,
    _dead_capacity_revival_due,
    _dead_capacity_source_ff_width,
    _revive_dead_ffn_model_capacity,
    _historical_language_recovery_due,
    _historical_language_source_genome,
    _language_rehabilitation_attempts,
    _language_rehabilitation_gate,
    _rehabilitation_cycle_due,
    _rehabilitation_replay_limit,
    _rehabilitation_strategy_rejection_count,
    _residual_language_rehabilitation_attempts,
    _phase5_memory_safe_batch_plan,
    _phase5_parameter_segment_cap,
    _phase5_recovery_plan,
    _phase5_recovery_segment_budget,
    _phase5_success,
    _rehabilitation_needed,
    _rehabilitation_replay_rows,
    _run_language_rehabilitation_stage,
    _sft_row_fingerprint,
    _load_optimizer_checkpoint,
    _optimizer_manifest,
    _optimizer_shard_paths,
    _save_optimizer_checkpoint,
    _segment_language_gate,
    _language_quality,
)
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.pretraining import (
    CorpusDocument,
    load_packed_block_cache,
    save_packed_block_cache,
)
from generalist_lm.phase5_diagnostics import (
    PHASE5_PROBES,
    _single_greedy_trace,
    degeneration_gate,
    evaluate_phase5_language,
    evaluate_sft_validation,
    protected_bootstrap_texts,
    sft_validation_gate,
)
from generalist_lm.research_cycle import _transfer_compatible_weights
from generalist_lm.runtime import GeneralistRuntime
from generalist_lm.tokenizer import ByteTokenizer
from generalist_lm.training import (
    SFTExample,
    causal_training_objective,
    train_sft_residual_recovery,
)




def test_phase5_large_models_use_short_transactional_segments():
    assert _phase5_parameter_segment_cap(
        parameters=7_021_248,
        context_length=128,
    ) is None
    assert _phase5_parameter_segment_cap(
        parameters=20_000_000,
        context_length=128,
    ) == 250_000
    assert _phase5_parameter_segment_cap(
        parameters=50_041_536,
        context_length=128,
    ) == 62_500
    assert _phase5_parameter_segment_cap(
        parameters=80_000_000,
        context_length=128,
    ) == 31_250
    assert _phase5_parameter_segment_cap(
        parameters=50_041_536,
        context_length=256,
    ) == 31_250


def test_phase5_recovery_budget_shrinks_after_rejected_segments():
    assert _phase5_recovery_segment_budget(
        1_000_000,
        consecutive_rejections=0,
    ) == 1_000_000
    assert _phase5_recovery_segment_budget(
        1_000_000,
        consecutive_rejections=1,
    ) == 500_000
    assert _phase5_recovery_segment_budget(
        1_000_000,
        consecutive_rejections=2,
    ) == 250_000
    assert _phase5_recovery_segment_budget(
        1_000_000,
        consecutive_rejections=3,
    ) == 125_000
    assert _phase5_recovery_segment_budget(
        250_000,
        consecutive_rejections=8,
    ) == 62_500


def test_phase5_recovery_plan_escapes_old_lr_floor_and_resets_momentum():
    plan = _phase5_recovery_plan(
        1_000_000,
        parameters=50_041_536,
        context_length=128,
        persisted_lr_scale=0.125,
        consecutive_rejections=2,
    )
    assert plan["effective_budget_tokens"] == 31_250
    assert plan["learning_rate_scale"] == pytest.approx(0.125)
    assert plan["stall_recovery"] is True
    assert plan["reset_optimizer"] is True
    assert plan["forced_stage"] == "B_short_sentence_completion"

    deeper = _phase5_recovery_plan(
        1_000_000,
        parameters=50_041_536,
        context_length=128,
        persisted_lr_scale=0.0625,
        consecutive_rejections=3,
    )
    assert deeper["effective_budget_tokens"] == 31_250
    assert deeper["learning_rate_scale"] == pytest.approx(0.0625)

    floor = _phase5_recovery_plan(
        1_000_000,
        parameters=50_041_536,
        context_length=128,
        persisted_lr_scale=0.001,
        consecutive_rejections=9,
    )
    assert floor["learning_rate_scale"] == pytest.approx(1.0 / 64.0)


def test_phase5_recovery_plan_keeps_successful_rescue_sticky():
    plan = _phase5_recovery_plan(
        1_000_000,
        parameters=50_041_536,
        context_length=128,
        persisted_lr_scale=0.0625,
        consecutive_rejections=0,
        recovery_hold=True,
    )
    assert plan["effective_budget_tokens"] == 31_250
    assert plan["learning_rate_scale"] == pytest.approx(0.0625)
    assert plan["stall_recovery"] is True
    assert plan["reset_optimizer"] is False
    assert plan["forced_stage"] == "B_short_sentence_completion"


def test_phase5_recovery_plan_reenters_rescue_after_low_lr_rejection():
    plan = _phase5_recovery_plan(
        1_000_000,
        parameters=50_041_536,
        context_length=128,
        persisted_lr_scale=0.0375,
        consecutive_rejections=1,
    )
    assert plan["effective_budget_tokens"] == 31_250
    assert plan["learning_rate_scale"] == pytest.approx(0.0375)
    assert plan["stall_recovery"] is True
    assert plan["reset_optimizer"] is True
    assert plan["forced_stage"] == "B_short_sentence_completion"


def test_phase5_recovery_plan_does_not_penalize_healthy_training():
    plan = _phase5_recovery_plan(
        1_000_000,
        parameters=50_041_536,
        context_length=128,
        persisted_lr_scale=0.3,
        consecutive_rejections=0,
    )
    assert plan["effective_budget_tokens"] == 62_500
    assert plan["learning_rate_scale"] == pytest.approx(0.3)
    assert plan["stall_recovery"] is False
    assert plan["reset_optimizer"] is False
    assert plan["forced_stage"] is None


def test_phase5_memory_plan_keeps_7m_fast_and_50m_bounded():
    micro, accumulation = _phase5_memory_safe_batch_plan(
        parameters=7_021_248,
        context_length=128,
        requested_batch_size=32,
    )
    assert micro == 32
    assert accumulation == 1

    micro, accumulation = _phase5_memory_safe_batch_plan(
        parameters=50_041_536,
        context_length=128,
        requested_batch_size=32,
    )
    assert micro == 2
    assert accumulation == 16
    assert micro * accumulation == 32


def test_phase5_memory_plan_bounds_future_large_lineage():
    micro, accumulation = _phase5_memory_safe_batch_plan(
        parameters=80_000_000,
        context_length=256,
        requested_batch_size=32,
    )
    assert micro == 1
    assert accumulation == 32
    assert micro * accumulation == 32



def _tiny_phase5_model():
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
    ).validate()
    return CausalTransformerLM(cfg)


def test_phase5_optimizer_state_shards_and_roundtrips_exactly(tmp_path: Path):
    torch = pytest.importorskip("torch")
    model = _tiny_phase5_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    ids = torch.randint(0, 264, (2, 16), dtype=torch.long)
    loss = model(ids, labels=ids)["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    path = tmp_path / "optimizer.pt"
    storage = _save_optimizer_checkpoint(
        optimizer,
        path,
        shard_raw_bytes=32_000,
    )
    assert storage["storage"] == "sharded"
    assert _optimizer_manifest(path) is not None
    shards = _optimizer_shard_paths(path)
    assert len(shards) >= 2
    assert all(shard.stat().st_size < 200_000 for shard in shards)

    fresh_model = _tiny_phase5_model()
    restored = torch.optim.AdamW(
        fresh_model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )
    loaded = _load_optimizer_checkpoint(restored, path)
    assert loaded["storage"] == "sharded"

    expected = optimizer.state_dict()
    actual = restored.state_dict()
    assert expected["param_groups"] == actual["param_groups"]
    assert expected["state"].keys() == actual["state"].keys()
    for param_id, expected_state in expected["state"].items():
        actual_state = actual["state"][param_id]
        assert expected_state.keys() == actual_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = actual_state[key]
            if torch.is_tensor(expected_value):
                assert torch.equal(expected_value, actual_value)
            else:
                assert expected_value == actual_value


def test_phase5_optimizer_shards_reject_tampering(tmp_path: Path):
    torch = pytest.importorskip("torch")
    model = _tiny_phase5_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    ids = torch.randint(0, 264, (2, 16), dtype=torch.long)
    model(ids, labels=ids)["loss"].backward()
    optimizer.step()

    path = tmp_path / "optimizer.pt"
    _save_optimizer_checkpoint(optimizer, path, shard_raw_bytes=32_000)
    shard = _optimizer_shard_paths(path)[0]
    with shard.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="optimizer shard (size|digest) mismatch"):
        _load_optimizer_checkpoint(
            torch.optim.AdamW(_tiny_phase5_model().parameters(), lr=3e-4),
            path,
        )


def test_continuum_watchdog_recovers_stranded_language_training():
    workflow = Path(".github/workflows/generalist-continuum.yml").read_text(
        encoding="utf-8"
    )
    assert "cron: '*/5 * * * *'" in workflow
    assert "bootstrap-data/progress.json?ref=generalist-state" in workflow
    assert "if (( target > processed )); then" in workflow
    assert 'recovery_target="${target}"' in workflow
    assert "recovery_target=100000000" in workflow
    assert "recovery_target=250000000" in workflow
    assert "recovery_target=500000000" in workflow
    assert "recovery_target=1000000000" in workflow
    assert "generalist-bootstrap.yml/dispatches" in workflow
    assert "Recovered AIRI Phase-5 language training" in workflow
    assert 'echo "active=true" >> "${GITHUB_OUTPUT}"' in workflow
    assert "if: needs.bootstrap-gate.outputs.active != 'true'" in workflow


def test_bootstrap_restores_state_shallow_and_unshallows_only_for_rebase():
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    assert "git clone --quiet --depth 1 --branch generalist-state --single-branch" in workflow
    assert 'git rev-parse --is-shallow-repository' in workflow
    assert "git fetch --quiet --unshallow origin generalist-state" in workflow
    assert "generalist-state advanced; expanding shallow history only for safe reconciliation." in workflow


def test_phase5_fasttrack_handoff_preserves_live_app_lineage():
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(encoding="utf-8")
    assert "Dispatch next in-place language rung" in workflow
    assert "same persisted AIRI Phase-5 lineage" in workflow
    assert "next=100000000" in workflow
    assert "next=250000000" in workflow
    assert "next=500000000" in workflow
    assert "next=1000000000" in workflow
    assert "ARCHITECTURE_INTERVAL_TOKENS: '5000000'" in workflow
    assert "force_search" in workflow
    assert "resume_bootstrap_target" in workflow
    assert '"handoff":"converged_swarm_first"' not in workflow
    assert "Generalist swarm must run once before it is dispatched" not in workflow
    assert "git rebase origin/generalist-state" in workflow
    assert "transactionally persist Phase-5 checkpoint" in workflow
    assert "refusing an unsafe overwrite" in workflow
    assert "Restore reusable packed-token cache" in workflow
    assert "airi-generalist-phase5-packed-v1-" in workflow
    assert "github.run_id" in workflow


def test_generic_architecture_yields_to_active_explicit_lineage_handoff():
    workflow = Path(".github/workflows/generalist-architecture-search.yml").read_text(
        encoding="utf-8"
    )
    assert "live-lineage-handoff.json" in workflow
    assert '"${FORCE_SEARCH}" != "true"' in workflow
    assert 'select(.event == "workflow_dispatch")' in workflow
    assert "explicit_active > 0" in workflow
    assert "explicit_live_lineage_handoff_owns_architecture_window" in workflow
    assert "run_search=false" in workflow
    assert "matrix=[]" in workflow


def test_bootstrap_janitor_preserves_old_worker_without_replacement():
    workflow = Path(
        ".github/workflows/generalist-state-writer-janitor.yml"
    ).read_text(encoding="utf-8")
    assert "if (( current_active > 0 )); then" in workflow
    assert "if (( stale_active > 0 )); then" in workflow
    assert "AIRI_BOOTSTRAP_JANITOR=PRESERVE" in workflow
    assert "No replacement exists. Keep the existing bootstrap alive" in workflow
    assert "AIRI_BOOTSTRAP_JANITOR_RESTART=PASS" in workflow


def test_phase5_refreshes_mobile_bundle_after_each_persisted_segment():
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(encoding="utf-8")
    assert "generalist-mobile-export.yml/dispatches" in workflow
    assert "Queued mobile export for the freshly persisted AIRI Live checkpoint." in workflow


def test_mobile_export_uses_actual_live_checkpoint_parameters_and_probe():
    source = Path("computer/generalist_lm/mobile_export.py").read_text(encoding="utf-8")
    assert '"parameters": int(parameter_count(runtime.model))' in source
    assert "active_probe = evaluate_phase5_language(active_runtime)" in source
    assert '"live_probe": active_probe' in source


def test_assisted_50m_growth_is_persisted_before_more_training():
    source = Path("computer/generalist_lm/bootstrap_training.py").read_text(
        encoding="utf-8"
    )
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    assert '"capacity_growth_only": True' in source
    assert 'reason="assisted_50m_capacity_growth"' in source
    assert '"segment_tokens_processed": 0' in source
    assert ".capacity_growth_only // false" in workflow
    assert "AIRI 50M assisted growth persisted" in workflow


def test_phase5_100m_amortizes_setup_without_changing_batch_or_lr():
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(encoding="utf-8")
    assert "segment_tokens=1000000" in workflow
    assert "if (( TARGET_TOKENS >= 100000000 )); then" in workflow
    assert '--segment-tokens "${segment_tokens}"' in workflow
    assert "--batch-size 32" in workflow
    assert "--learning-rate 0.0003" in workflow


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
    assert _bootstrap_corpus_target(250_000_000) == 40_000_000
    assert _bootstrap_corpus_target(500_000_000) == 60_000_000
    assert _bootstrap_corpus_target(1_000_000_000) == 100_000_000


def test_phase5_conversation_rescue_has_explicit_capacity_rungs():
    assert _bootstrap_capacity_target(5_000_000) is None
    assert _bootstrap_capacity_target(19_999_999) is None
    assert _bootstrap_capacity_target(20_000_000) == 1_250_000
    assert _bootstrap_capacity_target(50_000_000) == 3_000_000
    assert _bootstrap_capacity_target(100_000_000) == 7_000_000
    assert _bootstrap_capacity_target(250_000_000) == 12_000_000
    assert _bootstrap_capacity_target(500_000_000) == 20_000_000
    assert _bootstrap_capacity_target(1_000_000_000) == 32_000_000


def test_current_airi_lineage_receives_one_time_50m_capacity_assist():
    progress = {
        "lineage_id": "airi-5d3d25177d2e83f7",
        "assisted_capacity_growth_completed": False,
    }
    assert _assisted_capacity_target(progress, current_parameters=7_021_248) == 50_000_000
    assert _assisted_capacity_target(progress, current_parameters=49_000_000) is None

    completed = dict(progress)
    completed["assisted_capacity_growth_completed"] = True
    assert _assisted_capacity_target(completed, current_parameters=7_021_248) is None

    other = dict(progress)
    other["lineage_id"] = "future-airi-lineage"
    assert _assisted_capacity_target(other, current_parameters=7_021_248) is None


def test_50m_assist_has_a_same_width_function_preserving_candidate():
    from generalist_lm.evolution import GeneralistGenome, progressive_scale_candidate
    from generalist_lm.model import estimate_parameter_count

    live = GeneralistGenome(
        generation=4,
        parent_id="generalist-3-scale-56d0056e",
        genome_id="generalist-4-scale-367e26e1",
        context_length=128,
        d_model=96,
        n_heads=4,
        n_layers=12,
        d_ff=1888,
        dropout=0.0,
        learning_rate=0.0001875,
        tokenizer_version="bpe-v1",
        reasoning_depth=1,
    ).validate()
    grown = progressive_scale_candidate(
        live,
        target_parameters=50_000_000,
        vocab_size=384,
        max_width=512,
        max_layers=12,
        prefer_function_preserving=True,
    )
    parameters = estimate_parameter_count(grown.model_config(384))

    assert grown.d_model == live.d_model
    assert grown.n_layers >= live.n_layers
    assert grown.d_ff >= live.d_ff
    assert 49_000_000 <= parameters <= 51_000_000


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


def test_function_preserving_ff_growth_keeps_new_units_trainable():
    torch = pytest.importorskip("torch")
    source_cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tokenizer_version="byte-v1",
        ff_variant="swiglu",
    ).validate()
    target_cfg = GeneralistLMConfig(
        **{
            **source_cfg.to_dict(),
            "d_ff": 96,
        }
    ).validate()
    tokenizer = ByteTokenizer()
    source = CausalTransformerLM(source_cfg)
    target = CausalTransformerLM(target_cfg)

    transfer = _transfer_compatible_weights(
        source,
        target,
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
    )
    assert transfer["function_preserving_growth"] is True

    ids = torch.tensor([[1, 40, 41, 42, 43, 44, 45, 46]], dtype=torch.long)
    source.eval()
    target.eval()
    with torch.no_grad():
        source_logits = source(ids)["logits"]
        target_logits = target(ids)["logits"]
    assert torch.allclose(source_logits, target_logits, atol=1e-6, rtol=1e-6)

    up = target.blocks[0].ff.up.weight
    down = target.blocks[0].ff.down.weight
    old_ff = source_cfg.d_ff
    new_ff = target_cfg.d_ff

    # New gate/value rows keep normal initialization, while their output
    # columns are zero so the initial function is unchanged.
    assert torch.count_nonzero(up[old_ff:new_ff]).item() > 0
    assert torch.count_nonzero(up[new_ff + old_ff: 2 * new_ff]).item() > 0
    assert torch.count_nonzero(down[:, old_ff:]).item() == 0

    target.train()
    target.zero_grad(set_to_none=True)
    loss = target(ids, labels=ids)["loss"]
    loss.backward()
    assert target.blocks[0].ff.down.weight.grad[:, old_ff:].abs().sum().item() > 0


def test_residual_rehabilitation_uses_its_own_rejection_counter():
    old_strategy_state = {
        "consecutive_rejections": 5,
    }
    revived_progress = {
        "dead_capacity_revival": {"completed": True},
    }

    assert _rehabilitation_strategy_rejection_count(
        revived_progress,
        old_strategy_state,
    ) == 0

    residual_state = {
        "consecutive_rejections": 7,
        "residual_consecutive_rejections": 2,
    }
    assert _rehabilitation_strategy_rejection_count(
        revived_progress,
        residual_state,
    ) == 2

    pre_revival_progress = {}
    assert _rehabilitation_strategy_rejection_count(
        pre_revival_progress,
        {"consecutive_rejections": 3},
    ) == 3


def test_residual_rehabilitation_plan_strengthens_kl_instead_of_unlikelihood():
    first = _residual_language_rehabilitation_attempts(
        stage="R1_bilingual_foundations",
        consecutive_rejections=0,
    )
    later = _residual_language_rehabilitation_attempts(
        stage="R1_bilingual_foundations",
        consecutive_rejections=2,
    )

    assert first[0][1] == pytest.approx(1.0e-5)
    assert first[0][2] <= 0.05
    assert first[0][3] <= 1.25
    assert first[0][4] == pytest.approx(1.75)
    assert first[0][5] is False
    assert later[0][1] < first[0][1]
    assert later[0][4] > first[0][4]
    assert later[0][2] == first[0][2]
    assert later[0][3] == first[0][3]


def test_residual_kl_recovery_keeps_legacy_weights_bit_stable():
    torch = pytest.importorskip("torch")
    source_ff = 64
    target_ff = 96
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=target_ff,
        dropout=0.0,
        tokenizer_version="byte-v1",
        ff_variant="swiglu",
    ).validate()
    model = CausalTransformerLM(cfg)

    # Recreate the old dead branch, then revive it exactly as the live 50M was.
    with torch.no_grad():
        for block in model.blocks:
            block.ff.up.weight[source_ff:target_ff].zero_()
            block.ff.up.weight[target_ff + source_ff : 2 * target_ff].zero_()
            block.ff.down.weight[:, source_ff:target_ff].zero_()
    runtime = GeneralistRuntime(
        model,
        cfg,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )
    _revive_dead_ffn_model_capacity(
        runtime,
        source_d_ff=source_ff,
        seed=123,
    )

    reference = CausalTransformerLM(cfg)
    reference.load_state_dict(model.state_dict())
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    examples = [
        SFTExample([
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello there."},
        ]),
        SFTExample([
            {"role": "user", "content": "Name one fruit."},
            {"role": "assistant", "content": "Apple."},
        ]),
    ]

    report = train_sft_residual_recovery(
        model,
        reference,
        ByteTokenizer(),
        examples,
        anchor_examples=[
            SFTExample([
                {"role": "user", "content": "What is two plus two?"},
                {"role": "assistant", "content": "Two plus two is four."},
            ])
        ],
        source_d_ff=source_ff,
        steps=3,
        batch_size=1,
        learning_rate=2e-4,
        seed=9,
        device="cpu",
        repetition_unlikelihood_weight=0.0,
        eos_loss_weight=1.0,
        kl_weight=1.0,
        train_upstream=False,
    )

    changed_new_down = False
    after = model.state_dict()
    for name, old_value in before.items():
        new_value = after[name]
        if name.endswith(".ff.down.weight"):
            assert torch.equal(
                old_value[:, :source_ff],
                new_value[:, :source_ff],
            )
            if not torch.equal(
                old_value[:, source_ff:target_ff],
                new_value[:, source_ff:target_ff],
            ):
                changed_new_down = True
        else:
            assert torch.equal(old_value, new_value)

    assert changed_new_down is True
    assert report["mode"] == "residual_kl_recovery"
    assert report["train_upstream"] is False
    assert report["teacher_kl_weight"] == pytest.approx(1.0)
    assert report["anchor_example_count"] == 1
    assert report["mean_teacher_kl_loss"] >= 0.0
    assert report["trainable_coordinate_count"] > 0


def test_dead_capacity_revival_precedes_historical_replacement():
    progress = {
        "lineage_id": "airi-5d3d25177d2e83f7",
        "assisted_capacity_growth_completed": True,
        "language_rehabilitation": {"consecutive_rejections": 4},
    }
    assert _dead_capacity_revival_due(progress) is True
    assert _historical_language_recovery_due(progress) is False

    revived = {
        **progress,
        "dead_capacity_revival": {"completed": True},
        "language_rehabilitation": {"consecutive_rejections": 0},
    }
    assert _dead_capacity_revival_due(revived) is False
    assert _historical_language_recovery_due(revived) is False

    almost_due = {
        **revived,
        "historical_language_recovery_source_verified": True,
        "language_rehabilitation": {"consecutive_rejections": 5},
    }
    assert _historical_language_recovery_due(almost_due) is False

    historical_due = {
        **revived,
        "historical_language_recovery_source_verified": True,
        "language_rehabilitation": {"consecutive_rejections": 6},
    }
    assert _historical_language_recovery_due(historical_due) is True

    completed = {
        **historical_due,
        "historical_language_recovery": {"completed": True},
    }
    assert _historical_language_recovery_due(completed) is False


def test_dead_capacity_revival_preserves_logits_and_enables_new_gradients():
    torch = pytest.importorskip("torch")
    source_ff = 64
    target_ff = 96
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=target_ff,
        dropout=0.0,
        tokenizer_version="byte-v1",
        ff_variant="swiglu",
    ).validate()
    runtime = GeneralistRuntime(
        CausalTransformerLM(cfg),
        cfg,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )

    # Reproduce the legacy 7M->50M transfer bug: both sides of the newly added
    # SwiGLU branch were zero, so it was function preserving but gradient dead.
    with torch.no_grad():
        for block in runtime.model.blocks:
            block.ff.up.weight[source_ff:target_ff].zero_()
            block.ff.up.weight[target_ff + source_ff : 2 * target_ff].zero_()
            block.ff.down.weight[:, source_ff:target_ff].zero_()

    ids = torch.tensor([[1, 40, 41, 42, 43, 44, 45, 46]], dtype=torch.long)
    runtime.model.eval()
    with torch.no_grad():
        before = runtime.model(ids)["logits"].clone()

    report = _revive_dead_ffn_model_capacity(
        runtime,
        source_d_ff=source_ff,
        seed=123,
    )

    runtime.model.eval()
    with torch.no_grad():
        after = runtime.model(ids)["logits"]
    assert torch.equal(before, after)
    assert report["max_logit_delta"] == 0.0
    assert report["new_down_gradient_sum"] > 0.0

    for block in runtime.model.blocks:
        assert torch.count_nonzero(
            block.ff.up.weight[source_ff:target_ff]
        ).item() > 0
        assert torch.count_nonzero(
            block.ff.up.weight[target_ff + source_ff : 2 * target_ff]
        ).item() > 0
        assert torch.count_nonzero(
            block.ff.down.weight[:, source_ff:target_ff]
        ).item() == 0


def test_dead_capacity_source_width_resolves_pre_50m_ffn():
    progress = {
        "capacity_growth_history": [
            {
                "parameters": 7_021_248,
                "source_parameters": 1_251_264,
                "genome": {"d_ff": 1888},
            },
            {
                "parameters": 50_041_536,
                "source_parameters": 7_021_248,
                "genome": {"d_ff": 14336},
            },
        ]
    }
    assert _dead_capacity_source_ff_width(
        progress,
        current_d_ff=14336,
    ) == 1888


def test_historical_language_recovery_resolves_exact_source_genome():
    from generalist_lm.evolution import GeneralistGenome

    source = GeneralistGenome(
        generation=4,
        parent_id="parent",
        genome_id="healthy-7m",
        context_length=128,
        d_model=96,
        n_heads=4,
        n_layers=12,
        d_ff=1888,
        dropout=0.0,
        learning_rate=0.0001875,
        tokenizer_version="bpe-v1",
        reasoning_depth=1,
    ).validate()
    progress = {
        "capacity_growth_history": [
            {
                "parameters": 7_021_248,
                "genome": source.to_dict(),
            }
        ]
    }

    recovered = _historical_language_source_genome(
        progress,
        source_parameters=7_021_248,
    )
    assert recovered.to_dict() == source.to_dict()

    with pytest.raises(RuntimeError):
        _historical_language_source_genome(
            progress,
            source_parameters=50_041_536,
        )


def test_historical_recovery_workflow_is_transactional_and_pinned():
    workflow = Path(".github/workflows/generalist-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    source = Path("computer/generalist_lm/bootstrap_training.py").read_text(
        encoding="utf-8"
    )
    assert "a33056c2beef538ec68a7d3c88bd65c2fc69b079" in workflow
    assert "generalist-state/bootstrap-data/best" in workflow
    assert ".parameters == 7021248" in workflow
    assert ".tokens_processed == 31210573" in workflow
    assert "--historical-recovery-source" in workflow
    assert "historical_language_recovery_source_verified" in workflow
    assert ".dead_capacity_revival_only // false" in workflow
    assert '"dead_capacity_revival_only": True' in source
    assert ".historical_recovery_only // false" in workflow
    assert '"historical_recovery_only": True' in source
    assert '"discarded_effective_tokens"' in source
    assert 'reason="phase5_historical_language_recovery"' in source


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



def _language_report(
    *,
    nll: float,
    repetition: float,
    similarity: float,
    multiword: float,
    pathological: bool,
) -> dict:
    return {
        "language_nll": nll,
        "repetition_rate": repetition,
        "generation_similarity": similarity,
        "multiword_output_rate": multiword,
        "non_empty_rate": 1.0,
        "token_entropy": 3.0,
        "pathological_repetition": pathological,
        "longest_repeated_token_run": 4 if not pathological else 12,
    }


def test_language_rehabilitation_curriculum_is_bilingual_elementary_and_protected():
    curriculum = _elementary_rehabilitation_rows()
    assert tuple(curriculum) == PHASE5_LANGUAGE_REHABILITATION_STAGES
    protected = protected_bootstrap_texts()

    for stage in PHASE5_LANGUAGE_REHABILITATION_STAGES:
        assert set(curriculum[stage]) == {"it", "en"}
        assert len(curriculum[stage]["it"]) >= 4
        assert len(curriculum[stage]["en"]) >= 4
        for language in ("it", "en"):
            for row in curriculum[stage][language]:
                contents = {
                    " ".join(message["content"].strip().casefold().split())
                    for message in row.messages
                }
                assert not contents & protected


def test_language_rehabilitation_replay_filters_holdouts_and_stays_bounded():
    curriculum = _elementary_rehabilitation_rows()
    heldout = SFTExample([
        {"role": "user", "content": "held-out prompt"},
        {"role": "assistant", "content": "held-out answer"},
    ])
    leaked_probe = SFTExample([
        {"role": "user", "content": PHASE5_PROBES[0]["prompt"]},
        {"role": "assistant", "content": "not the protected target"},
    ])
    replay = [heldout, leaked_probe] + [
        SFTExample([
            {"role": "user", "content": f"safe replay {index}"},
            {"role": "assistant", "content": f"safe answer {index}"},
        ])
        for index in range(20)
    ]

    rows, counts = _rehabilitation_replay_rows(
        "R1_bilingual_foundations",
        curriculum,
        replay,
        heldout_sft=[heldout],
        max_replay_rows=8,
    )
    normalized_messages = {
        " ".join(message["content"].strip().casefold().split())
        for row in rows
        for message in row.messages
    }
    assert "held-out prompt" not in normalized_messages
    assert "ciao" not in normalized_messages
    assert counts["protected_rows_filtered"] == 2
    assert counts["protected_replay_rows"] == 8
    assert counts["elementary_it_rows"] == len(
        curriculum["R1_bilingual_foundations"]["it"]
    )
    assert counts["elementary_en_rows"] == len(
        curriculum["R1_bilingual_foundations"]["en"]
    )


def test_language_rehabilitation_retries_become_repetition_breakers_after_two_rollbacks():
    base = _language_rehabilitation_attempts(
        parameters=50_041_536,
        stage="R1_bilingual_foundations",
        consecutive_rejections=0,
    )
    retry = _language_rehabilitation_attempts(
        parameters=50_041_536,
        stage="R1_bilingual_foundations",
        consecutive_rejections=1,
    )
    rescue = _language_rehabilitation_attempts(
        parameters=50_041_536,
        stage="R1_bilingual_foundations",
        consecutive_rejections=2,
    )
    later_rescue = _language_rehabilitation_attempts(
        parameters=50_041_536,
        stage="R1_bilingual_foundations",
        consecutive_rejections=3,
    )

    assert retry != base
    assert rescue != retry
    assert len(rescue) == 3
    assert retry[0][1] < base[0][1]
    assert rescue[0][1] < retry[0][1]
    assert rescue[0][2] > retry[0][2]
    assert rescue[0][3] > retry[0][3]
    assert later_rescue[0][1] < rescue[0][1]
    assert later_rescue[0][2] > rescue[0][2]
    assert later_rescue[0][3] > rescue[0][3]


def test_language_rehabilitation_retry_expands_protected_replay_diversity():
    assert _rehabilitation_replay_limit("R1_bilingual_foundations", 0) == 48
    assert _rehabilitation_replay_limit("R1_bilingual_foundations", 1) == 96
    assert _rehabilitation_replay_limit("R1_bilingual_foundations", 2) == 144
    assert _rehabilitation_replay_limit("R1_bilingual_foundations", 3) == 192
    assert _rehabilitation_replay_limit("R3_short_dialogue", 1) == 192
    assert _rehabilitation_replay_limit("R3_short_dialogue", 10) == 192


def test_language_rehabilitation_retry_rotates_protected_replay_rows():
    curriculum = _elementary_rehabilitation_rows()
    replay = [
        SFTExample([
            {"role": "user", "content": f"safe replay prompt {index}"},
            {"role": "assistant", "content": f"safe replay answer {index}"},
        ])
        for index in range(12)
    ]

    rows_a, counts_a = _rehabilitation_replay_rows(
        "R1_bilingual_foundations",
        curriculum,
        replay,
        max_replay_rows=4,
        replay_offset=0,
    )
    rows_b, counts_b = _rehabilitation_replay_rows(
        "R1_bilingual_foundations",
        curriculum,
        replay,
        max_replay_rows=4,
        replay_offset=3,
    )

    replay_a = {_sft_row_fingerprint(row) for row in rows_a[-4:]}
    replay_b = {_sft_row_fingerprint(row) for row in rows_b[-4:]}
    assert replay_a != replay_b
    assert counts_a["protected_replay_offset"] == 0
    assert counts_b["protected_replay_offset"] == 3


def test_r1_recovery_entropy_uses_durable_anchor_not_collapsed_baseline():
    anchor_report = _language_report(
        nll=2.858,
        repetition=0.443,
        similarity=0.115,
        multiword=0.429,
        pathological=False,
    )
    anchor_report["token_entropy"] = 2.674
    anchor_report["longest_repeated_token_run"] = 2

    collapsed = _language_report(
        nll=4.100,
        repetition=0.449,
        similarity=0.033,
        multiword=0.0,
        pathological=True,
    )
    collapsed["token_entropy"] = 4.241
    collapsed["longest_repeated_token_run"] = 39

    recovered = _language_report(
        nll=2.647,
        repetition=0.277,
        similarity=0.183,
        multiword=0.286,
        pathological=False,
    )
    recovered["token_entropy"] = 2.386
    recovered["longest_repeated_token_run"] = 4

    accepted, report = _language_rehabilitation_gate(
        "R1_bilingual_foundations",
        collapsed,
        recovered,
        anchor_report,
    )

    assert accepted is True
    assert report["recovery_mode"] is True
    assert report["protected_entropy_floor"] == pytest.approx(2.074)
    assert report["anchor_violations_after"] == []
    assert "protected token entropy collapsed" not in report["reasons"]


def test_r1_non_recovery_still_rejects_material_entropy_collapse():
    stable = _language_report(
        nll=2.6,
        repetition=0.30,
        similarity=0.20,
        multiword=0.50,
        pathological=False,
    )
    stable["token_entropy"] = 4.0
    candidate = dict(stable)
    candidate["token_entropy"] = 2.5

    accepted, report = _language_rehabilitation_gate(
        "R1_bilingual_foundations",
        stable,
        candidate,
        stable,
    )

    assert accepted is False
    assert report["recovery_mode"] is False
    assert report["protected_entropy_floor"] == pytest.approx(3.25)
    assert "protected token entropy collapsed" in report["reasons"]


def test_language_rehabilitation_uses_progressive_fail_closed_gates():
    anchor = _language_report(
        nll=2.80,
        repetition=0.42,
        similarity=0.18,
        multiword=0.70,
        pathological=False,
    )
    collapsed = _language_report(
        nll=3.30,
        repetition=0.70,
        similarity=0.08,
        multiword=0.20,
        pathological=True,
    )
    repetition_recovered = _language_report(
        nll=3.31,
        repetition=0.60,
        similarity=0.08,
        multiword=0.20,
        pathological=False,
    )
    accepted, report = _language_rehabilitation_gate(
        "R1_bilingual_foundations",
        collapsed,
        repetition_recovered,
        anchor,
    )
    assert accepted is True
    assert report["gate"] == "repetition_recovery"

    elementary_recovered = _language_report(
        nll=3.20,
        repetition=0.56,
        similarity=0.12,
        multiword=0.43,
        pathological=False,
    )
    accepted, report = _language_rehabilitation_gate(
        "R2_simple_responses",
        repetition_recovered,
        elementary_recovered,
        anchor,
    )
    assert accepted is True
    assert report["gate"] == "elementary_language"

    dialogue_regressed = dict(elementary_recovered)
    dialogue_regressed["multiword_output_rate"] = 0.20
    accepted, report = _language_rehabilitation_gate(
        "R3_short_dialogue",
        elementary_recovered,
        dialogue_regressed,
        anchor,
    )
    assert accepted is False
    assert "multi-word" in " ".join(report["reasons"])


def test_language_rehabilitation_reenters_when_live_checkpoint_collapses():
    anchor = _language_report(
        nll=2.80,
        repetition=0.42,
        similarity=0.18,
        multiword=0.70,
        pathological=False,
    )
    healthy = dict(anchor)
    collapsed = dict(anchor)
    collapsed["pathological_repetition"] = True
    collapsed["repetition_rate"] = 0.72

    assert _rehabilitation_needed(healthy, anchor) is False
    assert _rehabilitation_needed(collapsed, anchor) is True


def test_language_rehabilitation_finishes_an_active_progressive_cycle():
    healthy = {
        "language_nll": 2.8,
        "pathological_repetition": False,
        "repetition_rate": 0.3,
        "generation_similarity": 0.7,
        "non_empty_rate": 1.0,
        "multiword_output_rate": 0.8,
        "token_entropy": 3.0,
    }
    assert _rehabilitation_cycle_due(healthy, healthy, 0) is False
    assert _rehabilitation_cycle_due(healthy, healthy, 1) is True
    assert _rehabilitation_cycle_due(healthy, healthy, 2) is True
    assert _rehabilitation_cycle_due(healthy, healthy, 3) is False


def test_language_rehabilitation_stage_counts_only_accepted_exact_tokens(
    tmp_path: Path,
    monkeypatch,
):
    pytest.importorskip("torch")
    runtime = GeneralistRuntime.fresh(_tiny_phase5_model().config)
    before = _language_report(
        nll=3.30,
        repetition=0.70,
        similarity=0.08,
        multiword=0.20,
        pathological=True,
    )
    after = _language_report(
        nll=3.31,
        repetition=0.58,
        similarity=0.09,
        multiword=0.30,
        pathological=False,
    )
    reports = iter((before, after))
    monkeypatch.setattr(
        bootstrap_training,
        "evaluate_phase5_language",
        lambda _runtime: dict(next(reports)),
    )
    stable_sft = {
        "language_nll": 2.0,
        "repetition_rate": 0.20,
        "token_entropy": 3.0,
        "unique_token_ratio": 0.80,
        "longest_repeated_token_run": 2,
        "non_empty_rate": 1.0,
    }
    monkeypatch.setattr(
        bootstrap_training,
        "evaluate_sft_validation",
        lambda *_args, **_kwargs: dict(stable_sft),
    )
    monkeypatch.setattr(
        bootstrap_training,
        "train_sft",
        lambda *_args, **_kwargs: {
            "ok": True,
            "supervised_tokens": 321,
        },
    )

    selected, report = _run_language_rehabilitation_stage(
        runtime,
        "R1_bilingual_foundations",
        [SFTExample([
            {"role": "user", "content": "safe prompt"},
            {"role": "assistant", "content": "safe answer"},
        ])],
        [SFTExample([
            {"role": "user", "content": "heldout prompt"},
            {"role": "assistant", "content": "heldout answer"},
        ])],
        before,
        bootstrap_root=tmp_path,
        base_model_sha="a" * 64,
        seed=7,
    )
    assert report["accepted"] is True
    assert report["accepted_supervised_tokens"] == 321
    assert report["rollback_verified"] is True
    assert sum(p.numel() for p in selected.model.parameters()) == sum(
        p.numel() for p in runtime.model.parameters()
    )


def test_language_rehabilitation_stage_restores_exact_checkpoint_on_rejection(
    tmp_path: Path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    runtime = GeneralistRuntime.fresh(_tiny_phase5_model().config)
    original = {
        name: tensor.detach().clone()
        for name, tensor in runtime.model.state_dict().items()
    }
    unchanged = _language_report(
        nll=3.20,
        repetition=0.56,
        similarity=0.12,
        multiword=0.43,
        pathological=False,
    )
    monkeypatch.setattr(
        bootstrap_training,
        "evaluate_phase5_language",
        lambda _runtime: dict(unchanged),
    )
    stable_sft = {
        "language_nll": 2.0,
        "repetition_rate": 0.20,
        "token_entropy": 3.0,
        "unique_token_ratio": 0.80,
        "longest_repeated_token_run": 2,
        "non_empty_rate": 1.0,
    }
    monkeypatch.setattr(
        bootstrap_training,
        "evaluate_sft_validation",
        lambda *_args, **_kwargs: dict(stable_sft),
    )
    monkeypatch.setattr(
        bootstrap_training,
        "train_sft",
        lambda *_args, **_kwargs: {
            "ok": True,
            "supervised_tokens": 999,
        },
    )

    restored, report = _run_language_rehabilitation_stage(
        runtime,
        "R2_simple_responses",
        [SFTExample([
            {"role": "user", "content": "safe prompt"},
            {"role": "assistant", "content": "safe answer"},
        ])],
        [SFTExample([
            {"role": "user", "content": "heldout prompt"},
            {"role": "assistant", "content": "heldout answer"},
        ])],
        unchanged,
        bootstrap_root=tmp_path,
        base_model_sha="b" * 64,
        seed=11,
    )
    assert report["accepted"] is False
    assert report["rolled_back"] is True
    assert report["rollback_verified"] is True
    assert report["accepted_supervised_tokens"] == 0
    for name, tensor in restored.model.state_dict().items():
        assert torch.equal(tensor, original[name])


def test_checkpoint_audit_enforces_valid_tokens_lineage_and_rollback_contract():
    audit_source = Path("computer/generalist_lm/bootstrap_audit.py").read_text(
        encoding="utf-8"
    )
    audit_workflow = Path(
        ".github/workflows/generalist-bootstrap-audit.yml"
    ).read_text(encoding="utf-8")
    bootstrap_workflow = Path(
        ".github/workflows/generalist-bootstrap.yml"
    ).read_text(encoding="utf-8")

    assert '"valid_tokens_processed"' in audit_source
    assert '"accepted_rehabilitation_tokens"' in audit_source
    assert '"lineage_preserved"' in audit_source
    assert '"rollback_verified"' in audit_source
    assert ".valid_tokens.invariant_holds == true" in audit_workflow
    assert ".language_rehabilitation.rollback_verified == true" in audit_workflow
    assert "bootstrap-data/language-rehabilitation.json" in bootstrap_workflow


def test_segment_language_guard_rejects_regression_from_good_anchor():
    anchor = _language_report(
        nll=2.85,
        repetition=0.44,
        similarity=0.12,
        multiword=0.43,
        pathological=False,
    )
    before = dict(anchor)
    after = _language_report(
        nll=3.40,
        repetition=0.60,
        similarity=0.08,
        multiword=0.40,
        pathological=True,
    )

    accepted, report = _segment_language_gate(before, after, anchor)

    assert accepted is False
    assert report["recovery_mode"] is False
    assert report["after_anchor_violations"]
    assert any(
        "pathological repetition" in reason
        for reason in report["after_anchor_violations"] + report["local_reasons"]
    )


def test_segment_language_guard_accepts_measurable_recovery_toward_anchor():
    anchor = _language_report(
        nll=2.85,
        repetition=0.44,
        similarity=0.12,
        multiword=0.43,
        pathological=False,
    )
    before = _language_report(
        nll=3.75,
        repetition=0.53,
        similarity=0.18,
        multiword=0.57,
        pathological=True,
    )
    after = _language_report(
        nll=3.60,
        repetition=0.50,
        similarity=0.19,
        multiword=0.57,
        pathological=True,
    )

    assert _language_quality(after) > _language_quality(before)
    accepted, report = _segment_language_gate(before, after, anchor)

    assert accepted is True
    assert report["recovery_mode"] is True
    assert report["after_quality"] > report["before_quality"]


def test_segment_language_guard_accepts_small_monotonic_gain_for_31k_rescue():
    anchor = _language_report(
        nll=2.85,
        repetition=0.44,
        similarity=0.12,
        multiword=0.43,
        pathological=False,
    )
    before = _language_report(
        nll=3.75,
        repetition=0.53,
        similarity=0.18,
        multiword=0.57,
        pathological=True,
    )
    after = _language_report(
        nll=3.747,
        repetition=0.53,
        similarity=0.18,
        multiword=0.57,
        pathological=True,
    )

    gain = _language_quality(after) - _language_quality(before)
    assert 0.002 < gain < 0.005

    accepted_small, small_report = _segment_language_gate(
        before,
        after,
        anchor,
        attempted_tokens=32_512,
    )
    accepted_large, large_report = _segment_language_gate(
        before,
        after,
        anchor,
        attempted_tokens=65_024,
    )

    assert accepted_small is True
    assert small_report["recovery_minimum_quality_delta"] == pytest.approx(0.002)
    assert accepted_large is False
    assert large_report["recovery_minimum_quality_delta"] == pytest.approx(0.005)


def test_segment_language_guard_rejects_nonrecovering_retry_below_anchor():
    anchor = _language_report(
        nll=2.85,
        repetition=0.44,
        similarity=0.12,
        multiword=0.43,
        pathological=False,
    )
    before = _language_report(
        nll=3.75,
        repetition=0.53,
        similarity=0.18,
        multiword=0.57,
        pathological=True,
    )
    after = _language_report(
        nll=3.76,
        repetition=0.53,
        similarity=0.18,
        multiword=0.57,
        pathological=True,
    )

    accepted, report = _segment_language_gate(before, after, anchor)

    assert accepted is False
    assert report["recovery_mode"] is True
    assert any(
        "did not measurably recover" in reason
        for reason in report["local_reasons"]
    )


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


def test_phase5_cached_greedy_trace_matches_uncached_reference():
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
    runtime = GeneralistRuntime(
        CausalTransformerLM(cfg),
        cfg,
        tokenizer=ByteTokenizer(),
        device="cpu",
    )

    uncached = _single_greedy_trace(
        runtime,
        "Hello",
        max_new_tokens=10,
        use_cache=False,
    )
    cached = _single_greedy_trace(
        runtime,
        "Hello",
        max_new_tokens=10,
        use_cache=True,
    )

    assert cached["generated_token_ids"] == uncached["generated_token_ids"]
    assert cached["raw_output"] == uncached["raw_output"]
    assert cached["longest_repeated_token_run"] == uncached["longest_repeated_token_run"]
    assert cached["repetition_rate"] == pytest.approx(uncached["repetition_rate"])
    assert cached["token_entropy"] == pytest.approx(uncached["token_entropy"], abs=1e-7)
    assert cached["top1_probability"] == pytest.approx(
        uncached["top1_probability"], abs=1e-7
    )
    assert cached["top5_probability_mass"] == pytest.approx(
        uncached["top5_probability_mass"], abs=1e-7
    )


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
