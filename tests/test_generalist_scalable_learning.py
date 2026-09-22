from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "computer"))

from generalist_lm.distillation import DistillationPrompt, distill_prompts
from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
from generalist_lm.pretraining import (
    load_local_corpus,
    pack_causal_blocks,
    pretrain_causal,
)
from generalist_lm.tokenizer import ByteTokenizer
from generalist_lm.transformers_lora import LoRATrainConfig, train_local_lora


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


def test_local_corpus_loader_bounds_deduplicates_and_blocks_escape(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha beta gamma " * 20, encoding="utf-8")
    (corpus / "b.md").write_text("alpha beta gamma " * 20, encoding="utf-8")
    (corpus / "skip.bin").write_bytes(b"\x00\x01\x02")

    report = load_local_corpus(
        [corpus],
        allowed_roots=[tmp_path],
        max_file_bytes=50_000,
        max_total_bytes=100_000,
    )
    assert len(report.documents) == 1
    assert any(row["reason"] == "duplicate" for row in report.skipped)
    assert any(row["reason"] == "unsupported_suffix" for row in report.skipped)
    assert report.total_bytes > 0

    outside = tmp_path.parent / "outside-corpus.txt"
    with pytest.raises(PermissionError):
        load_local_corpus([outside], allowed_roots=[tmp_path])


def test_local_corpus_loader_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "airi-outside-corpus.txt"
    outside.write_text("secret external corpus text", encoding="utf-8")
    try:
        link = tmp_path / "linked.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        with pytest.raises(PermissionError, match="escapes allowed roots"):
            load_local_corpus([link], allowed_roots=[tmp_path])
    finally:
        outside.unlink(missing_ok=True)


def test_causal_pretraining_reduces_loss_on_reviewed_corpus(tmp_path: Path):
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "tiny.txt").write_text(
        ("hello world. orange systems learn from text. " * 30),
        encoding="utf-8",
    )
    report = load_local_corpus([corpus], allowed_roots=[tmp_path])
    tok = ByteTokenizer()
    blocks = pack_causal_blocks(
        report.documents,
        tok,
        context_length=tiny_config().context_length,
    )
    assert blocks

    model = CausalTransformerLM(tiny_config())
    training = pretrain_causal(
        model,
        tok,
        report.documents,
        steps=35,
        batch_size=2,
        learning_rate=8e-3,
        weight_decay=0.0,
        seed=31,
    )
    assert training["ok"] is True
    assert training["final_loss"] < training["initial_loss"]
    assert training["loss_improvement"] > 0
    assert training["corpus_bytes"] == report.total_bytes


class Teacher:
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        del max_new_tokens
        if "add" in prompt.lower():
            return "def add(a, b):\n    return a + b"
        return "ORANGE"


def test_distillation_produces_reviewable_sft_rows_without_trusting_them():
    examples, report = distill_prompts(
        Teacher(),
        [
            DistillationPrompt("language", "Reply ORANGE"),
            DistillationPrompt("coding", "Write add(a,b)"),
            DistillationPrompt("coding", "Write add(a,b)"),
            DistillationPrompt("unknown", "bad"),
        ],
    )
    assert len(examples) == 2
    assert report["accepted"] == 2
    assert "does not bypass held-out qualification" in report["policy"]
    statuses = [row["status"] for row in report["rows"]]
    assert "duplicate" in statuses
    assert "rejected_domain" in statuses
    assert examples[0].messages[-1]["role"] == "assistant"


def test_lora_configuration_is_bounded_and_base_model_must_be_local(tmp_path: Path):
    assert LoRATrainConfig().validate().rank == 8
    with pytest.raises(ValueError):
        LoRATrainConfig(rank=9999).validate()
    with pytest.raises(FileNotFoundError):
        train_local_lora(
            tmp_path / "missing-model",
            [],
            tmp_path / "out",
        )


def test_lora_never_overwrites_base_model_even_before_optional_imports(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    from generalist_lm.training import SFTExample

    examples = [
        SFTExample([
            {"role": "user", "content": "Reply hi"},
            {"role": "assistant", "content": "hi"},
        ])
    ]
    with pytest.raises(ValueError, match="must not overwrite"):
        train_local_lora(base, examples, base / "adapter")


def test_cli_exposes_scalable_learning_commands(tmp_path: Path):
    from generalist_lm.cli import parser

    p = parser()
    args = p.parse_args([
        "pretrain-native",
        str(tmp_path / "checkpoint"),
        str(tmp_path / "corpus"),
        "--allowed-root", str(tmp_path),
        "--output", str(tmp_path / "out"),
    ])
    assert args.cmd == "pretrain-native"

    args = p.parse_args([
        "distill-transformers",
        str(tmp_path / "teacher"),
        str(tmp_path / "prompts.jsonl"),
        str(tmp_path / "distilled.jsonl"),
    ])
    assert args.cmd == "distill-transformers"

    args = p.parse_args([
        "lora-transformers",
        str(tmp_path / "base"),
        str(tmp_path / "sft.jsonl"),
        str(tmp_path / "adapter"),
    ])
    assert args.cmd == "lora-transformers"



class _FakeResponse:
    def __init__(self, blob: bytes, url: str):
        self._blob = blob
        self._url = url
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._blob) - self._offset
        chunk = self._blob[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        return None


def test_progressive_scaling_builds_larger_bounded_generalist():
    from generalist_lm.evolution import (
        GeneralistGenome,
        progressive_scale_candidate,
        progressive_scale_target,
    )
    from generalist_lm.model import estimate_parameter_count

    champion = GeneralistGenome(
        generation=3,
        parent_id="p",
        genome_id="g",
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    current = estimate_parameter_count(champion.model_config())
    target = progressive_scale_target(current, max_parameters=2_000_000)
    assert target == 250_000

    candidate = progressive_scale_candidate(
        champion,
        target_parameters=target,
        max_width=256,
        max_layers=6,
    )
    params = estimate_parameter_count(candidate.model_config())
    assert params > current
    assert params <= 2_000_000
    assert candidate.parent_id == champion.genome_id
    assert candidate.generation == champion.generation + 1
    assert 1e-4 <= candidate.learning_rate <= 1e-3
    assert candidate.learning_rate < champion.learning_rate


def test_adaptive_curriculum_weights_weak_domains_more_heavily():
    from collections import Counter

    from generalist_lm.evolution import GeneralistGenome
    from generalist_lm.research_cycle import (
        _training_rows_for_genome,
        adaptive_domain_weights,
    )

    report = {
        "domain_nll_per_byte": {
            "language": 5.0,
            "coding": 2.0,
            "data": 2.5,
            "reasoning": 3.0,
            "tools": 2.2,
            "structured": 2.1,
        }
    }
    weights = adaptive_domain_weights(report)
    assert weights["language"] == pytest.approx(3.0)
    assert weights["coding"] == pytest.approx(1.0)

    genome = GeneralistGenome(
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    rows = _training_rows_for_genome(
        genome,
        domain_weights=weights,
    )
    counts = Counter(row.domain for row in rows)
    assert counts["language"] > counts["coding"]


def test_progressive_weight_inheritance_preserves_same_width_function():
    torch = pytest.importorskip("torch")
    from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig
    from generalist_lm.research_cycle import _transfer_compatible_weights
    from generalist_lm.tokenizer import ByteTokenizer

    source_cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()
    target_cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=96,
        dropout=0.0,
    ).validate()
    torch.manual_seed(123)
    source = CausalTransformerLM(source_cfg)
    torch.manual_seed(456)
    target = CausalTransformerLM(target_cfg)

    report = _transfer_compatible_weights(
        source,
        target,
        source_tokenizer=ByteTokenizer(),
        target_tokenizer=ByteTokenizer(),
    )
    assert report["function_preserving_growth"] is True
    assert report["embedding_width_migrated"] is False
    assert any(
        name == "blocks.0.ff.up.weight"
        for name in report["partial_prefix_tensors"]
    )
    assert "blocks.1.attn.out.weight" in report["identity_initialized_tensors"]
    assert "blocks.1.ff.down.weight" in report["identity_initialized_tensors"]

    ids = torch.tensor(
        [[1, 40, 41, 42, 43, 44, 45, 46]],
        dtype=torch.long,
    )
    source.eval()
    target.eval()
    with torch.no_grad():
        source_logits = source(ids)["logits"]
        target_logits = target(ids)["logits"]
    assert torch.allclose(source_logits, target_logits, atol=1e-6, rtol=1e-6)


def test_generalist_data_growth_admits_only_permissive_immutable_text(tmp_path: Path):
    import json
    from urllib.error import HTTPError

    from generalist_lm.generalist_data_growth import grow_generalist_data

    commit = "a" * 40
    payloads = {
        "search": {
            "items": [{"full_name": "example/corpus"}],
        },
        "repo": {
            "default_branch": "main",
            "license": {"spdx_id": "MIT"},
        },
        "branch": {
            "commit": {"sha": commit},
        },
        "tree": {
            "tree": [
                {
                    "type": "blob",
                    "path": "data/sample.txt",
                    "size": 1500,
                }
            ]
        },
    }
    raw_text = (
        "This is a permissively licensed language corpus example with enough "
        "ordinary text to pass the bounded AIRI quality filter. "
    ).encode("utf-8") * 16

    def opener(request, timeout):
        del timeout
        url = request.full_url
        if "search/repositories" in url:
            return _FakeResponse(json.dumps(payloads["search"]).encode(), url)
        if url.endswith("/repos/example/corpus"):
            return _FakeResponse(json.dumps(payloads["repo"]).encode(), url)
        if "/branches/main" in url:
            return _FakeResponse(json.dumps(payloads["branch"]).encode(), url)
        if "/git/trees/" in url:
            return _FakeResponse(json.dumps(payloads["tree"]).encode(), url)
        if url.startswith("https://raw.githubusercontent.com/"):
            return _FakeResponse(raw_text, url)
        raise AssertionError(url)

    result = grow_generalist_data(
        tmp_path,
        signals=["language_gap"],
        max_new_bytes=100_000,
        max_total_bytes=200_000,
        max_repositories=1,
        max_files_per_repo=2,
        opener=opener,
    )
    assert result["ok"] is True
    assert result["added_files"] == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    row = manifest["files"][0]
    assert row["spdx"] == "MIT"
    assert row["commit"] == commit
    assert row["source_url"].startswith(
        "https://raw.githubusercontent.com/example/corpus/"
        + commit
    )
    assert (tmp_path / "approved").is_dir()
    assert list((tmp_path / "approved").iterdir())


def test_generalist_data_growth_rejects_unknown_license(tmp_path: Path):
    import json

    from generalist_lm.generalist_data_growth import grow_generalist_data

    def opener(request, timeout):
        del timeout
        url = request.full_url
        if "search/repositories" in url:
            return _FakeResponse(
                json.dumps({"items": [{"full_name": "example/closed"}]}).encode(),
                url,
            )
        if url.endswith("/repos/example/closed"):
            return _FakeResponse(
                json.dumps({
                    "default_branch": "main",
                    "license": {"spdx_id": "NOASSERTION"},
                }).encode(),
                url,
            )
        raise AssertionError(url)

    result = grow_generalist_data(
        tmp_path,
        max_new_bytes=100_000,
        max_total_bytes=100_000,
        max_repositories=1,
        opener=opener,
    )
    assert result["added_files"] == 0
    assert any("license" in row["reason"] for row in result["rejected"])



def test_generalist_swarm_reducer_prefers_safe_generation_and_nll(tmp_path: Path):
    import json

    from generalist_lm.generalist_swarm import select_survivors

    root = tmp_path / "results"
    root.mkdir()
    rows = [
        {
            "index": 0,
            "eligible": False,
            "generation": 0.0,
            "nll": 4.0,
            "regression": 0.01,
            "params": 100_000,
        },
        {
            "index": 1,
            "eligible": True,
            "generation": 0.0,
            "nll": 4.2,
            "regression": 0.01,
            "params": 120_000,
        },
        {
            "index": 2,
            "eligible": False,
            "generation": 0.2,
            "nll": 4.5,
            "regression": 0.02,
            "params": 150_000,
        },
        {
            "index": 3,
            "eligible": True,
            "generation": 0.1,
            "nll": 4.1,
            "regression": 0.50,
            "params": 110_000,
        },
    ]
    for row in rows:
        folder = root / str(row["index"])
        folder.mkdir()
        payload = {
            "ok": True,
            "version": "airi-generalist-free-speed-v4",
            "cycle": 1,
            "stage": 1,
            "steps": 3,
            "candidate_index": row["index"],
            "candidate_id": f"g-{row['index']}",
            "kind": "architecture",
            "genome": {},
            "reports": [],
            "all_seed_eligible": row["eligible"],
            "any_seed_eligible": row["eligible"],
            "mean_nll_per_byte": row["nll"],
            "best_nll_per_byte": row["nll"],
            "mean_generation_accuracy": row["generation"],
            "worst_domain_regression": row["regression"],
            "parameters": row["params"],
            "score": 10.0,
            "corpus": {},
            "checkpoint_dir": "best-checkpoint",
            "external_pretrained": False,
        }
        (folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "selection.json"
    result = select_survivors([root], output, survivors=2)
    selected = [row["index"] for row in result["selected"]]
    assert 3 not in selected  # catastrophic domain regression is cut early
    assert 1 in selected      # fully eligible candidate survives
    assert len(selected) == 2


def test_generalist_swarm_plan_contains_progressive_scale_and_adaptive_policy(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from generalist_lm.evolution import GeneralistGenome
    from generalist_lm.runtime import GeneralistRuntime
    from generalist_lm.generalist_swarm import prepare_swarm
    from generalist_lm.model import GeneralistLMConfig

    state = tmp_path / "state"
    champion = state / "champion"
    state.mkdir()
    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()
    GeneralistRuntime.fresh(cfg).save_checkpoint(
        champion,
        metadata={"role": "research_champion", "production_qualified": False},
    )
    genome = GeneralistGenome(
        generation=1,
        parent_id="seed",
        genome_id="tiny-champion",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    (state / "champion-genome.json").write_text(
        json.dumps(genome.to_dict()),
        encoding="utf-8",
    )

    plan = prepare_swarm(
        state,
        tmp_path / "plan.json",
        population_size=4,
        max_params=2_000_000,
        max_context=512,
        max_width=256,
        max_layers=6,
        grow_data=False,
    )
    assert plan["ok"] is True
    assert len(plan["candidates"]) == 4
    assert plan["policy"]["parallel_candidates"] is True
    assert plan["policy"]["adaptive_curriculum"] is True
    assert plan["progressive_scaling"]["target_parameters"] == 250_000
    assert plan["progressive_scaling"]["candidate_generated"] is True
    assert any(row["kind"] == "progressive_scale" for row in plan["candidates"])



def test_generalist_swarm_finalize_handles_no_eligible_finalist(tmp_path: Path):
    pytest.importorskip("torch")
    import json

    from generalist_lm.evolution import GeneralistGenome
    from generalist_lm.generalist_swarm import finalize_swarm
    from generalist_lm.model import GeneralistLMConfig
    from generalist_lm.runtime import GeneralistRuntime

    state = tmp_path / "state"
    champion_dir = state / "champion"
    state.mkdir()

    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()
    GeneralistRuntime.fresh(cfg).save_checkpoint(
        champion_dir,
        metadata={
            "role": "research_champion",
            "production_qualified": False,
        },
    )
    genome = GeneralistGenome(
        generation=1,
        parent_id="seed",
        genome_id="no-finalist-champion",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    (state / "champion-genome.json").write_text(
        json.dumps(genome.to_dict()),
        encoding="utf-8",
    )

    plan = {
        "cycle": 1,
        "champion": genome.to_dict(),
        "champion_report": {},
        "signals": [],
        "mathesis": None,
        "curriculum": {},
        "domain_weights": {},
        "data_growth": {"ok": True, "skipped": True},
        "progressive_scaling": {},
        "plateau": {},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    empty_results = tmp_path / "results"
    empty_results.mkdir()

    result = finalize_swarm(
        state,
        plan_path,
        [empty_results],
        tmp_path / "final.json",
    )
    assert result["ok"] is True
    assert result["promoted"] is False
    assert result["winner"] is None
    assert result["champion"]["genome_id"] == genome.genome_id
    assert "no finalist" in result["promotion_reason"]
    assert result["policy"]["candidate_can_self_promote"] is False



def test_generalist_data_growth_skips_oversized_repository_tree(tmp_path: Path):
    import json

    from generalist_lm.generalist_data_growth import grow_generalist_data

    big_commit = "b" * 40
    good_commit = "c" * 40
    raw_text = (
        "A compact permissive corpus can teach syntax, vocabulary, structure, and "
        "reasoning without importing pretrained model weights. The sample describes "
        "gardens, rivers, libraries, algorithms, weather, music, history, geometry, "
        "machines, languages, experiments, evidence, hypotheses, tables, programs, "
        "stories, questions, answers, causes, effects, comparisons, and explanations. "
        "Each sentence contributes different semantic material so the quarantine can "
        "distinguish useful prose from boilerplate repetition while keeping the corpus "
        "small enough for bounded autonomous pretraining."
    ).encode("utf-8")

    def opener(request, timeout):
        del timeout
        url = request.full_url
        if "search/repositories" in url:
            return _FakeResponse(
                json.dumps({
                    "items": [
                        {"full_name": "example/huge"},
                        {"full_name": "example/good"},
                    ]
                }).encode(),
                url,
            )
        if url.endswith("/repos/example/huge"):
            return _FakeResponse(
                json.dumps({
                    "default_branch": "main",
                    "license": {"spdx_id": "MIT"},
                }).encode(),
                url,
            )
        if url.endswith("/repos/example/good"):
            return _FakeResponse(
                json.dumps({
                    "default_branch": "main",
                    "license": {"spdx_id": "Apache-2.0"},
                }).encode(),
                url,
            )
        if "/repos/example/huge/branches/main" in url:
            return _FakeResponse(
                json.dumps({"commit": {"sha": big_commit}}).encode(),
                url,
            )
        if "/repos/example/good/branches/main" in url:
            return _FakeResponse(
                json.dumps({"commit": {"sha": good_commit}}).encode(),
                url,
            )
        if f"/repos/example/huge/git/trees/{big_commit}" in url:
            raise ValueError("GitHub metadata response exceeded budget:16000000")
        if f"/repos/example/good/git/trees/{good_commit}" in url:
            return _FakeResponse(
                json.dumps({
                    "truncated": False,
                    "tree": [{
                        "type": "blob",
                        "path": "corpus/sample.txt",
                        "size": len(raw_text),
                    }],
                }).encode(),
                url,
            )
        if url.startswith(
            f"https://raw.githubusercontent.com/example/good/{good_commit}/"
        ):
            return _FakeResponse(raw_text, url)
        raise AssertionError(url)

    result = grow_generalist_data(
        tmp_path,
        max_new_bytes=100_000,
        max_total_bytes=200_000,
        max_repositories=2,
        max_files_per_repo=2,
        opener=opener,
    )
    assert result["ok"] is True
    assert result["added_files"] == 1
    assert result["repositories_used"] == 1
    assert any(
        row["repo"] == "example/huge"
        and row["reason"] == "metadata_tree:ValueError"
        for row in result["rejected"]
    )



def test_generalist_cumulative_stage_source_is_identity_bound(tmp_path: Path):
    pytest.importorskip("torch")
    import json

    from generalist_lm.evolution import GeneralistGenome
    from generalist_lm.generalist_swarm import _load_stage_source
    from generalist_lm.model import GeneralistLMConfig
    from generalist_lm.runtime import GeneralistRuntime

    genome = GeneralistGenome(
        generation=3,
        parent_id="parent",
        genome_id="candidate-cumulative",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    runtime = GeneralistRuntime.fresh(
        GeneralistLMConfig(
            vocab_size=264,
            context_length=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
        ).validate()
    )
    checkpoint = tmp_path / "checkpoint"
    runtime.save_checkpoint(
        checkpoint,
        metadata={
            "role": "free_speed_candidate",
            "candidate_id": genome.genome_id,
            "cycle": 70,
            "stage": 1,
            "cumulative_steps": 3,
            "production_qualified": False,
        },
    )

    loaded, metadata = _load_stage_source(
        checkpoint,
        genome=genome,
        cycle=70,
        stage=2,
    )
    assert loaded.config.to_dict() == runtime.config.to_dict()
    assert metadata["cumulative_steps"] == 3

    with pytest.raises(ValueError, match="previous stage"):
        _load_stage_source(
            checkpoint,
            genome=genome,
            cycle=70,
            stage=3,
        )

    wrong = GeneralistGenome(
        **{
            **genome.to_dict(),
            "genome_id": "different-candidate",
        }
    ).validate()
    with pytest.raises(ValueError, match="candidate mismatch"):
        _load_stage_source(
            checkpoint,
            genome=wrong,
            cycle=70,
            stage=2,
        )



def test_generalist_cumulative_stage1_bootstraps_without_source_checkpoint(
    tmp_path: Path,
):
    pytest.importorskip("torch")
    import json

    from generalist_lm.evolution import GeneralistGenome
    from generalist_lm.generalist_swarm import run_candidate
    from generalist_lm.model import GeneralistLMConfig
    from generalist_lm.runtime import GeneralistRuntime

    state = tmp_path / "state"
    champion_dir = state / "champion"
    state.mkdir()

    cfg = GeneralistLMConfig(
        vocab_size=264,
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    ).validate()
    runtime = GeneralistRuntime.fresh(cfg)
    runtime.save_checkpoint(
        champion_dir,
        metadata={
            "role": "research_champion",
            "production_qualified": False,
        },
    )
    genome = GeneralistGenome(
        generation=1,
        parent_id="seed",
        genome_id="stage1-bootstrap",
        context_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    (state / "champion-genome.json").write_text(
        json.dumps(genome.to_dict()),
        encoding="utf-8",
    )

    plan = {
        "cycle": 1,
        "champion": genome.to_dict(),
        "champion_report": {
            "loss": 100.0,
            "nll_per_byte": 100.0,
            "domain_nll_per_byte": {},
        },
        "domain_weights": {},
        "limits": {
            "max_params": 2_000_000,
            "max_context": 512,
            "max_width": 256,
            "max_layers": 6,
        },
        "candidates": [{
            "index": 0,
            "kind": "continual",
            "candidate_id": genome.genome_id,
            "genome": genome.to_dict(),
        }],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    output = tmp_path / "out"
    result = run_candidate(
        plan_path,
        state,
        tmp_path,
        output,
        candidate_index=0,
        stage=1,
        steps=1,
        repeat_seeds=1,
        pretrain_steps=0,
        max_repo_bytes=64_000,
        max_external_bytes=0,
        source_checkpoint=None,
    )
    assert result["ok"] is True
    assert result["continued_from_stage"] is None
    assert result["cumulative_steps"] == 1

    metadata = json.loads(
        (output / "best-checkpoint" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["previous_stage"] is None
    assert metadata["cumulative_steps"] == 1


def test_progressive_scaling_can_prefer_function_preserving_growth():
    from generalist_lm.evolution import GeneralistGenome, progressive_scale_candidate
    from generalist_lm.model import estimate_parameter_count

    champion = GeneralistGenome(
        generation=3,
        parent_id="p",
        genome_id="preserve-source",
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        retrieval_adapter=False,
        symbolic_adapter=False,
        code_adapter=False,
        data_adapter=False,
        reasoning_depth=1,
    ).validate()
    current = estimate_parameter_count(champion.model_config())
    candidate = progressive_scale_candidate(
        champion,
        target_parameters=250_000,
        max_width=256,
        max_layers=6,
        prefer_function_preserving=True,
    )
    assert estimate_parameter_count(candidate.model_config()) > current
    assert candidate.d_model == champion.d_model
    assert candidate.n_layers >= champion.n_layers
    assert candidate.d_ff >= champion.d_ff


def test_generalist_swarm_reducer_reserves_one_safe_scale_probe(tmp_path: Path):
    from generalist_lm.generalist_swarm import select_survivors

    root = tmp_path / "scale-results"
    root.mkdir()
    rows = [
        (0, "architecture", 3.0),
        (1, "architecture", 3.1),
        (2, "progressive_scale", 3.4),
    ]
    for index, kind, nll in rows:
        folder = root / str(index)
        folder.mkdir()
        payload = {
            "ok": True,
            "version": "airi-generalist-free-speed-v4",
            "cycle": 1,
            "stage": 1,
            "steps": 3,
            "candidate_index": index,
            "candidate_id": f"scale-{index}",
            "kind": kind,
            "genome": {},
            "reports": [],
            "all_seed_eligible": True,
            "any_seed_eligible": True,
            "mean_nll_per_byte": nll,
            "best_nll_per_byte": nll,
            "mean_generation_accuracy": 0.0,
            "worst_domain_regression": 0.02,
            "parameters": 100_000 + index * 50_000,
            "score": 10.0,
            "corpus": {},
            "checkpoint_dir": "best-checkpoint",
            "external_pretrained": False,
        }
        (folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = select_survivors([root], tmp_path / "selected.json", survivors=2)
    selected = [row["index"] for row in result["selected"]]
    assert 0 in selected
    assert 2 in selected
    assert result["protected_progressive_scale"] is True


def test_generalist_data_growth_prioritizes_useful_domains_and_skips_quotes(tmp_path: Path):
    from generalist_lm.generalist_data_growth import grow_generalist_data

    commit = "d" * 40
    raw_by_path = {
        "quotes/tiny.yaml": ("quote: tiny motivational phrase\n" * 40).encode(),
        "math/proofs.txt": (
            "A theorem starts from explicit assumptions and derives a conclusion. "
            "An algebraic proof preserves equality when the same operation is applied "
            "to both sides. Contradiction assumes the negation and derives an impossibility. "
            "Induction proves a base case and then a general successor step. "
            "Geometry can connect parallel lines, angles, distance, area, and similarity. "
            "Number theory studies divisibility, primes, congruences, and integer structure. "
            "A counterexample is sufficient to disprove a universal proposition. "
            "Logical implication differs from equivalence because the reverse direction "
            "requires its own argument. Definitions constrain every later inference, while "
            "lemmas isolate reusable intermediate results. A rigorous proof records the "
            "chain of justified transformations rather than relying on an unexplained answer."
        ).encode(),
        "datasets/table.csv": (
            "name,value,category\nalpha,1,a\nbeta,2,b\ngamma,3,c\n" * 30
        ).encode(),
        "src/algorithm.py": (
            "def binary_search(items, target):\n    return target in items\n" * 30
        ).encode(),
    }

    def opener(request, timeout):
        del timeout
        url = request.full_url
        if "search/repositories" in url:
            return _FakeResponse(
                json.dumps({"items": [{"full_name": "example/balanced"}]}).encode(),
                url,
            )
        if url.endswith("/repos/example/balanced"):
            return _FakeResponse(
                json.dumps({
                    "default_branch": "main",
                    "license": {"spdx_id": "MIT"},
                }).encode(),
                url,
            )
        if "/branches/main" in url:
            return _FakeResponse(json.dumps({"commit": {"sha": commit}}).encode(), url)
        if "/git/trees/" in url:
            return _FakeResponse(
                json.dumps({
                    "truncated": False,
                    "tree": [
                        {"type": "blob", "path": path, "size": len(raw)}
                        for path, raw in raw_by_path.items()
                    ],
                }).encode(),
                url,
            )
        prefix = f"https://raw.githubusercontent.com/example/balanced/{commit}/"
        if url.startswith(prefix):
            path = url[len(prefix):]
            return _FakeResponse(raw_by_path[path], url)
        raise AssertionError(url)

    result = grow_generalist_data(
        tmp_path,
        signals=["reasoning_gap", "data_gap", "coding_gap"],
        max_new_bytes=500_000,
        max_total_bytes=500_000,
        max_repositories=1,
        max_files_per_repo=4,
        opener=opener,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    paths = {row["path"] for row in manifest["files"]}
    assert "quotes/tiny.yaml" not in paths
    assert "math/proofs.txt" in paths
    assert "datasets/table.csv" in paths
    assert "src/algorithm.py" in paths
    assert result["domain_files"]["reasoning"] >= 1
    assert result["domain_files"]["data"] >= 1
    assert result["domain_files"]["code"] >= 1
    assert result["desired_domains"][:3] == ["reasoning", "data", "code"]


def test_generalist_data_growth_discovery_is_license_first_and_quality_ranked():
    from generalist_lm.generalist_data_growth import (
        _candidate_queries,
        _domain_for,
        _search_repositories,
    )

    queries = _candidate_queries(["reasoning_gap", "data_gap"])
    assert queries
    assert all("license:" in query for query in queries)
    assert queries[0].startswith("reasoning dataset")
    assert any(query.startswith("csv dataset") for query in queries)
    assert _domain_for("proofs/chapter.tex") == "reasoning"

    seen = []

    def opener(request, timeout):
        del timeout
        seen.append(request.full_url)
        return _FakeResponse(
            json.dumps({"items": [{"full_name": "example/permissive"}]}).encode(),
            request.full_url,
        )

    rows = _search_repositories(
        ["reasoning dataset license:mit"],
        token=None,
        per_query=4,
        opener=opener,
    )
    assert rows[0]["full_name"] == "example/permissive"
    assert len(seen) == 1
    assert "license%3Amit" in seen[0]
    assert "size%3A%3C100000" in seen[0]
    assert "sort=stars" in seen[0]


def test_generalist_weaknesses_routes_language_gap():
    from generalist_lm.research_cycle import _weaknesses

    signals = _weaknesses({
        "domain_nll_per_byte": {
            "language": 5.0,
            "coding": 2.0,
            "data": 1.5,
            "reasoning": 1.0,
            "tools": 0.5,
            "structured": 0.25,
        }
    })
    assert signals[0] == "language_gap"
    assert "coding_gap" in signals


def test_generalist_weaknesses_routes_autoregressive_collapse():
    from generalist_lm.research_cycle import _weaknesses

    signals = _weaknesses({
        "domain_nll_per_byte": {
            "language": 2.0,
            "coding": 1.8,
            "data": 1.7,
        },
        "generation_pathological_repetition": True,
        "generation_repetition_rate": 0.81,
        "generation_longest_repeated_token_run": 14,
        "generation_unique_token_ratio": 0.18,
    })
    assert "language_collapse" in signals
    assert "autoregressive_collapse" in signals


def test_generalist_weaknesses_routes_canary_only_collapse():
    from generalist_lm.research_cycle import _weaknesses

    signals = _weaknesses({
        "domain_nll_per_byte": {
            "coding": 2.0,
            "data": 1.9,
            "reasoning": 1.8,
        },
        "generation_repetition_rate": 0.20,
        "generation_unique_token_ratio": 0.70,
        "canary": {
            "generation_pathological_repetition": True,
            "generation_repetition_rate": 0.88,
            "generation_longest_repeated_token_run": 28,
            "generation_unique_token_ratio": 0.11,
        },
    })
    assert "language_collapse" in signals
    assert "autoregressive_collapse" in signals


def test_generalist_weaknesses_does_not_invent_collapse_when_generation_is_healthy():
    from generalist_lm.research_cycle import _weaknesses

    signals = _weaknesses({
        "domain_nll_per_byte": {
            "language": 3.0,
            "coding": 2.5,
            "data": 2.0,
        },
        "generation_pathological_repetition": False,
        "generation_repetition_rate": 0.22,
        "generation_longest_repeated_token_run": 3,
        "generation_unique_token_ratio": 0.62,
    })
    assert "language_gap" in signals
    assert "language_collapse" not in signals
    assert "autoregressive_collapse" not in signals


def test_generalist_language_curriculum_has_compositional_targets():
    from generalist_lm.curriculum import train_rows, validation_rows

    train_targets = [
        row.messages[-1]["content"]
        for row in train_rows()
        if row.domain == "language"
    ]
    validation_targets = [
        row.messages[-1]["content"]
        for row in validation_rows()
        if row.domain == "language"
    ]
    assert any(" " in target and target.endswith(".") for target in train_targets)
    assert any(" " in target and target.endswith(".") for target in validation_targets)


def test_generalist_text_similarity_tracks_partial_generation():
    from generalist_lm.research_cycle import _text_similarity

    assert _text_similarity("orange", "orange") == pytest.approx(1.0)
    partial = _text_similarity("orang", "orange")
    unrelated = _text_similarity("111111", "orange")
    assert 0.0 < partial < 1.0
    assert partial > unrelated


def test_generalist_adaptive_pretraining_budget_scales_but_stays_bounded():
    from generalist_lm.generalist_swarm import _adaptive_pretrain_steps

    tiny = _adaptive_pretrain_steps(
        1,
        corpus_bytes=100_000,
        stage=1,
        scale_multiplier=1.0,
    )
    large = _adaptive_pretrain_steps(
        2,
        corpus_bytes=10_000_000,
        stage=3,
        scale_multiplier=1.44,
    )
    assert tiny == 1
    assert large > tiny
    assert large <= 24


def test_generalist_swarm_reducer_prefers_partial_generation_before_nll(tmp_path: Path):
    from generalist_lm.generalist_swarm import select_survivors

    root = tmp_path / "partial-generation-results"
    root.mkdir()
    cases = [
        (0, 0.40, 1.60),
        (1, 0.10, 1.40),
    ]
    for index, similarity, nll in cases:
        folder = root / str(index)
        folder.mkdir()
        payload = {
            "ok": True,
            "version": "airi-generalist-free-speed-v4",
            "cycle": 1,
            "stage": 1,
            "steps": 3,
            "candidate_index": index,
            "candidate_id": f"partial-{index}",
            "kind": "architecture",
            "genome": {},
            "reports": [],
            "all_seed_eligible": False,
            "any_seed_eligible": False,
            "mean_nll_per_byte": nll,
            "best_nll_per_byte": nll,
            "mean_generation_accuracy": 0.0,
            "mean_generation_similarity": similarity,
            "mean_generation_nonempty_rate": 1.0,
            "worst_domain_regression": 0.02,
            "parameters": 120_000,
            "score": 10.0,
            "corpus": {},
            "checkpoint_dir": "best-checkpoint",
            "external_pretrained": False,
        }
        (folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    result = select_survivors([root], tmp_path / "partial-selected.json", survivors=1)
    assert result["selected"][0]["index"] == 0


def test_generalist_external_corpus_preserves_manifest_domains(tmp_path: Path):
    from generalist_lm.generalist_swarm import _corpus_documents

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text(
        "A small repository document used for corpus classification. " * 4,
        encoding="utf-8",
    )

    state = tmp_path / "state"
    approved = state / "autodata" / "approved"
    approved.mkdir(parents=True)
    raw = ("name,value\nalpha,1\nbeta,2\n" * 30).encode("utf-8")
    digest = __import__("hashlib").sha256(raw).hexdigest()
    (approved / "dataset.csv").write_bytes(raw)
    (state / "autodata" / "manifest.json").write_text(
        json.dumps({
            "version": "generalist-data-growth-v3",
            "files": [{
                "sha256": digest,
                "domain": "data",
                "repo": "example/data",
                "path": "dataset.csv",
            }],
        }),
        encoding="utf-8",
    )

    documents, report = _corpus_documents(
        repo_root,
        state,
        max_repo_bytes=100_000,
        max_external_bytes=100_000,
    )
    external = [row for row in documents if row.sha256 == digest]
    assert len(external) == 1
    assert external[0].domain == "data"
    assert report["external"]["domain_documents"]["data"] == 1


def test_generalist_language_weakness_weights_general_and_language_autodata():
    from generalist_lm.research_cycle import _pretraining_domain_weights

    weights = _pretraining_domain_weights({
        "language": 3.0,
        "coding": 1.5,
        "data": 2.0,
        "reasoning": 2.25,
    })
    assert weights["general"] == pytest.approx(3.0)
    assert weights["language-it"] == pytest.approx(3.0)
    assert weights["code"] == pytest.approx(1.5)


def test_generalist_language_gap_expands_license_first_discovery():
    from generalist_lm.generalist_data_growth import _candidate_queries

    queries = _candidate_queries(["language_gap"])
    assert any(query.startswith("natural language corpus") for query in queries)
    assert any(query.startswith("italian corpus") for query in queries)
    assert all("license:" in query for query in queries)


def test_generalist_phase4_language_bridge_is_deterministic_and_grounded():
    from generalist_lm.language_bridge import build_language_bridge_rows
    from generalist_lm.pretraining import CorpusDocument

    docs = [
        CorpusDocument(
            source="approved:story.txt",
            text=(
                "The small robot studies language every morning. "
                "It reads clear examples and learns how sentences continue. "
                "After practice, the robot writes a short useful answer."
            ),
            sha256="a" * 64,
            bytes=180,
            domain="general",
        ),
        CorpusDocument(
            source="approved:data.csv",
            text="alpha,beta,gamma " * 20,
            sha256="b" * 64,
            bytes=200,
            domain="data",
        ),
    ]
    first = build_language_bridge_rows(docs, max_rows=8, seed=17)
    second = build_language_bridge_rows(docs, max_rows=8, seed=17)

    assert first
    assert [row.messages for row in first] == [row.messages for row in second]
    assert all(row.domain == "language" for row in first)
    assert all(row.messages[-1]["content"] for row in first)
    assert all("alpha,beta,gamma" not in row.messages[-1]["content"] for row in first)


def test_generalist_phase4_bpe_refreshes_only_until_target_vocab():
    from dataclasses import replace
    from types import SimpleNamespace

    from generalist_lm.bpe_tokenizer import train_bpe
    from generalist_lm.research_cycle import _tokenizer_for_genome, research_seed

    genome = replace(research_seed(), tokenizer_version="bpe-v1").validate()
    source = train_bpe(
        ["language learning improves with repeated natural sentences. " * 20],
        vocab_size=300,
        max_bytes=50_000,
    )
    runtime = SimpleNamespace(tokenizer=source)
    refreshed = _tokenizer_for_genome(
        genome,
        source_runtime=runtime,
        replay_rows=[],
        pretrain_documents=[],
        bpe_vocab_size=340,
        bpe_max_bytes=50_000,
    )
    assert refreshed is not source

    stable_runtime = SimpleNamespace(tokenizer=refreshed)
    stable = _tokenizer_for_genome(
        genome,
        source_runtime=stable_runtime,
        replay_rows=[],
        pretrain_documents=[],
        bpe_vocab_size=min(340, refreshed.vocab_size),
        bpe_max_bytes=50_000,
    )
    assert stable is refreshed


def test_generalist_phase4_bpe_extension_preserves_existing_token_ids():
    from generalist_lm.bpe_tokenizer import extend_bpe, train_bpe

    text = (
        "Natural language models learn repeated words and useful sentence patterns. "
        "Language improves when repeated patterns remain stable across training. "
    ) * 40
    source = train_bpe([text], vocab_size=300, max_bytes=40_000)
    extended = extend_bpe(
        source,
        [text],
        vocab_size=min(340, source.vocab_size + 40),
        max_bytes=40_000,
    )

    assert extended.merges[: len(source.merges)] == source.merges
    assert extended.vocab_size >= source.vocab_size
    for token_id in range(source.vocab_size):
        if token_id >= 8:
            assert extended.token_bytes(token_id) == source.token_bytes(token_id)


def test_airi_pc_lab_snapshot_and_curriculum_are_read_only():
    from pathlib import Path

    from generalist_lm.airi_pc_lab import (
        build_airi_pc_lab_rows,
        execute_lab_tool,
        lab_tools,
        snapshot_airi_pc_lab,
    )
    from generalist_lm.tool_protocol import ToolCall

    repo_root = Path(__file__).resolve().parents[1]
    snapshot = snapshot_airi_pc_lab(repo_root)

    assert snapshot["version"] == "airi-pc-lab-v1"
    assert snapshot["mode"] == "read_only_sandbox"
    assert snapshot["modules"]
    assert "filesystem_write" in snapshot["denied_capabilities"]
    assert "phone_control" in snapshot["denied_capabilities"]
    assert "production_promotion" in snapshot["denied_capabilities"]

    tools = lab_tools(snapshot)
    assert set(tools) == {
        "lab_list_capabilities",
        "lab_inspect_module",
        "lab_describe_task_flow",
    }
    rows = build_airi_pc_lab_rows(snapshot, max_rows=8)
    assert rows
    assert all(row.domain in {"tools", "coding"} for row in rows)
    assert all(row.messages[-1]["role"] == "assistant" for row in rows)
    assert all("shell" not in str(row.messages[-1]).lower() for row in rows)

    result = execute_lab_tool(
        snapshot,
        ToolCall(name="lab_list_capabilities", arguments={}),
    )
    assert result["mode"] == "read_only_sandbox"


def test_airi_pc_lab_rejects_non_allowlisted_module_and_tool():
    from pathlib import Path
    import pytest

    from generalist_lm.airi_pc_lab import execute_lab_tool, snapshot_airi_pc_lab
    from generalist_lm.tool_protocol import ToolCall

    repo_root = Path(__file__).resolve().parents[1]
    snapshot = snapshot_airi_pc_lab(repo_root)

    with pytest.raises(PermissionError):
        execute_lab_tool(
            snapshot,
            ToolCall(
                name="lab_inspect_module",
                arguments={"module": "../../secrets"},
            ),
        )

    with pytest.raises(PermissionError):
        execute_lab_tool(
            snapshot,
            ToolCall(name="shell", arguments={"command": "whoami"}),
        )


def test_curriculum_memory_accepts_verified_extra_lab_rows(tmp_path):
    from pathlib import Path

    from generalist_lm.airi_pc_lab import build_airi_pc_lab_rows, snapshot_airi_pc_lab
    from generalist_lm.curriculum_memory import CurriculumMemory

    repo_root = Path(__file__).resolve().parents[1]
    rows = build_airi_pc_lab_rows(snapshot_airi_pc_lab(repo_root), max_rows=4)
    memory = CurriculumMemory(tmp_path, max_rows=120)
    report = memory.expand(7, signals=["language_gap"], extra_rows=rows)

    assert report["extra_rows_considered"] == len(rows)
    stored = memory.rows()
    prompts = [row.messages[1]["content"] if len(row.messages) > 1 else "" for row in stored]
    assert any("AIRI-PC Lab" in prompt for prompt in prompts)


def test_airi_pc_lab_verified_experience_round_trip(tmp_path):
    from generalist_lm.airi_pc_lab import (
        load_verified_lab_experiences,
        record_verified_lab_experience,
    )

    probe = {
        "ok": True,
        "tool_call_valid": True,
        "tool": "lab_describe_task_flow",
        "arguments": {},
        "tool_result": {
            "flow": [
                "task_engine:start",
                "bounded_operation",
                "verification_engine",
                "judge",
                "experience_record",
            ],
            "mutation": False,
            "production_promotion": False,
        },
    }
    stored = record_verified_lab_experience(
        tmp_path,
        probe,
        cycle=9,
        source="champion",
    )
    assert stored["stored"] is True

    rows = load_verified_lab_experiences(tmp_path)
    assert len(rows) == 1
    assert rows[0].domain == "tools"
    assert rows[0].messages[-1]["role"] == "assistant"
    assert "Verified AIRI-PC Lab result" in rows[0].messages[-1]["content"]

    duplicate = record_verified_lab_experience(
        tmp_path,
        probe,
        cycle=9,
        source="champion",
    )
    assert duplicate["stored"] is False
    assert duplicate["reason"] == "duplicate"


def test_airi_pc_lab_does_not_store_failed_probe(tmp_path):
    from generalist_lm.airi_pc_lab import (
        load_verified_lab_experiences,
        record_verified_lab_experience,
    )

    result = record_verified_lab_experience(
        tmp_path,
        {
            "ok": False,
            "tool_call_valid": False,
            "error": "invalid tool JSON",
        },
        cycle=9,
        source="champion",
    )
    assert result["stored"] is False
    assert load_verified_lab_experiences(tmp_path) == []
