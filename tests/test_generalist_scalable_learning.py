from __future__ import annotations

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


def test_progressive_weight_inheritance_copies_overlap_and_embeddings():
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
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=128,
        dropout=0.0,
    ).validate()
    source = CausalTransformerLM(source_cfg)
    target = CausalTransformerLM(target_cfg)
    with torch.no_grad():
        source.token_embedding.weight.fill_(0.125)
        source.blocks[0].attn.qkv.weight.fill_(0.25)

    report = _transfer_compatible_weights(
        source,
        target,
        source_tokenizer=ByteTokenizer(),
        target_tokenizer=ByteTokenizer(),
    )
    assert report["embedding_width_migrated"] is True
    assert report["partial_prefix_tensors"]
    assert torch.allclose(
        target.token_embedding.weight[:, :32],
        source.token_embedding.weight,
    )
    assert torch.allclose(
        target.blocks[0].attn.qkv.weight[:96, :32],
        source.blocks[0].attn.qkv.weight,
    )


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
