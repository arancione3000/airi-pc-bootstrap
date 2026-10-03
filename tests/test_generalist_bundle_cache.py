from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from generalist_lm import bootstrap_data as bd
from generalist_lm import bootstrap_bundle_cache as bc
from generalist_lm.tokenizer import ByteTokenizer
from generalist_lm.bpe_tokenizer import BPETokenizer


def prepared_bundle(tmp_path, monkeypatch, tokenizer=None):
    tokenizer = tokenizer or ByteTokenizer()
    def download(source, cache):
        raw = source['id'].encode()
        suffix = '.bz2' if source['id'].startswith('tatoeba') else '.gz'
        (cache / (source['id'] + suffix)).write_bytes(raw)
        meta = {'sha256': hashlib.sha256(raw).hexdigest(), 'download_bytes': len(raw)}
        (cache / (source['id'] + '.download.json')).write_text(json.dumps(meta))
        return raw, meta
    monkeypatch.setattr(bd, '_download', download)
    monkeypatch.setattr(bd, '_parse_tatoeba', lambda source, raw: [
        (str(i), f"A useful sentence from {source['id']} number {i} describes a quiet village.", None)
        for i in range(20)
    ])
    monkeypatch.setattr(bd, '_parse_oasst', lambda raw: {})
    monkeypatch.setattr(bd, '_oasst_conversations', lambda parsed: [
        (str(i), [{'role': 'user', 'content': f'Question {i}'},
                  {'role': 'assistant', 'content': f'A different conversation answer {i} about a cheerful dog.'}], 'en')
        for i in range(20)
    ])
    bundle = bd.build_bootstrap_bundle(tokenizer, cache_dir=tmp_path, target_tokens=100000)
    return tokenizer, bundle


def test_warm_build_reuses_identical_splits_without_selection_or_encoding(tmp_path, monkeypatch):
    tok, cold = prepared_bundle(tmp_path, monkeypatch)
    assert cold.cache_hit is False
    monkeypatch.setattr(bd, '_download', lambda *args: pytest.fail('download on warm path'))
    monkeypatch.setattr(bd, '_take_documents', lambda **kwargs: pytest.fail('selection/encoding on warm path'))
    warm = bd.build_bootstrap_bundle(tok, cache_dir=tmp_path,
        target_tokens=100000, previous_manifest=cold.manifest)
    assert warm.cache_hit is True
    assert warm.train_documents == cold.train_documents
    assert warm.validation_documents == cold.validation_documents
    assert warm.sft_train == cold.sft_train
    assert warm.sft_validation == cold.sft_validation
    assert warm.manifest == cold.manifest


@pytest.mark.parametrize('mutation', ['source', 'metadata', 'pins', 'code', 'payload', 'target'])
def test_bundle_invalidates_instead_of_reusing_changed_inputs(tmp_path, monkeypatch, mutation):
    tok, bundle = prepared_bundle(tmp_path, monkeypatch)
    previous = json.loads(json.dumps(bundle.manifest))
    target = 100000
    if mutation == 'source':
        (tmp_path / 'tatoeba-en-cc0.bz2').write_bytes(b'changed source')
    elif mutation == 'metadata':
        (tmp_path / 'tatoeba-en-cc0.download.json').write_text('{}')
    elif mutation == 'pins':
        previous['sources'][0]['sha256'] = 'different'
    elif mutation == 'code':
        original = bc._digest_file
        monkeypatch.setattr(bc, '_digest_file', lambda path:
            'different' if path.name == 'bootstrap_data.py' else original(path))
    elif mutation == 'payload':
        data, _ = bc._paths(tok, tmp_path, target)
        data.write_bytes(b'corrupt compressed cache')
    else:
        target += 1
    assert bc.load_reviewed_bundle(tok, tmp_path, target, previous) is None


def test_same_size_bpe_with_different_merges_cannot_reuse_bundle(tmp_path, monkeypatch):
    first = BPETokenizer(merges=[(8 + ord('a'), 8 + ord('b'))])
    second = BPETokenizer(merges=[(8 + ord('c'), 8 + ord('d'))])
    tok, bundle = prepared_bundle(tmp_path, monkeypatch, first)
    assert first.version == second.version and first.vocab_size == second.vocab_size
    assert bc.load_reviewed_bundle(second, tmp_path, 100000, bundle.manifest) is None


def test_cache_requires_existing_reviewed_manifest(tmp_path, monkeypatch):
    tok, bundle = prepared_bundle(tmp_path, monkeypatch)
    assert bc.load_reviewed_bundle(tok, tmp_path, 100000, None) is None
    assert bc.load_reviewed_bundle(tok, tmp_path, 100000, {}) is None


def test_bundle_rejects_changed_document_even_with_updated_file_digest(tmp_path, monkeypatch):
    tok, bundle = prepared_bundle(tmp_path, monkeypatch)
    data, index = bc._paths(tok, tmp_path, 100000)
    with gzip.open(data, 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    rows[1]['document']['text'] = 'Unexpected replacement'
    with gzip.open(data, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    meta = json.loads(index.read_text())
    meta['data_sha256'] = bc._digest_file(data)
    index.write_text(json.dumps(meta))
    assert bc.load_reviewed_bundle(tok, tmp_path, 100000, bundle.manifest) is None


def test_workflow_restores_and_saves_prepared_cache_separately():
    workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/generalist-bootstrap.yml').read_text()
    assert 'path: ${{ env.CACHE_DIR }}/prepared-bootstrap-v1' in workflow
    assert 'airi-generalist-phase5-bundle-v1-${{ runner.os }}-${{ env.TARGET_TOKENS }}-${{ github.run_id }}' in workflow
