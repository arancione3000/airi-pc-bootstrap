# Reviewed bootstrap bundle reuse — 2026-10-03

Goal: accelerate real accepted-token throughput without changing training data,
quality gates, optimizer updates, lineage or transaction sizes.

## Observed bottleneck

Inspected production run37127825887 on main c24c66f:

| Segment | Start UTC | Checkpoint commit UTC | Approximate duration |
|---|---|---|---|
| 1 | 13:56:38 | 14:09:24 | 12m46s |
| 2 | 14:10:49 | 14:22:12 | 11m23s |
| 3 | 14:22:13 | 14:33:29 | 11m16s |
| 4 | 14:34:52 | 14:46:15 | 11m23s |

The source and packed-block caches were restored successfully, yet every CLI
invocation rebuilt the entire bootstrap bundle: decompression, parsing,
selection, token-budget encoding and train/validation filtering. Training itself
usually takes30–70seconds. This identifies repeated preparation as a major
avoidable cost; it does not imply every remaining second is preparation.

## Implementation

Save the exact reviewed text/SFT splits and original manifest as compressed
JSONL. Reuse only when persisted manifest digest, source pins, source-file bytes,
exact tokenizer (including BPE merges), token quota and preparation-code hashes
match. Verify compressed payload digest, each document text digest/byte length,
manifest self-digest and split counts. Corrupt/stale/missing cache falls back to
normal preparation; no checkpoint, weights or optimizer is cached here.

Keep the original data builder as the cold path and preserve split ordering and
manifest identity. No new data, external model, relaxed guard or accepted-token
counter change. Add preparation elapsed seconds/cache-hit telemetry to progress.

Restore/save this directory through a separate Actions cache with a per-run key.
The old source-cache key is immutable on a primary hit and would otherwise
prevent publishing newly prepared bundles for future runs. First cold build
still pays preparation once; following segments/runs can reuse it.

## Validation and limits

278 passed /1 optional skip across cache, Phase5, learning-efficiency,
efficiency-engine and GeneralistLM suites. Compilation/diff checks passed.
Tests prove equality of all four splits and the manifest, skip downloads and
selection/encoding on the warm path, and deny reuse for changed source bytes,
metadata, pins, code, quota, BPE merges, corrupted payload or altered document.
The dedicated Generalist CI now runs these tests as well.

No new live speedup or100k/hour claim yet. Production data_preparation telemetry
must confirm warm hits and their duration; real throughput includes evaluation,
checkpoint uploads, queueing and runner setup. Large corpus decode/hash costs
remain. Gate rejections still reduce accepted throughput.
