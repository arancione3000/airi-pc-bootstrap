# AIRI Generalist learning-efficiency v3 — live candidate

Status: **draft / no live rollout yet**

This report supplements `reports/generalist-learning-efficiency-v3/REPORT.md` and the
continuation report. It records the candidate that emerged after the original
ablation study and the guardrails required before any live use.

## Baseline

The task baseline remains the dynamically pinned baseline from the start of the
learning-efficiency investigation. At that point accepted throughput was about
92.8k causal tokens/hour. The continuing legacy live trainer has remained near
~94–95k accepted tokens/hour with roughly 84% rollback and a ~4,064-token trust
region. These live numbers are telemetry, not a redefinition of the task
baseline.

The live 50M lineage remains close to the durable repetition ceiling:
repetition is about 0.23616 against an anchor around 0.15772 and the unchanged
hard allowance of +0.08.

## Candidate discovered offline

The successful candidate is `wide128_causal20_ar`.

Exact contract:

- effective causal batch: 128 examples;
- context: 128;
- nominal new supervised labels/update: 16,256;
- CPU-safe microbatching remains bounded (2 on the current ~50M model);
- AdamW protected optimizer, reset in trust-region recovery;
- existing discriminative optimizer LR multipliers preserved;
- base LR and 1/128 live trust-region LR scale preserved;
- 4-step segment warmup preserved;
- causal EOS loss weight: 1.60;
- causal teacher-forced anti-repetition weight: 0.05;
- physical causal replay: 20% of total supervised labels;
- replay CE weight: 0.25;
- replay reference KL weight: 0.50;
- reference checkpoint: immutable pre-segment lineage;
- negative-only autoregressive repetition objective: weight 0.10;
- autoregressive training prefixes: 8;
- no generated positive targets;
- no external teacher;
- no pretrained external weights;
- existing durable language/repetition gates are unchanged.

The important change is therefore not simply "batch 128". A bare wide batch
failed. The accepted candidate combines a wide accumulated causal update,
physical causal replay, the existing reference KL and a small negative-only
autoregressive repetition term.

## Existing evidence

Validated immutable-copy experiments have shown:

- 16,256-token screen: 3/3 independent seeds accepted;
- sustained 16,256-token sequence: 5/5 consecutive accepted updates;
- repetition remained about 0.2361575 across that sequence;
- held-out Phase-5 language NLL improved from about 2.02491 to 2.02151;
- recovered pinned validation subset improved from about 2.72453 to 2.72360;
- per-segment accepted-equivalent throughput was approximately 1.17M–1.44M
  tokens/hour;
- end-to-end equivalent throughput after evaluation was approximately
  0.64M–0.95M tokens/hour;
- 32k showed promise but was not stable and is not part of the live candidate.

The offline numbers are not claimed as live useful throughput until the
production shadow and live canary confirm them.

## Offline-to-production parity

Regression tests directly compare `wide128_causal20_ar` with the opt-in
production recovery plan. The test requires equality for:

- effective segment budget;
- effective batch size;
- replay format and fraction;
- replay CE weight;
- KL weight;
- anti-repetition weight;
- EOS weight;
- autoregressive objective weight;
- prefix count.

This catches silent drift between the benchmark and product implementation.

## Corpus reproducibility issue discovered by production shadow

The first production-shadow attempt failed before training because the current
rolling Italian Tatoeba export no longer matched the SHA-256 pinned by the live
manifest.

This is a correct fail-closed behavior and must not be bypassed.

The historical production source cache is not guaranteed to be available to a
new PR runner. For the shadow only, the workflow therefore recovers the
immutable benchmark corpus already used by the validated experiments (pinned
English Tatoeba plus immutable OASST1), injects only the data-loader result, and
then executes the real production `run_segment` optimizer/gate/rollback path.
The live replay files from the cloned state remain the protected replay source.

This shadow does **not** claim to reproduce the unavailable full historical
32M-token corpus. It exists to test offline-to-production training semantics.
No source digest check is weakened in production.

The rolling-source reproducibility problem should be solved separately by
persisting or otherwise immutably addressing future reviewed corpora.

## Live implementation safety

The production candidate is opt-in only via `--wide-batch-recovery`.
Automatic bootstrap runs retain the legacy behavior.

The workflow exposes a manual `wide_batch_recovery` switch and a
`max_segments` limit. During this canary phase:

- wide-batch requires `max_segments=1`;
- the result is validated locally before any state push;
- telemetry must prove batch 128;
- telemetry must prove causal replay;
- physical replay fraction must be within 0.18–0.22;
- autoregressive objective weight must equal 0.10;
- an accepted canary must contain at least 16k new supervised causal tokens;
- all existing language/repetition gates remain authoritative;
- a rejected canary retains normal rollback behavior;
- the workflow does not automatically dispatch another bootstrap after a
  manual wide-batch canary.

Thus a malformed candidate cannot be persisted merely because the Python
process exits successfully.

## Merge gate

Do not merge or enable the candidate until all of the following are true:

1. Generalist LM tests pass.
2. Airi-PC runtime verification passes.
3. MATHESIS verification passes.
4. Phase-5 checkpoint audit passes.
5. Offline/production parity tests pass.
6. Production shadow reaches the real `run_segment` training path and either:
   - is accepted by the unchanged language gate, or
   - is safely rejected with no causal-token persistence.
7. The equal-data paired comparison does not contradict the prior 16k evidence.

Only after these gates should a one-segment live canary be considered.

## First live canary

If the merge gate is satisfied, the first live run must be manually bounded to:

- `wide_batch_recovery=true`
- `max_segments=1`
- current 100M target
- unchanged durable anchor and language gate.

After the canary, compare against both the task baseline and the immediately
preceding live checkpoint:

- accepted/rejected result;
- causal tokens persisted;
- attempted and accepted wall-clock throughput;
- language NLL;
- repetition and durable-anchor headroom;
- multiword rate;
- unique-token ratio;
- pathological repetition;
- recovered/full validation when available;
- conversational probe;
- actual replay token fraction;
- checkpoint digest and lineage id.

Do not chain a second wide update until those values are reviewed.

## Rollback

No special rollback mechanism replaces the existing transactional guard.
A rejected segment keeps the pre-segment checkpoint and does not increment
persistent causal tokens. If a merged implementation itself must be reverted,
revert the feature commit(s) on `main`; do not manually edit
`generalist-state` or replace the single live lineage.
