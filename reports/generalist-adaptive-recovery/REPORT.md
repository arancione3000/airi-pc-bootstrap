# Protected recovery step adaptation — 2026-10-03

Main baseline: 86cff7a. Read-only state pin: a00bb69a98d9a6058ba485f5f38e6375a07bcd6d.
Checkpoint digest: 8d184426029ffb504caa7c1d61764268698b76e1deda43bcee079636ab140f0f.
Current lineage and random-init weights are preserved. No live runtime branch is manually edited.

## Diagnosed loop

At 21:59 Rome live causal tokens were33,478,285, with only3 acceptances among the
14 attempts after18:15. Repetition .236422 is only .001298 below the unchanged
durable limit .237721. Corpus reuse works (~5s), so preparation is no longer
this rejection bottleneck.

The protected recovery plan clamps every halved LR back to1/128. The acceptance
path also snaps a reduced LR back to1/128. Reducing batch128->64->32 changes the
sample, but does not proportionally reduce a fresh AdamW parameter update.
Meanwhile one tolerated acceptance immediately doubles the batch again, even
if NLL worsened within the gate tolerance. This causes repeated near-boundary
proposals and premature widening.

## Change

After a protected-wide attempt, permit bounded LR adaptation1/128->1/256->1/512
->1/1024->1/2048. Rollback halves the next real update; accepted updates retain
their proven LR. Legacy entry floor and healthy training remain unchanged.
Keep fresh AdamW reset, same causal/replay/AR objective and existing batch
recovery ladder. Require two same-batch/same-LR accepted checkpoints with
non-worsening NLL and repetition, finite metrics and no anchor violations
before widening32->64->128. Failed/tolerated-worsening results reset that proof.
Missing historical proof starts at zero. Transaction growth above16k still
requires its existing stronger durable headroom and three improving successes.

No threshold, anchor, holdout, token-accounting, model, tokenizer, source or
optimizer topology change. Rejected tokens remain uncounted. New shadow check
verifies persisted LR actually halves on rollback and stays fixed on acceptance.

## Paired current-checkpoint diagnostics

Same immutable checkpoint, same seeds/data slices, fresh AdamW, batch128,
16,256 causal labels,20% physical causal replay, identical unchanged gate.

| LR scale | Accepted trials | Accepted-equivalent labels | Observed rejections |
|---|---:|---:|---:|
|1/128|1/3|16,256|2/3|
|1/512|1/3|16,256|2/3|
|1/2048|2/3|32,512|1/3|

At seed7100000 the old and quarter-sized update increased repetition .032889
and .018263;1/2048 held repetition fixed and improved NLL by .00005705.
Seed7100001 still failed at1/2048 (.017998 repetition increase). Seed7100002
passed at all three rates, with tiny tolerated NLL increases. This is evidence
for a conservative adaptive option, not proof of sustained language improvement
or a representative acceptance rate. Two of three passing is not a guarantee.

Full journals: ablation.jsonl.gz and smallstep.jsonl.gz; compact paired hashes
and deltas: summary.json. Data scope is the persisted training-only replay
sample, not the full production corpus. Independent reset trials do not show a
sustained accepted sequence. Timing is excluded because two independent
benchmark processes overlapped; no throughput inference is made.

## Validation and follow-up

281 passed /1 optional skip across Phase5, efficiency, cache and LM tests.
Compilation and whitespace checks passed. GitHub production shadow and live
rollout follow this report; completion of those checks must be verified remotely.
Evaluate recent live acceptance and accepted tokens per calendar hour after
activation. Lower LR can reduce learning per label and can still fail when the
direction crosses a discrete greedy-generation boundary; no100k/hour claim.
The correction prevents retry-policy defects but does not yet create the
anchor+.04 headroom needed for large transaction growth.

Reproduction from a separate checkout of the pinned state:

```bash
PYTHONPATH=computer OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python reports/generalist-adaptive-recovery/run_paired.py --state-dir /tmp/pinned-state/generalist-state --output /tmp/paired-adaptive-recovery.jsonl
```
