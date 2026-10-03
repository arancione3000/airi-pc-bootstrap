# AIRI quality-gated token ramp — 2026-10-03

## Baseline and goal

Code inspected: main `553d3c57c5d9621d0aa9d7fb522c7c402df347a6`.
Read-only live snapshot: generalist-state `e217df0c3f27b58b50cf9e8399ee6ea0e821ac9f`, committed 08:11:50 UTC.
Lineage: `airi-5d3d25177d2e83f7`; 50,041,536 parameters, context 128,
12 transformer layers, d_model 96, SwiGLU FF width 14336, BPE vocabulary 384.
Training uses causal curriculum, protected replay, AdamW parameter groups and held-out
language diagnostics. Candidate/optimizer writes happen only after the language gate.

- Accepted causal cumulative count: 33,250,701; including rehabilitation: 33,256,355.
- Efficiency counter: 4,258,952 attempted / 434,848 accepted, 646 attempts / 92 acceptances.
- Historical compute rates: 931,994 attempted and 94,523 accepted tokens/hour.
- These compute rates exclude setup, queueing and other workflow costs.
- One-hour window ending at the snapshot's last event: 48,768 attempted / 16,256 accepted.
- Six-hour window: 109,728 attempted / 16,256 accepted (~2,709 accepted/hour).
- Last event: 16,256 accepted, causal replay 4,064 tokens (20% physical share), NLL delta
  -0.00004845, composite quality delta +0.00006798, unchanged repetition.
- The global acceptance rate is historical, not a rate for the new protected method.

The goal is 100,000 accepted causal tokens per real hour with measured learning.
It has **not** been demonstrated. Accepted tokens are supervised training volume, not
unique facts learned or a guarantee of conversational capability.

## Changes

1. Fix the workflow contract to accept the existing protected batches 32/64/128
   (roughly 4k/8k/16k). Retain causal replay 18–22%, autoregressive objective and
   a minimum causal token budget proportional to batch and checkpoint context.
   Previously the contract demanded batch 128 even when the recovery ladder
   selected 64 or 32, preventing persistence of valid recovery telemetry.
2. Preserve the protected method on the first rejection and after health recovers.
   Automatic runs may use the remaining bounded segment slots after rollback;
   explicit manual canaries still stop after one attempt. Maximum remains four.
3. Add transaction rungs 16,256 -> 32,512 -> 65,024 for the current context.
   Effective batch remains 128 and LR remains 1/128. Growth changes optimizer
   updates per transaction, not batch size. The final rung aligns four updates,
   slightly above the old nominal 62,500 cap.
4. Promotion requires three consecutive accepted, improving transactions at the
   same rung, non-worsening language NLL, no anchor violations, no recovery hold,
   multiword rate >=80% and repetition <= durable anchor +0.04. Missing/nonfinite
   essential measurements deny promotion. At a new rung the proof streak restarts.
   Rejection resets proof and falls back to the protected smaller-batch ladder.
5. Add recent calendar throughput from at most 64 completion events. Exclude the
   first event's tokens because its start is unknown. Preserve legacy compute
   counters. Target attainment needs >=one hour of observed event span,
   >=100k accepted/hour and positive measured quality gain. This is a recent
   observed-window rate, not a lifetime or fixed one-hour rate.

The live snapshot remains language-fragile. It is not eligible for an immediate
32k/65k promotion. No timer, fabricated acceptance or relaxed gate forces growth.
Data sources, held-out sets, anchor, lineage and pretrained-weight prohibition
remain unchanged. Only the normal workflow may write runtime state.

## Verification

267 passed / 1 optional skip: Phase 5, learning efficiency, efficiency engine and
Generalist LM suites, with CPU PyTorch and MATHESIS dependencies installed.
New tests exercise growth rungs, rollback, quality/headroom denial, proof streak
resets, real calendar gaps and the actual jq workflow contract against all three
valid batches and malformed replay/batch contracts. Python compilation and diff
whitespace checks passed. An initial broad-suite failure also occurred with the
unmodified source; installing the repository's z3 dependency resolved it.

These are functional/controller tests. No new live 50M training benchmark proves
100k/hour or successful 32k/65k updates. Watch normal persisted telemetry before
claiming either. If the model cannot establish headroom, it stays in recovery.
