# Airi-PC Project Memory

## 2026-09-19 — MATHESIS-Ω v0

Scope: add a separate proof-gated mathematical agent without weakening Airi-PC
runtime or Evolution Lab security boundaries.

Design decisions:
- natural-language interpretation is advisory; mathematical truth is proof-gated;
- safe AST-to-SymPy IR, independent SymPy/Z3/Lean checks, adversarial counterexamples;
- restricted program-synthesis arena with reference/property tests;
- pure-Python growing neural router is advisory only;
- web access is HTTPS read-only through Airi's hardened researcher;
- self-evolution rewrites only architecture/router state, never verifier-kernel files;
- candidate/champion promotion requires benchmark improvement and kernel-integrity checks;
- autonomous state persists on a dedicated `mathesis-state` Git branch;
- hourly continuum plus independent stale-heartbeat watchdog.

The feature must be merged only after the dedicated MATHESIS CI and end-to-end
tests pass.


## 2026-09-19 — MATHESIS-Ω v1 autonomous discovery

- broadened safe SymPy IR with assumptions and whitelisted mathematical functions;
- added multi-domain symbolic laboratory;
- added bounded conjecture discovery with internally-novel/human-novelty separation;
- added exact Faulhaber-style discovery using interpolation + finite-difference proof obligations;
- added rotating HTTPS read-only mathematical curriculum;
- architecture genome now evolves topology, symbolic depth, discovery beam and research budget;
- architecture arena evaluates multiple challengers per cycle;
- intent router now understands calculus, symbolic analysis and discovery requests;
- old router shapes are migrated by retraining when the intent vocabulary changes;
- fast continuum scheduled every five minutes with a duplicate-aware watchdog every ten minutes;
- deep theorem workflow uses Lean 4.34.0 and Mathlib 4.34.0;
- all discovery/curriculum histories are bounded to avoid unbounded persistent-state growth;
- no result may be called new to humanity automatically; only new to MATHESIS state.


## 2026-09-19 — MATHESIS experience feedback + anti-forgetting

- added an auditable ExperienceAnalyzer;
- discoveries and curriculum coverage now generate bounded architecture-search hints;
- challenger generation diversifies across different observed weaknesses;
- experience-derived priorities are considered before generic benchmark gaps;
- verified relation-style discoveries become critical replay tests for all future champions;
- a candidate that forgets previously verified mathematical relations cannot be promoted;
- web/research evidence still cannot directly edit the verifier kernel;
- five-minute continuum cron uses an explicit minute list plus duplicate-aware watchdog.


## 2026-09-19 — MATHESIS continuous self-handoff

- continuum no longer depends only on GitHub cron;
- each completed main-branch cycle waits to the five-minute cadence and dispatches its successor;
- handoff checks queued/in-progress runs to avoid duplicate chains;
- cron and watchdog remain independent recovery paths;
- workflow has explicit Actions write permission for workflow_dispatch;
- verified discovery-domain gaps persist until the corresponding expert is acquired;
- read-only curriculum cadence increased so research occurs roughly every 15 minutes at research_budget=2;
- external GitHub outages/limits remain outside the repository's ability to guarantee zero downtime.


## 2026-09-19 — MATHESIS domain-closure invariant

- fixed a feedback-loop gap where curriculum could request a polynomials expert that architecture could not build;
- added polynomials to evolvable experts;
- added algebra/polynomial domain benchmarks;
- CI now requires every SymPy lab domain to be representable by the architecture expert set.


- fixed symbolic-depth scoring to use the explicit symbolic-depth proof result rather than a fragile positional task index;
- added regression coverage showing anti-forgetting theorem failures cannot corrupt the reported verified symbolic depth.


- legacy Faulhaber discoveries are now re-proved with the new nontrivial induction obligation;
- successful legacy proofs are upgraded in place; failed reproofs are archived out of the active theorem corpus;
- quality migration persists immediately even if the current discovery cycle finds no new candidate.


- curriculum no_sources/research_error retries now remain on the same mathematical domain;
- successful retry clears the per-domain retry counter;
- after three failures the curriculum advances to avoid permanent source-induced stalls.


## 2026-09-20 — MATHESIS handoff network hardening

- added explicit connect/total timeouts and retries to continuum, watchdog and deep-Mathlib GitHub API calls;
- raised the continuum job timeout margin to 15 minutes so cold setup plus persistence/handoff cannot be killed prematurely;
- CI now enforces that autonomous workflow API calls remain time-bounded and retrying.
