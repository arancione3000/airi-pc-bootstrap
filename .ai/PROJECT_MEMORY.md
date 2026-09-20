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


- if active-run lookup fails after bounded retries, continuum attempts one serialized continuity dispatch;
- watchdog converts bounded run-list lookup failure into a stale recovery condition instead of crashing before recovery.


## 2026-09-20 — Structured anti-forgetting + persistent-state Health Gate

- structured anti-forgetting now replays verified Faulhaber induction recurrences;
- rich sum-of-powers discoveries contribute critical future promotion obligations even when the full quantified theorem string is outside the simple relation parser;
- persistent curriculum v1 metadata migrates in-memory to schema v2 with retry counters;
- added a persistent-state Health Gate before every autonomous state push;
- health checks architecture/domain closure, topology, router compatibility, discovery proof quality, Faulhaber certificates, curriculum retry bounds, benchmark failures and kernel/verifier status;
- failed health audits cannot replace the previous known-good `mathesis-state` checkpoint.


## 2026-09-20 — MATHESIS independent proof-integrity audit

- persistent `verified` metadata is no longer trusted by health or anti-forgetting;
  active relation discoveries are re-verified with the current CompositeVerifier;
- Faulhaber migration now checks that theorem text, exponent and stored
  polynomial agree before rebuilding base case, bounded exact samples and the
  structurally nontrivial finite-difference recurrence;
- legacy Faulhaber sample counts are capped during reproof, failed reproofs are
  archived, schema-v3 upgrades persist even without theorem moves, and the
  migration is idempotent;
- legacy non-Faulhaber discoveries missing the current proof-quality gate are
  re-proved or archived; structural tautologies keep their explicit discard
  classification;
- anti-forgetting accepts only currently valid active proof obligations,
  replays the complete active corpus by default (not a 16-theorem window), and
  ignores discarded, unverified, certificate-invalid or currently disproved
  rows;
- discovery novelty now distinguishes memory novelty from source novelty and
  human novelty; source/human novelty remain unassessed automatically;
- architecture benchmarks and persistent health now reject duplicate/unknown
  experts or strategies, disconnected nodes, bad topology, unknown proof-order
  methods, broken research-strategy coupling and out-of-bound genome parameters;
- CI exercises every SymPy domain through study signal -> weakness -> challenger
  -> expert/topology -> real domain benchmark, including polynomials;
- evolution challenger selection now filters hard-gate-ineligible candidates
  before score selection;
- interactive evolution preserves promoted symbolic-depth and discovery-beam
  settings immediately when rebuilding the discovery engine;
- self-rewrite path validation is a strict direct-file whitelist: nested state
  paths, unknown files, the state root and external paths are rejected;
- the Deep Mathlib smoke suite includes a nontrivial Faulhaber
  finite-difference obligation;
- continuum state writers remain serialized by the GitHub concurrency group,
  health runs before persistence and state pushes are non-force pushes;
- persistent-state health now fails closed on an unreadable champion and on
  malformed discovery/curriculum object or numeric fields, rather than
  silently validating fallback defaults or crashing;
- real continuum logs were observed carrying persistent discovery state across
  consecutive cycles and self-dispatching the successor; GitHub availability
  remains an external best-effort dependency.


## 2026-09-20 — AIRI Generalist LM foundation

A new general-purpose generative-model layer is being added behind Airi-PC.

Architecture decisions:
- the generalist model is a real decoder-only causal Transformer, separate from
  the old factual classifier and separate from MATHESIS-Ω;
- MATHESIS remains a proof-gated research/evolution signal source, not the user-
  facing language model and not a direct source of LM truth or weights;
- the mutable GeneralistGenome is bounded to architecture/training/reasoning
  parameters and adapters; verifier, permissions, tool execution policy,
  qualification and host boundaries remain outside the mutable genome;
- benchmark domains are separated into language, coding, data, reasoning,
  structured output and tool protocol so aggregate score cannot hide protected-
  domain regressions;
- supervised fine-tuning masks prompt tokens and optimizes assistant targets;
- tool calls are parsed as allowlisted JSON requests and are executed only by an
  injected Control Plane executor; unknown tools never reach execution and the
  agent loop has a hard step cap;
- a local checkpoint is not a production reasoning provider just because
  weights exist: config/weights/metadata/benchmark must be complete, benchmark
  qualification must be bound to the exact checkpoint SHA-256 digest, and both
  AIRI_GENERALIST_ENABLE=1 and AIRI_GENERALIST_PREFER=1 are required before the
  model can be preferred over ChatGPT;
- vision remains routed to ChatGPT until a separately benchmarked local vision
  model exists;
- already-downloaded Hugging Face causal models may be benchmarked locally via
  LocalTransformersBackend, using local_files_only=True and
  trust_remote_code=False. Autonomous model-code downloads are not allowed.

Capability honesty:
- CI uses a deliberately tiny model to prove causal generation, SFT learning,
  checkpointing and gating mechanics;
- no claim is made that the tiny CI model is GPT/Claude-class;
- useful general capability requires high-quality pretrained weights or
  substantial pretraining/fine-tuning plus a much broader held-out benchmark.

## 2026-09-20 — AIRI Generalist LM foundation

- added a real decoder-only causal Transformer layer for general-purpose language generation rather than reusing the factual classifier as a chatbot;
- added reversible versioned tokenization, chat serialization, autoregressive generation, supervised fine-tuning, checkpointing and local open-weight Transformers support with local-files-only loading and remote-code trust disabled;
- generalist capability is benchmarked separately across language, coding, data, reasoning, tool protocol and structured output;
- added a bounded GeneralistGenome with real architectural variants and research signals; MATHESIS can influence research directions only through proof-gated signals and cannot directly create trusted LM outputs or weights;
- added a persistent H24 generalist research continuum with serialized state writers, watchdog recovery, held-out validation and anti-forgetting;
- added a bounded Generalist Agent whose model can request only explicitly allowlisted tools; current integrated tools are read-only or computational;
- added an exact-digest qualification gate for native and local Transformers checkpoints, plus an authenticated local OpenAI-style model gateway;
- Airi-PC keeps ChatGPT as the default reasoning authority. A local Generalist provider appears only with an explicitly enabled, exact-digest qualified checkpoint; preference for it is a separate opt-in;
- added research-to-production promotion: a changed research champion is qualified once per digest, compared to the previous production champion with protected-domain anti-regression, atomically swapped, post-swap reverified and rolled back on integrity failure;
- research and production promotion logic remain outside the evolvable genome.
