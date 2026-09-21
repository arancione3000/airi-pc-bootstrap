# AIRI Generalist LM

AIRI Generalist LM is the generative-model layer intended to become the general-purpose language/reasoning core behind Airi-PC.

Roles are deliberately separated:
- Generalist LM: chat, language, coding, structured output, data analysis, reasoning and tool-call generation.
- MATHESIS-Ω: proof-gated mathematical research and bounded architecture-research signals.
- NeuroEvolution: evolutionary experimentation and specialist models.
- Airi-PC Control Plane: permissions, tools, browser/runtime actions, persistence and verification.

The repository now contains a real decoder-only causal Transformer. CI uses a deliberately small configuration to prove training and generation mechanics; that tiny CI model is not claimed to have GPT/Claude-level capability.

## Core model

The local model has causal multi-head self-attention, residual pre-norm blocks, SwiGLU feed-forward layers, tied token/output embeddings, autoregressive generation and next-token cross-entropy training.

The dependency-free ByteTokenizer is reversible for UTF-8 and versioned as `byte-v1`. Supervised fine-tuning masks prompt tokens so assistant targets drive learning.

## Capability surface

The runtime exposes text completion, multi-turn chat, code generation, structured-data analysis and allowlisted tool-call generation. Tool calls are data, never executable model output: Airi-PC still validates permissions and performs any action.

## Benchmark and promotion

`benchmarks.py` scores language, coding, data, reasoning, tool protocol and structured output separately. A candidate cannot hide a protected-domain regression behind gains elsewhere. Promotion requires no critical failures, no protected-domain regression and a meaningful overall gain.

## Evolution DSL

`GeneralistGenome` may vary bounded context length, width, heads, layer count, feed-forward width, dropout, learning rate, reasoning depth and symbolic/code/data/retrieval adapters. Tokenizer family and tool protocol are allowlisted. Verifier code, Airi-PC permissions, secrets, host boundaries and promotion rules are not mutable genome fields.

MATHESIS may contribute proof-gated signals such as `symbolic_reasoning_signal`. Mathematical discoveries never become language-model truth or weights directly; they only influence candidate research directions that must still pass the independent generalist benchmark.

## Open-weight models

`LocalTransformersBackend` can benchmark an already-downloaded Hugging Face causal LM from a local directory with `local_files_only=True` and `trust_remote_code=False`. The autonomous loop does not download model code.

## Operational gate

A checkpoint is not an Airi-PC reasoning provider just because it exists. `qualification.py` writes a benchmark report and marks it qualified only after critical-domain tests and the configured minimum score pass. Until then, ChatGPT remains the production reasoning authority and the local generalist model is a challenger.

## What is real today

CI verifies that the causal model generates tokens, supervised fine-tuning lowers loss, checkpoints round-trip, tool output is allowlisted data, benchmark domains are independent, architecture mutation stays inside the DSL, and MATHESIS contributes only proof-gated research signals.

What is not claimed: a tiny CI model has frontier-level knowledge. Reaching that level requires a qualified pretrained foundation model or large-scale pretraining/fine-tuning plus broader benchmarks.

## Provider and agent integration

Airi-PC can expose the local checkpoint through
`control_plane/generalist_provider.py`, but the provider is fail-closed:

1. `config.json`, `model.pt`, `metadata.json` and `benchmark.json` must exist;
2. the benchmark attestation must be bound to the exact SHA-256 checkpoint
   digest;
3. the checkpoint must be benchmark-qualified;
4. `AIRI_GENERALIST_ENABLE=1` must be set;
5. the router only prefers it when `AIRI_GENERALIST_PREFER=1` is also set.

Vision remains routed to ChatGPT unless a future independently benchmarked
vision-capable local model is added.

The bounded `GeneralistAgent` implements model -> tool request -> Control
Plane result -> model loops. Unknown tools are rejected before execution,
tool results are size-bounded and every run has a hard maximum number of
steps. Model text never directly executes shell, browser or file actions.

Checkpoint qualification is invalidated automatically if config, weights or
metadata change after benchmarking. This prevents stale benchmark results from
silently qualifying new weights.

## Research-to-production promotion

The autonomous generalist research champion is not automatically the operational Airi-PC model.
After a research champion changes, the continuum attempts a digest-cached protected qualification:

1. copy the exact research checkpoint into an isolated production candidate;
2. run the broader generative qualification suite;
3. bind the attestation to the exact checkpoint digest;
4. if no production champion exists, require the configured minimum score;
5. if a production champion already exists, additionally require a meaningful aggregate score gain and zero protected-domain regression;
6. atomically swap the candidate into production;
7. re-read qualification after the swap and roll back on any integrity failure;
8. persist promotion history together with the research state.

A failed qualification is a normal research outcome and does not stop future research cycles.
The same failed research checkpoint is not repeatedly re-qualified: its digest is cached until the research champion really changes.

Research can experiment continuously, while production promotion stays conservative and cannot rewrite its own qualification rules.

## Coding and data tool boundaries

Qualified Generalist autocoding is opt-in and requires both an explicit file scope and an explicit test command. Proposed edits still pass the existing snapshot, rollback, diff and guardrail workflow before any commit.

For data work, the Generalist Agent exposes deterministic read-only helpers for arithmetic, descriptive statistics, JSON-table profiling, and bounded aggregate/group-by operations. These helpers do not execute model-generated Python.


## Continual learning and persistent curriculum

The Generalist research loop no longer relies only on architecture search over a fixed toy curriculum.

Each autonomous research cycle now:

1. expands a bounded, persistent, mechanically-labeled curriculum memory;
2. guarantees that persistent training prompts do not overlap the protected validation prompts;
3. replays the stored curriculum on every research candidate;
4. creates a dedicated continual-learning challenger by copying the current champion weights and fine-tuning them further;
5. still trains independent architecture challengers in parallel;
6. rejects any candidate that regresses in held-out domain loss, teacher-forced target accuracy, solved-item replay, or real autoregressive generation.

The curriculum memory is stored inside the same persistent Generalist state directory, so the cloud continuum carries it from one cycle to the next. It is bounded by `AIRI_GENERALIST_RESEARCH_CURRICULUM_MAX_ROWS` (default 1200) to keep compute/storage finite.

Research validation now includes a real generation probe: one held-out item per protected domain is decoded autoregressively. A model cannot be promoted merely because teacher-forced loss improves while previously generated capabilities disappear.


## Architecture inheritance and RoPE exploration

Architecture challengers no longer discard all accumulated knowledge by default.
Before fine-tuning, the research loop copies every champion state tensor whose
name and shape exactly match the challenger. Progressive same-width growth uses
a layout-aware **Net2Grow** transfer instead of a raw tensor prefix: SwiGLU
gate/value halves are mapped into the corresponding larger halves, newly added
FFN capacity starts with zero contribution, and newly inserted Transformer
blocks have zero attention/FFN output projections so each new residual block is
an identity at initialization. This preserves the champion function while the
new capacity begins learning.

Context-row growth retains learned positions and initializes only new positions
neutrally. Width growth may retain compatible embedding coordinates but is not
claimed function-preserving because normalization spans the hidden width.
Byte-to-BPE migration keeps its deterministic byte-derived initialization.
Shrinking tensors, concatenated-layout mismatches and incompatible tokenizer
migrations remain fail-closed. The transfer report records exact, structured
growth and identity-initialized tensors separately.

The bounded architecture DSL can now explore learned absolute positions,
sinusoidal positions, and RoPE (rotary positional embeddings). RoPE is applied
inside causal attention and is permitted only when the attention head dimension
is even. It remains a challenger option, not a hard-coded claim that RoPE is
always superior.

Persistent curriculum state is fail-closed. Every replay row carries a digest;
invalid domains, malformed SFT messages, duplicate rows, replay-cap violations,
digest mismatches, or protected-validation overlap stop research health and
prevent state persistence.


## Generalist Free-Speed continuum

The autonomous Generalist continuum now uses a parallel successive-halving
tournament instead of spending a full training budget on candidates one after
another.

Each research cycle performs:

1. one planner restores the exact research champion, MATHESIS signals,
   curriculum memory and persistent auto-data state;
2. eight candidate genomes run concurrently on standard GitHub CPU runners;
3. a reducer removes catastrophic held-out regressions and keeps four;
4. the four survivors receive a larger training/pretraining budget;
5. a second reducer keeps two finalists;
6. both finalists run a 20-step SFT budget plus causal pretraining on two
   independent seeds;
7. an external reducer reloads the winning checkpoint and recomputes protected
   validation/canary metrics before research promotion;
8. the ordinary digest-bound production qualification remains a separate,
   stricter gate.

The tournament prefers real autoregressive-generation gains first, then
byte-normalized held-out NLL, parameter efficiency and research score. A matrix
worker cannot promote itself.

### Weakness-directed curriculum

Held-out per-domain NLL is converted into bounded replay weights. The strongest
domain stays at weight 1x while the weakest may receive up to 3x replay.
Domain-balanced replay remains the neutral base, so the adaptation is explicit
rather than an accidental consequence of historical curriculum frequency.
Promotion thresholds are unchanged.

### Progressive scaling

Research capacity is no longer fixed at the initial ~100k-parameter scale.
The planner can propose bounded tiers of approximately 250k, 500k, 1M and 2M
parameters. The first tiny champion receives an early 250k probe; later tiers
are explored after several cycles without meaningful champion/score progress.
Every larger candidate still competes under the same held-out and production
gates, so adding parameters is never treated as an improvement by itself.

### Autonomous permissive corpus growth

The planner may grow a persistent external text/code corpus without downloading
model weights. Discovery is restricted to the GitHub HTTPS API. Admission is:

repository search -> SPDX license allowlist -> immutable commit SHA -> bounded
raw-file quarantine -> UTF-8/printability/information/secret filters -> SHA-256
deduplication -> approved corpus.

Allowed licenses are explicit permissive/public-domain identifiers such as MIT,
Apache-2.0, BSD, ISC, CC0, 0BSD and Unlicense. Repositories with missing or
unclear license metadata are rejected. Each cycle has a small growth budget and
the persistent corpus has a larger hard cap; both repository-grounded and
approved external documents are content-deduplicated before causal pretraining.

### Runner efficiency

The Free-Speed workflow uses dependency caching through `setup-python` and
keeps the expensive candidate stages parallel (8 -> 4 -> 2). Standard
production qualification and state persistence occur only once in the final
reducer.

## Qualification v2 and immutable governance

Production qualification is versioned independently from model checkpoints.
Qualification v2 uses prompts that are disjoint from fixed training,
validation, deterministic continual-learning replay and rotating canaries. Each
generalist capability domain has at least one critical production gate.
Changing the protected suite invalidates legacy v1 attestations, so old
checkpoints must be requalified against the new examination.

The local Generalist autocoder cannot edit its own model/research package,
qualification and promotion code, Control Plane provider/router/gateway,
autocoding engine, reasoning policy, Generalist workflows, or Generalist state
and attestations. Architecture improvement happens through the bounded genome
DSL and research cycle instead of source-level self-modification of the
verifier/governance layer.

The Generalist agent tool loop remains read-only: calculator, bounded data
statistics/table aggregation, workspace file reads/search and project
inspection. Tool requests are parsed against an explicit allowlist and numeric
analysis rejects non-finite inputs.


## Hardened MATHESIS bridge and local deployment

MATHESIS state is treated as untrusted at the Generalist boundary. Persisted
`verified=true`, proof-gate labels and certificate metadata are not sufficient:
the bridge re-runs the current MATHESIS CompositeVerifier before emitting a
symbolic/deep-mathematics architecture signal. If the verifier stack is not
available, the bridge fails closed and emits no MATHESIS-derived signals.

The read-only Generalist tool bridge also excludes sensitive workspace paths
(such as environment/credential/secret/key material), limits readable file
formats and sizes, and applies bounded file/byte traversal budgets to search
and project analysis.

A qualified research champion can reach an enabled local Airi-PC runtime through
an explicit transactional sync:

- set `AIRI_GENERALIST_ENABLE=1`;
- optionally set `AIRI_GENERALIST_SYNC_FROM_GITHUB=1`;
- the bootstrap fetches only the `generalist-state` branch;
- the copied production weights are re-qualified locally against the protected
  qualification suite rather than trusting the remote attestation;
- installation uses staged swap + backup rollback;
- fetch/requalification failure leaves the existing local checkpoint untouched.

The sync is intentionally opt-in. A research promotion on GitHub never silently
changes the user's active reasoning provider.


## Scalable learning: pretraining, distillation and LoRA

The research Continuum deliberately remains small enough for bounded GitHub
Actions CPU experiments. It is not the mechanism used to train a frontier-sized
foundation model.

The scalable-learning path is separate:

1. **Native causal pretraining** — `pretraining.py` ingests reviewed local text
   under explicit allowed roots, rejects path/symlink escapes, unsupported
   formats, oversized files and duplicate content, packs causal token blocks,
   and trains next-token prediction. The resulting checkpoint is unqualified
   until it passes the normal protected benchmark.
2. **Response distillation** — `distillation.py` can query an already-local
   teacher backend and turn bounded responses into reviewable SFT JSONL. Teacher
   output is training data only; it is never treated as factual truth or as a
   qualification result.
3. **Local LoRA** — `transformers_lora.py` can fine-tune an already-downloaded
   open-weight causal LM with PEFT/LoRA. Base weights are opened with
   `local_files_only=True` and `trust_remote_code=False`. The adapter is
   written outside the base model directory and must be independently
   qualified before production use.

CLI examples:

```bash
python -m generalist_lm.cli pretrain-native CHECKPOINT CORPUS_DIR \
  --allowed-root /reviewed/data --output /models/airi-pretrained

python -m generalist_lm.cli distill-transformers /models/local-teacher \
  prompts.jsonl distilled.jsonl

python -m generalist_lm.cli lora-transformers /models/local-foundation \
  distilled.jsonl /models/airi-lora
```

Large-model LoRA is intentionally an optional environment and requires
`transformers` and `peft` in addition to PyTorch. These dependencies are not
silently installed by Airi-PC and the autonomous Continuum does not download
foundation weights.

This separation is important: MATHESIS and the small Continuum can discover
architectural/curriculum ideas continuously, while expensive model training can
run on suitable hardware. Nothing trained by either path becomes the active
reasoning provider until protected qualification succeeds.


## KV-cache and scalable SFT execution

Autoregressive decoding supports a bounded per-layer KV cache. Cached greedy decoding is regression-tested against the original full-prefix path across the supported positional-encoding variants, and the cache is discarded/recomputed when the configured context window is reached so stale absolute positions are not reused.

Supervised fine-tuning supports gradient accumulation and explicit fp32/bf16/fp16 precision policies. Unsafe precision/device combinations fail closed. These execution optimizations do not alter qualification, protected-domain regression gates, tool permissions or the MATHESIS verifier boundary.


## Grounded repository pretraining

The research continuum can optionally perform a small amount of causal
pretraining on the repository's own code and documentation before supervised
generalist training. This is intended to improve project vocabulary, APIs,
coding syntax and architectural context without teaching the model its exam.

The corpus builder is bounded and deterministic. It:

- prunes VCS, virtualenv, build, node_modules, state and internal .ai trees;
- excludes all tests;
- excludes Generalist benchmark, qualification, production-promotion,
  research-health/evolution and Control Plane governance files;
- rejects symlink escapes and sensitive file paths;
- redacts common credential/token patterns from otherwise safe source files;
- limits files, bytes, per-file size and emitted training documents.

Causal-pretraining loss evaluation is sampled deterministically with a bounded
number of blocks. Training may sample the larger permitted corpus, but measuring
pretraining progress cannot accidentally turn a modest corpus into thousands of
extra full-model forward passes per candidate.

Repository pretraining is research-only. It cannot qualify or promote a model
to production by itself. Held-out generalist validation, rotating canaries and
production qualification remain separate.


## BPE tokenizer checkpoint support

The Generalist stack includes a deterministic `bpe-v1` tokenizer research
implementation. It starts from the reversible byte vocabulary and learns
bounded byte-pair merges, preserving the existing chat-role special tokens.

BPE artifacts are versioned and digest-bound. A `bpe-v1` checkpoint must
contain `tokenizer.json`; the runtime refuses to load the checkpoint without
it or when vocabulary/version metadata disagree. Production qualification
includes the tokenizer artifact in the exact checkpoint digest, so modifying
the tokenizer after qualification invalidates the attestation.

This tranche adds tokenizer training, persistence and checkpoint support only.
The autonomous research genome remains `byte-v1` until a later migration path
can compare byte and BPE checkpoints fairly without silently resetting learned
weights or changing the protected benchmark contract.


## Domain-balanced replay

Persistent curriculum expansion is intentionally weakness-directed, so its raw
history can contain more coding/reasoning examples than language or data.
Before supervised fine-tuning, the neutral replay base is now deterministically
equalized across every observed generalist domain. Explicit genome focus genes
are applied only after that balance is established.

This is an anti-interference mechanism, not a relaxed promotion rule. Held-out
per-domain loss, rotating canaries, token accuracy and autoregressive
anti-forgetting gates remain unchanged.


## Evolutionary BPE migration

The research loop may now evaluate `bpe-v1` as an architecture challenger.
This is deliberately stricter than merely supporting BPE checkpoint files:

- BPE merges are learned only from the normal training/replay corpus and the
  protected repository-pretraining corpus; validation, rotating-canary and
  production-qualification prompts are excluded;
- the 264 stable special/byte embedding rows are inherited exactly;
- each newly created BPE token is initialized from the mean of the inherited
  byte embeddings that spell that token, instead of random reinitialization;
- the actual BPE vocabulary size is included in the parameter budget;
- cross-tokenizer research promotion uses held-out negative log-likelihood per
  UTF-8 target byte (and bits/byte), not cross-entropy per token;
- token-accuracy comparisons are retained within one tokenizer family but are
  not used as a cross-tokenizer promotion signal;
- real autoregressive generation, per-domain byte-normalized validation,
  rotating canaries and solved-item anti-forgetting remain hard gates.

This prevents a tokenizer from appearing better merely because it emits fewer
tokens while still allowing genuinely more efficient representations to
compete.



## AIRI Native Foundation — canonical scratch track

The canonical AIRI path does **not** require or inherit an external pretrained
model. The repository now contains a separate `AIRI Native Foundation` family
whose root checkpoint is created from random initialization only.

Native Foundation v1 is implemented directly in
`generalist_lm/native_foundation.py` and intentionally has no Hugging Face,
Transformers or `from_pretrained` dependency. Its architecture is a
decoder-only causal LM with:

- pre-norm RMSNorm;
- rotary positional embeddings (RoPE);
- grouped-query attention (GQA) with compact KV caches;
- SwiGLU feed-forward blocks;
- residual-projection scaling;
- optional tied token/output embeddings;
- causal generation with KV-cache reuse.

The legacy small Generalist LM remains untouched for checkpoint compatibility.
The Native family is a new architecture line intended to scale independently.

Every Native root checkpoint contains
`airi-native-foundation.json`, `native-config.json` and `model.pt`.
A BPE-based root additionally contains an AIRI-owned `tokenizer.json`.
The manifest records and verifies the architecture family, random-init seed,
weight origin, tokenizer identity and digests of the config/model/tokenizer.
`external_pretrained=true` is invalid by construction.

The root-creation API accepts no source-model or pretrained-weight argument.
BPE roots accept only a tokenizer that parses as AIRI `bpe-v1`, and its
vocabulary must exactly match the Native model config.

Planning is deliberately separate from allocation:

```bash
# No weights allocated; useful for hardware/training planning.
python -m generalist_lm.cli native-foundation-plan 1b
python -m generalist_lm.cli native-foundation-plan 3b
python -m generalist_lm.cli native-foundation-plan 7b

# Actual random initialization. The default safety guard refuses accidental
# large allocations unless the caller raises the limit explicitly.
python -m generalist_lm.cli native-foundation-init ./native-root \
  --profile micro \
  --tokenizer-json ./tokenizer.json \
  --seed 17

python -m generalist_lm.cli native-foundation-status ./native-root
```

The provided planning profiles are approximately 1.1B, 3.3B and 7.1B
parameters with GQA. These are architecture plans only; CI never instantiates
the billion-parameter profiles.

This Native track is the canonical basis for the later data/training and
autonomous-evolution phases. The external Foundation/Transformers track below
remains optional compatibility infrastructure and is not required to become the
AIRI champion.

### Native Phase 2 — AIRI-owned data and scratch pretraining

The Native track has its own training path in
`generalist_lm/native_data.py` and `generalist_lm/native_training.py`.
It does not call Transformers, Hugging Face or `from_pretrained`.

A corpus is admitted through a local `native-corpus-v1` JSON manifest. Every
document must be explicitly marked `approved_for_training=true` and declare its
domain, language, license/provenance class and sampling weight. Corpus loading is
bounded, exact-content deduplicated, local-only and rejects path/symlink escapes.
The canonical domain surface is:

- general language;
- English;
- Italian;
- code;
- mathematics;
- reasoning;
- data analysis;
- tool use.

The AIRI BPE tokenizer is trained directly from that reviewed corpus. Its digest
and vocabulary are then bound into the Native root checkpoint, so tokenizer and
model ancestry stay independently verifiable.

Example manifest:

```json
{
  "version": "native-corpus-v1",
  "documents": [
    {
      "path": "text/it-0001.txt",
      "domain": "language-it",
      "language": "it",
      "license": "REVIEWED_LICENSE_ID",
      "source_type": "permissive",
      "approved_for_training": true,
      "weight": 1.0
    }
  ]
}
```

The corpus can also be materialized automatically from a reviewed
`native-sources-v1` catalog. This removes the need to copy training files by
hand while keeping source admission fail-closed. AIRI downloads only pinned
HTTPS objects, verifies the exact SHA-256 before accepting bytes, enforces host
and size budgets, refuses unsafe paths/redirects, records source URL/license in
the generated manifest, and then hands the result to the same local corpus
audit used above. Source discovery is intentionally **not** part of Phase 2;
later autonomous research may propose catalog changes, but it does not bypass
these acquisition checks.

Example source catalog:

```json
{
  "version": "native-sources-v1",
  "allowed_hosts": ["datasets.example.org"],
  "sources": [
    {
      "name": "approved-general-shard-0001",
      "url": "https://datasets.example.org/general-0001.jsonl",
      "sha256": "REPLACE_WITH_64_HEX_SHA256",
      "filename": "general/general-0001.jsonl",
      "domain": "general",
      "language": "en",
      "license": "REVIEWED_LICENSE_ID",
      "source_type": "permissive",
      "approved_for_training": true,
      "weight": 1.0
    }
  ]
}
```

Materialize the corpus without manually supplying document files:

```bash
python -m generalist_lm.cli native-corpus-acquire \
  ./sources/native-sources.json ./corpus \
  --allowed-host datasets.example.org

python -m generalist_lm.cli native-corpus-audit \
  ./corpus/native-corpus.json --allowed-root ./corpus
```

Audit and train the tokenizer before allocating a BPE Native model:

```bash
python -m generalist_lm.cli native-corpus-audit ./corpus/native-corpus.json \
  --allowed-root ./corpus

python -m generalist_lm.cli native-tokenizer-train \
  ./corpus/native-corpus.json ./artifacts/tokenizer.json \
  --allowed-root ./corpus \
  --vocab-size 32768

# Use the vocab_size reported by the tokenizer command.
python -m generalist_lm.cli native-foundation-init ./native-root \
  --profile micro \
  --vocab-size 32768 \
  --tokenizer-json ./artifacts/tokenizer.json \
  --seed 17
```

Native pretraining uses AdamW with beta2=0.95, warmup plus cosine decay,
gradient clipping, gradient accumulation, held-out validation, deterministic
weighted curriculum sampling and optional fp16/bf16 autocast. It understands a
`torchrun` distributed environment and uses DDP when `WORLD_SIZE>1`. Training
writes a new `airi-native-descendant` checkpoint; it never overwrites the
parent. The child manifest binds the exact parent checkpoint digest, and the
trainer state is independently hashed so optimizer/RNG resume data cannot be
silently modified.

Single-process example:

```bash
python -m generalist_lm.cli native-train \
  ./native-root ./corpus/native-corpus.json \
  --allowed-root ./corpus \
  --output ./native-step-10000 \
  --max-steps 10000 \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --precision auto

python -m generalist_lm.cli native-training-status ./native-step-10000
```

Multi-GPU example:

```bash
torchrun --standalone --nproc-per-node=8 -m generalist_lm.cli native-train \
  ./native-root ./corpus/native-corpus.json \
  --allowed-root ./corpus \
  --output ./native-step-10000 \
  --max-steps 10000 \
  --device cuda \
  --precision bf16
```

CI intentionally proves this path only on tiny randomly initialized models and
small project-owned corpora: it verifies that held-out next-token loss falls,
lineage is preserved, optimizer state resumes, and tampering is rejected. A
1B/3B/7B model is not claimed to be trained by CI; those profiles require a
separate reviewed large corpus and suitable accelerator capacity.


## AIRI Native Phase 3 — proof-gated autonomous evolution

Phase 3 connects the scratch Native Foundation to a bounded autonomous research
loop. It is intentionally separate from the legacy Generalist research genome:
Native challengers operate on the real Native architecture/training surface.

The cycle is:

1. freeze and independently evaluate the current Native champion;
2. collect re-verified MATHESIS signals when available;
3. search bounded public research metadata from arXiv, fall back to OpenAlex when arXiv is unavailable, and search GitHub for code/dataset evidence;
4. convert only derived research **tags** into a fixed local mutation DSL;
5. train an equal-budget control continuation;
6. train challengers for optimizer, curriculum, GQA/FFN/context/RoPE and
   tokenizer experiments;
7. reload every checkpoint and independently recompute held-out loss per domain;
8. promote only a candidate that beats both the frozen champion and the
   equal-budget control without forbidden provenance, integrity failures or
   held-out regressions.

Remote material is never executed and cannot provide code, shell commands, file
paths or arbitrary hyperparameter values. arXiv is not a single point of failure:
when its public API is unavailable or returns a transient service error, AIRI uses
OpenAlex academic metadata instead while preserving the same untrusted-evidence
boundary. The online layer emits bounded tags
such as `optimizer`, `tokenizer`, `gqa`, `long-context`, `curriculum`,
`dataset` and `efficiency`. All concrete mutations are generated locally
from reviewed ranges. GitHub results that look like permissively licensed
datasets are stored as source proposals only; automatic admission still
requires an immutable revision/content URL, a pinned SHA-256 and the Phase-2
corpus audit. A challenger cannot edit or call its own promotion gate.

Architecture and tokenizer mutations require a new random-init Native root and
scratch retraining. A short scratch run may be recorded as a proxy experiment,
but it cannot replace a champion with a larger cumulative training budget.
External pretrained weights remain forbidden.

Useful commands:

```bash
python -m generalist_lm.cli native-research-online \
  --signal symbolic_reasoning_signal

python -m generalist_lm.cli native-evaluate \
  ./native-checkpoint ./corpus/native-corpus.json \
  --allowed-root ./corpus

python -m generalist_lm.cli native-evolve \
  ./native-evolution-state \
  ./native-seed \
  ./corpus/native-corpus.json \
  --allowed-root ./corpus \
  --mathesis-state ./mathesis-state \
  --challengers 3 \
  --trial-steps 100
```

The repository also contains `.github/workflows/native-evolution.yml`. It runs
an hourly research-scale cycle, persists the verifier-approved champion on the
`native-evolution-state` branch, restores proof-gated MATHESIS signals, and
attempts bounded online research each cycle. Its automatic bootstrap corpus is
derived from project-owned repository text so the workflow can prove the whole
loop without requiring a manually uploaded dataset.

That workflow is a research harness, not a claim that a billion-parameter model
has been trained. The same engine can be pointed at a larger reviewed Native
corpus and accelerator-backed checkpoint by changing the runtime/state inputs;
promotion rules stay external to the challenger.



## AIRI Native Phase 4 — Lattice architecture research

Phase 4 explores whether AIRI can obtain more useful intelligence per unit of
compute by changing the model family itself instead of only tuning a
Transformer. The experimental family is **AIRI Lattice v0**.

Lattice is causal but does not use global self-attention. Each token updates a
constant-size hierarchy of recurrent memory bands with mathematically different
time scales. A geometric novelty/surprise signal controls how deeply new
information is written: ordinary predictable tokens mostly update fast memory,
while surprising inputs can reach slower bands. Learned row-normalized routing
lets memory bands exchange information without making the recurrent state grow
with context length.

Each Lattice cell also contains sparse top-k experts. The router sees both the
current token representation and the current memory summary, and only selected
experts execute for each token. A shared recurrent reasoner spends extra passes
only when the surprise-derived compute budget says the token is difficult.
This lets AIRI change reasoning depth without duplicating parameter blocks.

The architecture is deliberately not declared superior or novel by fiat.
Modern state-space/recurrent-memory/MoE systems already demonstrate many useful
ingredients. AIRI Lattice is treated as a project-specific experimental
composition whose value must be established empirically.

MATHESIS can evolve the structural genome itself:

- number and mathematical half-life of memory bands;
- how selectively surprise reaches deep memory;
- cross-band routing strength;
- adaptive reasoning threshold, power and maximum recurrent passes;
- number, width and top-k sparsity of experts;
- number of physical recurrent cells;
- optimizer learning rate and regularization.

Every generated configuration must stay inside a conservative bounded-state
region before it receives any SGD budget. Because routing rows are convex
softmax mixtures, candidate writes are tanh-bounded, memory decays remain in
(0,1), and lattice mixing is capped at 1, the memory recurrence has a
conservative infinity-norm stability certificate independent of sequence
length.

Architecture search is deliberately compute-frugal:

1. MATHESIS/research signals generate a diverse mutation population;
2. mathematical stability/cost filters reject bad regions essentially for free;
3. only a tiny shortlist receives scratch training;
4. AIRI Lattice and a parameter-matched Native Transformer see the same corpus,
   tokens, optimizer budget and held-out split;
5. the empirical gate compares held-out loss, domain regressions, active
   parameters and state memory;
6. a Lattice win promotes only the **research architecture champion**;
7. the canonical Native Transformer remains unchanged until repeated,
   larger-budget evidence justifies a migration.

The existing Native Evolution workflow keeps a small sequential Lattice probe
as a safety net. The dedicated `.github/workflows/lattice-research.yml`
workflow is the high-throughput free-compute path: every 15 minutes it plans
eight mathematically stable challengers, evaluates all eight concurrently,
Pareto-reduces them 8 -> 4 -> 2 with increasing scratch-training budgets, and
requires two independent final seeds before the external reducer may update the
research architecture champion. The state is persisted separately on
`lattice-research-state`.

No matrix worker can promote itself. Every empirical comparison still trains a
matched Native Transformer from scratch on the same tokens, split and step
budget, and the final reducer revalidates the winning genome's stability before
persistence.

The swarm also keeps a persistent **elite archive**. A finalist that is not
good enough to become research champion can still be retained when it is
Pareto-useful or wins some independent seeds. Those elites are research-only:
they cannot self-promote, but later cycles may use them as parents and apply a
second structural mutation. This lets AIRI cross local valleys and test
multi-gene combinations such as routing + predictive memory without weakening
the champion gate.

A separate fail-closed migration-readiness gate accumulates evidence for the
current research champion across independent cycles and seeds. It requires
repeated all-seed wins, positive loss margins and active-parameter compliance
before writing `lattice-migration-readiness.json` with
`migration_ready=true`. Even that attestation does **not** rewrite the
canonical model; it only authorizes a later scale-transfer qualification on
larger corpora and budgets.

The canonical Native Transformer therefore remains unchanged by ordinary
Lattice research wins.

The elite archive itself is persisted with the Lattice research state, so failed-but-promising lineages survive across independent scheduled runs instead of being forgotten when a GitHub Actions VM exits.

Useful commands:

```bash
python -m generalist_lm.cli lattice-plan

python -m generalist_lm.cli lattice-population \
  --signal symbolic_reasoning_signal \
  --signal deep_symbolic_signal \
  --count 8

python -m generalist_lm.cli lattice-benchmark \
  ./corpus/native-corpus.json \
  --allowed-root ./corpus \
  --steps 8
```

This remains architecture research on small scratch models. A tiny Lattice
winning a tiny benchmark is evidence about architecture efficiency, not evidence
that AIRI has suddenly become a frontier-scale model.



## Foundation Track v1

The scalable path now has a first-class foundation-model track instead of
treating every local Transformers directory as an anonymous backend.

A foundation candidate must contain a reviewed `airi-foundation-manifest.json`
that declares model identity, source revision, license, architecture, context
window, parameter count/dtype/quantization and the complete protected capability
surface. The manifest is intentionally fail-closed: foundation candidates are
local-files-only and `trust_remote_code` is permanently disabled.

Foundation qualification is separate from both the tiny research benchmark and
generic Transformers qualification. It adds protected long-context and
robustness domains and binds the attestation to three integrity identities:

1. the exact local model-tree digest;
2. the canonical foundation-manifest digest;
3. the current protected foundation-suite digest/version.

Changing weights, tokenizer/config files, provenance metadata or the protected
suite invalidates the attestation automatically. The repository-grounded corpus
explicitly excludes the foundation manifest/governance and foundation benchmark
files, preventing the research learner from training on its own examination.

Typical local workflow:

```bash
python -m generalist_lm.cli foundation-init /models/foundation \
  --model-id reviewed/model \
  --revision COMMIT_OR_REVIEWED_REVISION \
  --license LICENSE_ID \
  --architecture decoder-only \
  --context-length 8192 \
  --parameter-count 7000000000 \
  --dtype bfloat16

python -m generalist_lm.cli foundation-preflight /models/foundation \
  --max-memory-json '{"0":"14GiB","cpu":"32GiB"}'

python -m generalist_lm.cli qualify-foundation /models/foundation \
  --device-map auto \
  --torch-dtype auto \
  --max-memory-json '{"0":"14GiB","cpu":"32GiB"}'
python -m generalist_lm.cli foundation-status /models/foundation
```

The metadata-only preflight runs before Foundation qualification and never
loads model tensors. It validates the local config, declared context window,
quantization metadata, chat-template presence, sharded weight index integrity,
Transformers config support with `trust_remote_code=False`, optional memory
budgets, and a read-only hardware snapshot. Missing shards, shard path escapes,
manifest/config quantization drift, unsupported local config classes and other
hard incompatibilities fail closed.

The preflight also reports protocol-specific requirements. For
`model_type=gpt_oss`, AIRI requires the official `openai-harmony==0.0.8`
runtime in addition to a local chat template. Harmony input is rendered through
the model's Transformers chat template, generation stops on the canonical
Harmony stop-token set, and generated tokens are parsed with the official
Harmony parser. Only the assistant `final` channel is returned as normal user
text; analysis is never surfaced as the answer, while tool-action completions
fail closed until the dedicated Harmony tool-loop adapter is implemented.
MXFP4 candidates additionally report local Accelerate/Triton/kernel and
CUDA-capability facts without downloading any runtime code.

A Foundation candidate is not activated merely by existing on disk. Airi-PC
uses it only when the dedicated Foundation attestation is still exact-digest
valid and activation is explicit:

```bash
export AIRI_GENERALIST_ENABLE=1
export AIRI_GENERALIST_FOUNDATION_MODEL=/models/foundation
export AIRI_GENERALIST_FOUNDATION_DEVICE_MAP=auto
export AIRI_GENERALIST_FOUNDATION_DTYPE=auto
# Optional examples:
# export AIRI_GENERALIST_FOUNDATION_MAX_MEMORY='{"0":"14GiB","cpu":"32GiB"}'
# export AIRI_GENERALIST_FOUNDATION_OFFLOAD=/var/tmp/airi-foundation-offload
```

The Foundation path is deliberately separate from
`AIRI_GENERALIST_TRANSFORMERS_MODEL`. Setting both at once is rejected rather
than silently downgrading to the generic Transformers qualification. With
`device_map=auto` (the Foundation default), Accelerate is required and the
backend never applies a global `model.to(device)` after sharding. Set
`AIRI_GENERALIST_FOUNDATION_DEVICE_MAP=none` for an explicit single-device
load using `AIRI_GENERALIST_DEVICE`.

This still does not make the model a production champion by declaration:
Foundation qualification v4 additionally records the Harmony-aware preflight
v2, while qualification remains digest-bound to the model, manifest, protected
suite and inference dtype. Status re-runs the metadata preflight so a model
whose protocol/runtime support disappears becomes unqualified before provider
activation. The existing router and
provider opt-ins remain in force.
