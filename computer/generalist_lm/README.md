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
name and shape exactly match the challenger. The transfer report records target
coverage and source tensors that could not be reused. This keeps the rule
conservative: there is no shape coercion or unsafe partial slicing.

The bounded architecture DSL can now explore learned absolute positions,
sinusoidal positions, and RoPE (rotary positional embeddings). RoPE is applied
inside causal attention and is permitted only when the attention head dimension
is even. It remains a challenger option, not a hard-coded claim that RoPE is
always superior.

Persistent curriculum state is fail-closed. Every replay row carries a digest;
invalid domains, malformed SFT messages, duplicate rows, replay-cap violations,
digest mismatches, or protected-validation overlap stop research health and
prevent state persistence.


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
robustness domains and binds the attestation to three independent identities:

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

python -m generalist_lm.cli qualify-foundation /models/foundation
python -m generalist_lm.cli foundation-status /models/foundation
```

This tranche does not automatically make a foundation candidate the active
Airi-PC provider. Provider activation remains a later, separately verified
promotion step.
