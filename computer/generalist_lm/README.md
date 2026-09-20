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
