# Airi-PC Reconstruction Architecture

Airi-PC is reconstructed as a deterministic, source-of-truth-driven runtime. GitHub `main` is the canonical source; `/home/user/airi` is the materialized runtime; `.ai/state` is runtime state that must survive source replacement.

## Cross-session contract

1. Fetch the exact canonical branch before starting work.
2. Preserve `.ai/PROJECT_MEMORY.md` and `.ai/state` across source replacement.
3. Record the exact source fingerprint in `.ai/.runtime_source_sha`.
4. Start the GUI/MCP stack only after activation.
5. Verify `/ready` and `scripts/airi-runtime-verify` before declaring the runtime usable.
6. Record checkpoint and report data under `.ai/state` so a later session can resume or audit the reconstruction.

## Separation of concerns

- **Source layer:** GitHub repository and versioned code.
- **Runtime layer:** `/home/user/airi`, Python environment, Xvfb/Openbox, browser, MCP server.
- **State layer:** `.ai/state` and project memory; never treated as disposable source files.
- **Control layer:** `scripts/airi-next-session`, `scripts/airi-rebuild`, `scripts/airi-control`.
- **Evidence layer:** ready response, runtime verification log, rebuild report and checkpoint.

## Planned capability upgrades

The strongest next additions are a capability registry with health/latency scores, a transaction journal for multi-tool operations, a reversible action queue for risky GUI/file mutations, structured task memory with resumable DAGs, and a plugin sandbox for new skills. These should live behind the existing MCP contract rather than scattering orchestration across individual tools.
