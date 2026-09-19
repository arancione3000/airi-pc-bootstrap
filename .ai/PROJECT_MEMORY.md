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
