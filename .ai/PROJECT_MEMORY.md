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
