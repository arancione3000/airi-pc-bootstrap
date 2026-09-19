# Airi Evolution State

This branch is machine-maintained by Airi-PC's Evolution Lab.

The tracked state under `evolution-state/shadow-router/` is deliberately privacy-safe:
- no raw Control Plane observations;
- no original user goal text;
- no argument values;
- only structural route features, success/failure labels, bounded champion artifacts and metrics.

Do not manually merge this branch into `main`.


## Initial synthetic bootstrap

The first schema-v2 training corpus contains 96 synthetic structural route examples (48 success / 48 failure). It contains no user text or argument values and exists only to bootstrap cloud evolution before enough real v2 observations arrive from Airi-PC. Real observations remain the intended long-term learning signal.
