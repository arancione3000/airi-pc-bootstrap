# evolution-lab

## Description
Operate Airi-PC's perpetual **shadow-only Evolution Lab**. It learns from sanitized Control Plane execution outcomes and evolves a compact model that estimates which candidate tool/route is likely to succeed for a task. It is advisory only and is never allowed to execute tools or replace the production router.

## Safety boundary
The lab must remain shadow-only.

- Never use its score to automatically execute, select, block, or reorder production tools.
- Never copy a lab champion into the production router.
- The evolved genome changes only bounded architecture/hyperparameter values; it cannot generate or execute code.
- Training runs CPU-only in a separate process.
- The worker blocks network/DNS connections, child subprocesses, and writes outside `.ai/evolution-lab/shadow-router`.
- The worker receives a minimal environment without API keys or common credential variables.
- Resource use is bounded by architecture limits, short evolution cycles, CPU thread limits and OS resource limits where available.
- Raw traces contain a sanitized goal, operation, selected tool and argument **types/keys only**. Argument values are never written to the training trace.
- The lab only observes completed Control Plane executions. Recording failure must never affect the production operation.

## Continual training
Autopilot is enabled by default on normal Airi-PC scheduler startup unless `AIRI_EVOLUTION_LAB_AUTOPILOT=0` is set or a persistent `autopilot.disabled` marker exists.

The scheduler checks periodically (default 900 seconds). It starts a sandbox training cycle only when enough **new** observations have accumulated (default 20). With no new data the lab remains idle. This is deliberate: repeatedly optimizing forever on an unchanged benchmark would waste resources and overfit evaluation sets.

Use:
- `computer_evolution_lab_status` to inspect observations, champion and worker state.
- `computer_evolution_lab_audit` to verify the sandbox boundary.
- `computer_evolution_lab_score` to compare candidate routes in shadow mode.
- `computer_evolution_lab_start` / `computer_evolution_lab_stop` for manual worker control.
- `computer_evolution_lab_maintenance` for one scheduler-style check.
- `computer_evolution_lab_autopilot` to enable/disable the persistent recurring job.

A score is always advisory. If the lab says a route has 90% estimated success, production Airi-PC still uses its normal router unless a future, separately reviewed feature explicitly changes that policy.
