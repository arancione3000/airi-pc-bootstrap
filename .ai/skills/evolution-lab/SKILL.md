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

The scheduler checks periodically (default 900 seconds). It starts a sandbox training cycle only when enough **new** observations have accumulated (default 40). With no new data the lab remains idle. This is deliberate: repeatedly optimizing forever on an unchanged benchmark would waste resources and overfit evaluation sets.

Use:
- `computer_evolution_lab_status` to inspect observations, champion and worker state.
- `computer_evolution_lab_audit` to verify the sandbox boundary.
- `computer_evolution_lab_score` to compare candidate routes in shadow mode.
- `computer_evolution_lab_start` / `computer_evolution_lab_stop` for manual worker control.
- `computer_evolution_lab_maintenance` for one scheduler-style check.
- `computer_evolution_lab_autopilot` to enable/disable the persistent recurring job.

A score is always advisory. If the lab says a route has 90% estimated success, production Airi-PC still uses its normal router unless a future, separately reviewed feature explicitly changes that policy.


## Persistent continuum

The Evolution Lab has a lifecycle independent from the chat/runtime session.

- `computer/evolution/daemon.py` is the resident coordinator.
- On Airi OS it runs as `airi-evolution-lab.service` with `Restart=always` and is not `PartOf=airi-pc.service`.
- On generic bootstrap installations, `computer/start.sh` launches the daemon detached with `nohup`; server restarts do not target it.
- The daemon checks for fresh observations and, while idle, launches another bounded search cycle roughly every 15 minutes.
- Search is continuous by default even on an unchanged dataset. Each individual cycle remains bounded and only one worker runs at a time.
- Promotion thresholds never relax: continuous search may keep discovering candidates indefinitely, but a candidate replaces the champion only if it passes the normal independent promotion gates.
- The daemon syncs state to the dedicated Git branch `airi-evolution-state` and verifies the pushed remote SHA.
- State sync is bidirectional: privacy-safe records are merged, and a better remote/cloud champion can be imported locally.
- A fresh runtime can restore state from the Git branch.

GitHub Actions workflow `.github/workflows/evolution-continuum.yml` runs every hour. It restores `airi-evolution-state`, performs one bounded cloud search cycle when enough data exists, then pushes and verifies the resulting state. The per-dataset cycle limit is `0` by default, meaning continuous search. This means research continues even when the Airi-PC host itself is offline.

### Public-state privacy contract

Only feature schema v2 may be synchronized. V2 contains no user goal words and no argument values or argument names. It contains structural request-shape buckets, internal operation/tool names, candidate tool names, labels and aggregate metrics. Legacy v1 observations are purged before a v2 dataset is rebuilt.

Never export `raw/observations.jsonl`. Every dataset export must pass `state_sync.privacy_audit_dataset`.

Tomorrow/status review can inspect:
- `airi-evolution-state:evolution-state/shadow-router/status.json`
- `airi-evolution-state:evolution-state/shadow-router/manifest.json`
- `airi-evolution-state:evolution-state/shadow-router/cloud-meta.json` when cloud cycles have run
- the branch commit history for verified heartbeats/syncs.
