# Airi-PC Reasoning Engine

`computer/control_plane/reasoning_engine.py` is the persistent coordination layer above the existing Control Plane. It does not replace the model, coding agent, TaskEngine, verification engine, evidence system, rollback, audit, or model router.

## Architecture

`goal -> ReasoningEngine -> plan/dependencies/state -> ControlPlane -> existing tools/agents -> observations/feedback -> diagnosis/replan -> verification -> DONE`

The model remains responsible for reasoning, interpretation, and proposing plans/strategies. The Reasoning Engine makes that process durable and executable: it owns run state, step state, dependencies, retry counts, feedback, errors, evidence references, lifecycle phase, and completion criteria.

## Lifecycle

Phases are `UNDERSTAND`, `PLAN`, `EXECUTE`, `OBSERVE`, `TEST`, `DIAGNOSE`, `REPLAN`, `VERIFY`, `DONE`, and `FAILED`.

Step states are `PENDING`, `READY`, `RUNNING`, `BLOCKED`, `FAILED`, `COMPLETED`, and `SKIPPED`.

`next_action()` is dependency-aware and never returns an already completed step. `feedback()` records tool/task outcomes. `replan()` classifies failures and either retries or inserts a bounded fallback rather than looping forever. `finish()` refuses completion until all steps are complete/skipped and explicit verification is supplied.

## Persistence

Runtime state is stored at `.ai/control_plane/reasoning.json` through the existing crash-safe `save_json()` store. Public source control ignores this runtime state. Redaction is applied before persistence so secrets such as API keys, passwords, cookies, tokens, and authorization values are not written as reasoning data.

## Control Plane integration

The `ControlPlane` exposes:

- `reasoning_start`
- `reasoning_status`
- `reasoning_next_action`
- `reasoning_observe`
- `reasoning_mark_step`
- `reasoning_replan`
- `reasoning_feedback`
- `reasoning_finish`
- `reasoning_goal`

`reasoning_goal()` creates a reasoning run and delegates actual work to the existing `autonomous_goal()` implementation. Task results are copied back into the reasoning state and the existing audit/task/verification infrastructure remains authoritative for execution.

## MCP

The HTTP/MCP server exposes matching `computer_reasoning_*` tools. They are additive; existing tool names and APIs remain available.

## Testing

`tests/test_reasoning_engine.py` covers goal creation, planning, dependency resolution, next action, observation, feedback, diagnosis/replanning, persistence, invalid input, retry limits, completed-step protection, recovery/fallback, finish gating, and an end-to-end Control Plane execution.
