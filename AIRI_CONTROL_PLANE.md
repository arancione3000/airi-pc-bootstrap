# Control Plane

`computer/control_plane/orchestrator.py` is the coordinator. It composes the capability manager, transaction engine, task engine, audit engine, project index, maintenance manager, and skill manager. The server exposes them through `computer_control_plane` actions so the original MCP tools remain unchanged.

Supported facade actions include status, bootstrap, route, task operations, transaction operations, audit, index refresh/search, maintenance, skills refresh/verify, and verification recording.


Reliability: capability telemetry and circuit breaker state are persisted in `.ai/control_plane/reliability.json`; the runtime supervisor writes `.ai/control_plane/supervisor.json`.

## Reasoning Engine

`computer/control_plane/reasoning_engine.py` is the higher-level persistent planning/state layer. It coordinates the existing task, coding, verification, evidence, recovery, audit, and model-routing systems without replacing them. The runtime exposes `computer_reasoning_*` MCP tools for lifecycle, observation, feedback, replanning and goal execution.
