# Airi-PC Autonomous Operating Platform

Airi-PC keeps its existing Runtime and MCP Tool layers and adds a modular Control Plane plus persistent control-plane state. The Control Plane coordinates capability discovery/routing, transactions, resumable task graphs, maintenance, skills metadata, project indexing, audit evidence, and orchestration.

## Layers
- Runtime: GUI, browser, filesystem, terminal and system services.
- Tool layer: the existing 83 MCP capabilities, preserved and routed rather than replaced.
- Control Plane: `computer/control_plane/` modules with one MCP facade, `computer_control_plane`.
- Persistent state: `.ai/control_plane/` plus existing `.ai/state/` and `PROJECT_MEMORY.md`.

## Safety
All mutating workflows remain workspace-scoped. Transactions snapshot files before mutation, task graphs checkpoint progress, audit rows redact common secret fields, and persistence still requires a scoped commit followed by remote verification.
