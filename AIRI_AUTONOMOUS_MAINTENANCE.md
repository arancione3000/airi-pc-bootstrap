# Autonomous Maintenance

`maintenance.py` probes disk, server process, display, MCP status and readiness. Results are persisted. The intended recovery ladder is retry, component restart, component repair, runtime rebuild, then escalation. Destructive recovery remains outside automatic maintenance unless an explicit guarded operation is used.
