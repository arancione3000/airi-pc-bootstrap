# Control Plane

`computer/control_plane/orchestrator.py` is the coordinator. It composes the capability manager, transaction engine, task engine, audit engine, project index, maintenance manager, and skill manager. The server exposes them through `computer_control_plane` actions so the original MCP tools remain unchanged.

Supported facade actions include status, bootstrap, route, task operations, transaction operations, audit, index refresh/search, maintenance, skills refresh/verify, and verification recording.
