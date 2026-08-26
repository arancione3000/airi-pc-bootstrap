# Airi-PC Fresh Session

This public bootstrap contains the Airi-PC Computer Mode runtime only.

Fresh session contract:
1. Reconstruct `/home/user/airi` from this public repository.
2. Run `sh computer/start.sh`.
3. Run `python3 scripts/airi-agent.py`.
4. Require `ready: true`, GUI available and MCP reachable.
5. Use `sh scripts/airi-control ...` for desktop control.
6. Run `python3 scripts/airi-selftest` only for diagnostics or certification; a healthy runtime ends with `ALL= True`.

No private task state, logs, OAuth credentials or user workspace data belong in this public repository.
