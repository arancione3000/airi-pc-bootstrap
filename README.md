# Airi-PC Bootstrap

Public bootstrap/runtime for Airi-PC Computer Mode. This repository contains only the control runtime and deterministic self-test; private logs, task state, OAuth credentials and private workspace data are intentionally excluded.

## Fresh-session bootstrap
```bash
mkdir -p /home/user/airi
cd /home/user/airi
# download/extract this repository into /home/user/airi, then:
sh computer/start.sh
python3 scripts/airi-agent.py
```

The runtime listens on `127.0.0.1:9010` and exposes the Airi Computer MCP plus the `scripts/airi-control` CLI.

## Verification
Run `python3 scripts/airi-selftest`. A healthy session must end with `ALL= True`.
