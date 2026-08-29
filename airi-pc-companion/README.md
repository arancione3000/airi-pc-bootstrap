# Airi-PC Companion

Local, permissioned desktop companion for a user's physical computer.

## Security model
- Binds to loopback by default.
- Every `/status` and `/v1/execute` request requires a bearer token.
- The token is stored only as a SHA-256 hash in `auth.json` (0600 where supported).
- Requests expire by timestamp.
- HIGH_RISK and DESTRUCTIVE operations require explicit confirmation metadata.
- Filesystem access is sandbox-scoped; deletion is limited to `test-*` files.
- Shell execution is not exposed by the protocol.
- No credentials, cookies, API keys, or pairing secrets belong in this repository.

## Current capability surface
`system`, `screen`, `mouse`, `keyboard`, `windows`, `applications`, `filesystem`, `processes`, `process_control`.

## Important limitation
The package provides the secure local control endpoint and protocol, but a real physical-PC deployment still requires a reachable authenticated transport from Airi-PC to the companion. The default is loopback-only, intentionally preventing accidental Internet exposure. Do not change the bind address until a secure tunnel/relay is configured.

## Run
Linux/macOS: `./installer/install.sh` then `$HOME/.airi-pc-companion/run.sh`.

Windows PowerShell: `./installer/install.ps1` then run the generated `run.ps1`.

## Pairing
The first start creates a one-time local bearer token in memory (`initial_token`) for provisioning. A production pairing flow should transfer only a short-lived pairing code over an authenticated control channel and store only a verifier/hash locally.
