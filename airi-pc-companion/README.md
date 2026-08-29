# Airi-PC Companion

Desktop companion for safe control of the user's physical PC.

## End-user mode
The preferred distribution is the packaged `AiriPC-Companion.exe`, produced by the Windows build workflow. The app starts the local companion service, exposes connection/auth state, and provides STOP ALL and GAME MODE controls.

## Security
- Companion service binds to `127.0.0.1` by default.
- Auth material is stored locally as a verifier; the desktop UI never displays or exports credentials.
- The app does not expose arbitrary shell execution.
- Filesystem deletion is restricted to `test-*` files.
- HIGH_RISK operations require explicit confirmation.
- STOP ALL disables game automation locally.

## Airi transport
The desktop app contains an optional `airi_url` setting for a future authenticated relay. The repository does not invent or bundle a public relay. Until a secure transport is provided, the companion remains local-only.

## Game Agent foundation
`game_agent/` provides generic screen capture, state estimation, action timing, safety control and GameProfile support. It intentionally contains no game-specific exploit, anti-cheat bypass, or unrestricted process control.
