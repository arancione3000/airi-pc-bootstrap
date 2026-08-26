# Security policy

Airi-PC Bootstrap is a public repository. Never commit API keys, access tokens, passwords, cookies, private keys, personal data, runtime logs, or session state.

## Local-only files

Keep runtime state, logs, `.ai/` project memory, virtual environments, and credentials outside the repository. The `.gitignore` is configured to exclude common local-secret and runtime patterns.

## Remote MCP

The `/mcp` endpoint must remain authenticated when exposed remotely. Never publish a real bearer token, session token, tunnel credential, cookie, or OAuth secret in source control.

## Reporting

Before publishing changes, run GitHub Secret Scanning/Push Protection and review the tree for local state and credential-like files.
