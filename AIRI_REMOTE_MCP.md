# Airi-PC Remote MCP

Airi-PC exposes a JSON-RPC MCP endpoint at `/mcp` on the local runtime (`127.0.0.1:9010`).

For the historical Pinggy free bridge, `scripts/airi-tunnel-supervisor` now maintains the tunnel process, records its current public URL under `.ai/state/tunnel/`, detects expiry/failure, and recreates the tunnel automatically.

Important limitation: Pinggy Free tunnels expire after 60 minutes and a replacement tunnel receives a new URL. Airi-PC can automatically recreate the tunnel, but ChatGPT/Composio must still be given the replacement endpoint unless a persistent endpoint is used. A fixed ChatGPT connector therefore cannot be made permanent by local code alone.

Security: do not publish an unauthenticated Airi-PC MCP endpoint. Prefer authenticated MCP/OAuth or a managed persistent tunnel. Never commit tokens, cookies, passwords, or browser auth state.
