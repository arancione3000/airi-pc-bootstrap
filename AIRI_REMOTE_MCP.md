# Airi-PC Remote MCP

Airi-PC exposes its JSON-RPC MCP endpoint at `/mcp` on the local runtime `127.0.0.1:9010`.

## Persistent remote transport

The preferred public transport is now **Tailscale Funnel**. Tailscale Funnel can expose a local service over HTTPS using the device's stable `*.ts.net` hostname; it does not have the one-hour URL rotation of the historical Pinggy free bridge. Funnel is available on all Tailscale plans, but the machine must have Tailscale installed and authenticated and HTTPS/Funnel must be enabled for the tailnet.

`scripts/airi-tailscale-supervisor` keeps Funnel configured, records the current base URL under `.ai/state/tailscale/`, and recreates the Funnel configuration if it disappears. It never stores Tailscale or ChatGPT credentials in Git.

## OAuth

**Keep the MCP OAuth authorization layer enabled when registering this endpoint in ChatGPT/Composio.** Tailscale provides transport/HTTPS; it is not a replacement for the MCP OAuth authorization flow. The OAuth authorize/token endpoints must be supplied by the MCP authorization layer used by the connector. Do not remove OAuth just because the transport moved from Pinggy to Tailscale.

Airi-PC's local bearer-token middleware remains supported for `/mcp` when `AIRI_MCP_TOKEN` or `.mcp_token` is configured. Never commit tokens, cookies, passwords, or browser auth state.

## Runtime checks

The canonical connector runs:

1. `scripts/airi-next-session`
2. runtime readiness checks
3. `scripts/airi-tailscale-supervisor`
4. `scripts/airi-selftest`

For a remote connection, verify the Tailscale base URL first and then verify `/mcp` with the configured OAuth/authorization layer. Do not treat the existence of port `9010` alone as proof that remote MCP is healthy.
