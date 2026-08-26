# Airi-PC Remote MCP

Airi-PC exposes a JSON-RPC MCP endpoint at `/mcp`.

For remote use, set `AIRI_MCP_TOKEN` to a strong bearer token (or use the local `.mcp_token` file in a controlled runtime). Send `Authorization: Bearer <token>` on `/mcp` requests.

The MCP endpoint exposes 21 Computer Mode tools, including status, observe, screenshot, mouse, keyboard, browser and act/verify.

Security: do not expose an unauthenticated Airi-PC control endpoint to the public internet.
