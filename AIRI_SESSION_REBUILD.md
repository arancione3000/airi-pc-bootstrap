# Airi-PC session rebuild

## Purpose

This repository is the canonical, reproducible source for the Airi-PC runtime. A new ChatGPT/Composio session should **rebuild the runtime from `main`** instead of relying on a previous machine, browser process, tunnel, or conversation.

## One-command rebuild

```sh
curl -fsSL https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/scripts/airi-session-rebuild | sh
```

The script downloads the current `main` archive, recreates `/home/user/airi`, installs Python/Playwright dependencies, starts the canonical connector, waits for `/ready`, and runs `airi-selftest`.

By default `AIRI_PRESERVE_STATE=1` preserves `.ai` state from an existing runtime. Set `AIRI_PRESERVE_STATE=0` for a completely fresh runtime.

## Canonical chain

`airi-session-rebuild` -> `airi-connect` -> `airi-next-session` -> `computer/start.sh` -> MCP on `127.0.0.1:9010`.

The runtime verifies:

- GUI `DISPLAY=:99`, 1280x800
- browser/Playwright
- MCP initialization and tool listing
- source revision consistency
- readiness and self-test
- local watchdog/recovery services

## Remote transport

Airi-PC also contains the tunnel supervisors used by the historical remote setup. Pinggy Free is treated as a fallback only because its public hostname can expire/change. Tailscale Funnel support is included as the preferred stable transport when Tailscale is authenticated and Funnel is enabled.

The MCP authorization layer uses OAuth 2.0/PKCE endpoints. Transport and authorization are separate: moving to Tailscale does **not** remove OAuth.

## Session model

This approach intentionally creates **one isolated Airi-PC runtime per ChatGPT/Composio session**. The runtime itself has no one-hour Airi timeout. However, the surrounding remote sandbox/session can still be stopped by its hosting platform; when a new session is created, run the rebuild script again.

This is the expected recovery path:

```text
new ChatGPT session
  -> download main
  -> rebuild Airi-PC
  -> verify /ready + self-test
  -> connect MCP/OAuth
  -> continue from .ai state when preserved
```

Never claim Airi-PC is connected until `computer_status` (or the canonical equivalent) is actually executed.

## Future-session instruction

When a future session cannot see the old Airi-PC tool, do not substitute another computer. Rebuild from `main` with `scripts/airi-session-rebuild`, verify the runtime locally, then discover/register the resulting MCP endpoint through the available Composio/ChatGPT MCP path.
