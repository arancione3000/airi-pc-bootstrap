# Airi-PC Public Bootstrap

Minimal public bootstrap/runtime for Airi-PC Computer Mode.

It provides the verified desktop runtime, GUI control, screenshots, OCR/observe, mouse/keyboard actions, browser state, MCP and deterministic self-test.

## Fresh-session bootstrap

```sh
mkdir -p /home/user/airi
# extract this repository into /home/user/airi
sh /home/user/airi/computer/start.sh
python3 /home/user/airi/scripts/airi-agent.py
```

A healthy runtime reports `ready: true` and `gui_available: true`; the full self-test ends with `ALL= True`.

## Internet access

Run `sh scripts/airi-web-check`. If it reports `gui_web_access=true`, use the Airi-PC GUI browser. If it reports `gui_web_access=false`, keep Airi-PC as the desktop runtime and use the authorized Composio Browser Tool for public web navigation and page screenshots. See `AIRI_WEB_ACCESS.md`. Never bypass network administration with proxies, DNS tricks, VPNs or tunnels.

The public bootstrap contains no credentials, tokens, personal task history, or private repository state.
