# Airi-PC Public Bootstrap

Minimal public bootstrap/runtime for Airi-PC Computer Mode.

It provides the verified desktop runtime, GUI control, screenshots, OCR/observe, mouse/keyboard actions, browser state, MCP, deterministic self-test, and Smart Cleanup.

## Fresh-session bootstrap

```sh
curl -fsSL https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/scripts/airi-chat-bootstrap | sh
```

A healthy runtime reports `ready: true` and `gui_available: true`; the full self-test ends with `ALL= True`.

## Smart Cleanup

Airi-PC can analyze disk capacity, current usage, largest user-facing directories, disposable caches/temporary files/logs, old downloads, and duplicate files. Automatic cleanup is deliberately conservative: only low-risk disposable items are removed automatically. Old downloads and duplicates are review-only.

See `AIRI_SMART_CLEANUP.md`.

## Internet access

Run `sh scripts/airi-web-check`. If it reports `gui_web_access=true`, use the Airi-PC GUI browser. If it reports `gui_web_access=false`, keep Airi-PC as the desktop runtime and use the authorized Composio Browser Tool for public web navigation and page screenshots. Never bypass network administration with proxies, DNS tricks, VPNs or tunnels.

The public bootstrap contains no credentials, tokens, personal task history, or private repository state.

## Coding agent

Airi-PC includes a coding-agent runtime with project analysis, file search/read/write/patch, terminal execution, test/build/lint, Git inspection/commit, persistent project memory and a skill system. See `AIRI_CODING_AGENT.md`.
