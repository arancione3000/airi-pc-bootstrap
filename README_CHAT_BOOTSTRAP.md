# Airi-PC chat bootstrap

For a fresh Composio session, rebuild Airi-PC with one shell command:

```sh
curl -fsSL https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/scripts/airi-chat-bootstrap | sh
```

This needs a session exposing Composio Remote Bash/Sandbox. Composio documents that sessions include the sandbox by default unless it is disabled. The script downloads the public bootstrap, fixes executable permissions, clears a stale local Airi process, starts Computer Mode, and verifies the local runtime.

The runtime is recreated per fresh sandbox/session. This is intentionally separate from the optional authenticated remote MCP endpoint.
