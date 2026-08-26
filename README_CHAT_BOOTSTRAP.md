# Airi-PC chat bootstrap

For a new Composio session, the agent can rebuild Airi-PC without a custom MCP or private GitHub access.

Canonical one-shot bootstrap:

```sh
curl -fsSL https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/scripts/airi-chat-bootstrap | sh
```

Requirements: a session exposing Composio Remote Bash/Sandbox. Composio sessions include the sandbox by default unless disabled. The script downloads the public bootstrap, repairs executable permissions, starts Computer Mode, and checks `/status`.

This does not create a permanent computer; the desktop belongs to the current Composio session. Re-run the bootstrap in each fresh session.
