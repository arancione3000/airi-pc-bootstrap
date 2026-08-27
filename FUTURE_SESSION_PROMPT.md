# FUTURE SESSION PROMPT — AIRI-PC FROM GITHUB

Use the canonical repository only:
`arancione3000/airi-pc-bootstrap`, branch `main`.

Do not assume the previous Airi-PC machine or conversation still exists.
Do not use Orgo, Daytona, Freestyle, Hyperbrowser, or another computer.

First execute:
```sh
curl -fsSL https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/scripts/airi-session-rebuild | sh
```

Then verify:
1. `/home/user/airi` exists.
2. `http://127.0.0.1:9010/status` is healthy.
3. `http://127.0.0.1:9010/ready` returns `ready:true` and all checks true.
4. `airi-selftest` passes.
5. MCP `initialize` and `tools/list` work.
6. Execute the real `computer_status` tool if exposed.

Keep OAuth/PKCE enabled for the remote MCP connection.
If a previous tunnel is dead, do not reuse its old URL. Use the current runtime's transport state and the available Composio/ChatGPT MCP registration path.

Do not say “Airi-PC connected” until a real computer-status call succeeds.
