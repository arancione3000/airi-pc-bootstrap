# Airi-PC agent entrypoint

ChatGPT is the sole reasoning authority. Airi-PC is the execution, verification, persistence, recovery, and rollback layer.

## Bootstrap order
1. Load `config/AIRI_SYSTEM_MANIFEST.json`.
2. Load `config/AIRI_CHATGPT_ONLY.json` and `config/DEFAULT_REASONING_DIRECTIVE.md`.
3. Autoload `config/AGENT_PROMPT_CONFIG.json` via `config/AGENT_PROMPT_AUTOLOAD.md`.
4. Start `computer/start.sh`; keep Playwright under `$ROOT/.cache/ms-playwright`.
5. Verify `/status`, `/ready`, `/tools`, and MCP `tools/list`.
6. Execute operations with scope/guardrails.
7. Run tests and verify the deliverable.
8. Persist the verified state and keep rollback/recovery evidence.

No local or remote LLM is started by the runtime.
