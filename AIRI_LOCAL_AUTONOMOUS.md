# Airi-PC Autonomous Execution (ChatGPT-only)

Airi-PC does not contain a second reasoning model. **ChatGPT is the sole reasoning authority**; Airi-PC is the execution layer.

Supported architecture:

`ChatGPT reasoning -> Airi-PC tools -> observe/execute -> verify -> persist -> recover/rollback`

Airi-PC may expose terminal, filesystem, browser, GUI, mouse, keyboard, screenshots, observation, OCR, MCP, persistence, verification, rollback, and supervised command execution. These capabilities do not constitute an independent reasoning engine.

## Local runtime rule

The local runtime must never start or contact a second LLM. `scripts/airi-local-autonomous` is a compatibility guard and exits explicitly; it does not start a local model or model gateway.

The compatibility shim in `computer/control_plane/model_gateway.py` is deliberately disabled and returns an error rather than forwarding requests.

## Operational autonomy

Autonomous means ChatGPT can drive Airi-PC tools through the execution layer without a human manually performing each operation. It does **not** mean Airi-PC gains an independent LLM.

## Bootstrap order
1. Load the ChatGPT-only manifest and reasoning directive.
2. Start the execution runtime and GUI/browser surfaces.
3. Verify `/status`, `/ready`, `/tools`, and MCP `tools/list`.
4. Execute the requested task with explicit scope and guardrails.
5. Run tests and verify the deliverable.
6. Persist the verified state and record recovery/rollback evidence.
