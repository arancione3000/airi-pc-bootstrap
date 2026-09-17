# Airi-PC Autonomy 2026

This release adds four capabilities without removing the existing Airi-PC runtime contract.

## Compatibility model

Airi-PC keeps the immutable **83-tool base manifest** and the existing Control Plane extensions. The public runtime surface grows from 106 to **115 tools** by adding nine 2026 autonomy tools. Existing MCP `2025-06-18` clients continue to work unchanged. MCP `2026-07-28` is additive and stateless.

## Airi Reflex

`computer/control_plane/reflex.py` is a durable event engine. Rules match `source` and `event_type` and dispatch bounded actions through the existing Control Plane. Events are idempotent by event ID, retries are persisted, duplicate delivery is suppressed, and exhausted work is moved to a dead-letter state instead of looping forever.

GitHub can push events to `POST /events/github`. The endpoint requires `AIRI_GITHUB_WEBHOOK_SECRET` and validates the `X-Hub-Signature-256` HMAC before an event is accepted. A valid event is queued and processed automatically; the user does not need to poll GitHub manually.

Supported Reflex actions are deliberately narrow: autonomous goal execution, reasoning-goal execution, and experience recording. This keeps zero-touch automation inside existing Airi-PC guardrails rather than exposing arbitrary webhook-to-shell execution.

## Skill Hunter / ARD

`computer/control_plane/resource_discovery.py` implements Agentic Resource Discovery search using configured ARD registries.

Configuration:

- `AIRI_ARD_REGISTRIES`: comma-separated HTTPS ARD registry base URLs.
- `AIRI_ARD_TRUSTED_PUBLISHERS`: comma-separated publisher IDs allowed for automatic Markdown-skill promotion.

Important trust rule: **ARD relevance score is never treated as trust or safety**. Discovery and trust are separate. Remote endpoints must resolve to public IP addresses, redirects are blocked, response sizes are bounded, and candidates are quarantined first.

Airi-PC never executes arbitrary code downloaded by Skill Hunter. Automatic installation is limited to Markdown skill documents from explicitly trusted publishers after static validation. Installed skills are refreshed into the existing Skill Manager immediately. Unknown publishers remain quarantined for later inspection.

## Airi Judge

`computer/control_plane/judge.py` is an independent verifier. It copies the target project into a temporary sandbox and runs verification there, so verification commands cannot dirty the live workspace. It supports bounded command checks, file existence/content checks, and JSON-key checks. Executables are allowlisted; shell execution is not used.

The Judge also inspects live Git state before and after verification and emits a SHA-256 attestation over the result. A PASS therefore means the requested checks passed in the isolated copy and the Judge itself did not mutate the live workspace.

## MCP 2026-07-28

`computer/control_plane/mcp2026.py` adds the stateless 2026 protocol alongside the legacy route.

Modern requests use request `_meta` with the MCP protocol version and client capabilities, plus HTTP routing headers. Airi-PC validates `Mcp-Protocol-Version`, `Mcp-Method`, and `Mcp-Name` where a principal name exists. The server implements:

- `server/discover`;
- modern `tools/list` and `tools/call`;
- durable Tasks for the canonical public `computer_autonomous_goal`;
- `tasks/get`, `tasks/update`, and `tasks/cancel`;
- `subscriptions/listen` with task-status notifications;
- per-request `resultType` and server metadata.

Durable task state is stored under `.ai/control_plane/` and can resume after a runtime restart.

## New public tools

1. `computer_reflex_status`
2. `computer_reflex_rule_add`
3. `computer_reflex_emit`
4. `computer_reflex_process`
5. `computer_resource_discover`
6. `computer_resource_stage`
7. `computer_resource_acquire_skill`
8. `computer_resource_staged`
9. `computer_judge`

## Verification

The canonical GitHub Actions runtime workflow rebuilds `/home/user/airi` from `main`, compiles all sources, runs the full pytest suite, starts the real runtime, exercises legacy and modern MCP, validates the signed GitHub event path, invokes the sandbox Judge, executes the coding self-test, runs the rebuild verifier, and restarts the runtime before verifying it again.
