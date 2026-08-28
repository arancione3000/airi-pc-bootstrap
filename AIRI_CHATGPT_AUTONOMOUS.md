# Airi-PC + ChatGPT autonomous workflow

## Secure login takeover

Run `sh scripts/airi-chatgpt-login`. Airi-PC opens the official ChatGPT site in its own browser profile and then pauses for a human to complete authentication. The login page receives the credentials directly; the script never reads, echoes, or stores the password.

The authenticated browser storage is saved only after explicit human confirmation. It is kept under `.ai/auth/` with restrictive permissions and is excluded by `.gitignore`. It is never committed to the public repository.

## Autonomy

ChatGPT/Codex is the planning/programming side when an authenticated product session and the required platform features are available. Airi-PC remains the controlled execution side for files, tests, browser actions, checkpoints, rollback, audit, and Git.

No script attempts to bypass platform authentication or to scrape passwords/cookies from the browser.

## Stop

Run `sh scripts/airi-stop` or issue the control-plane stop command. A stop writes `.ai/STOP`, terminates the local autonomous worker, and preserves the checkpoint. No new autonomous work should start while `.ai/STOP` exists.

## Rate and message discipline

Autonomous work should proceed in bounded iterations rather than an unrestricted message stream. Each iteration must reach a checkpoint before the next one begins. Suggested limits are 20 iterations, 30 minutes, or 5 consecutive failures per task.

## Thinking / Computer mode

Airi-PC may open ChatGPT/Codex, but it must not assume that a UI click successfully enabled platform-level Thinking or Computer Mode. The operator verifies the active mode in the ChatGPT interface. Airi-PC can report the observed page state but does not bypass platform controls.

## Recovery

On browser crash, restart the browser manager and retry the current idempotent step. On Airi-PC process failure, restart the runtime and resume from the last checkpoint. On workspace corruption, use the canonical `airi-rebuild` path and restore persisted state.
