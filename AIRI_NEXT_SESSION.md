# Airi-PC — Next Session Canonical Procedure

Canonical repository: `arancione3000/airi-pc-bootstrap`
Branch: `main`
Runtime root: `/home/user/airi`
Control: `/home/user/airi/scripts/airi-control`
Bootstrap: `/home/user/airi/scripts/airi-chat-bootstrap`
One-shot: `/home/user/airi/scripts/airi-next-session`
Self-test: `/home/user/airi/scripts/airi-selftest`

## Cold-session sequence
1. Run `scripts/airi-next-session`.
2. It restores the public `main` archive when the runtime is missing.
3. It hydrates local Git metadata and points `origin` at the canonical repository when `.git` is absent.
4. Bootstrap download/extraction happens in a unique temporary directory; an existing workspace is only moved to a timestamped backup after the new archive is fully downloaded and validated.
5. It starts Computer Mode and requires healthy GUI/browser state.
6. It runs the complete self-test and requires `ALL=True` with exit code 0.
7. Only then begin the user task.

## Coding sequence
Before edits, automatically read `AGENTS.md` or `CLAUDE.md`. Create an explicit task/todo and declared scope. Use structured search, snapshot files, edit/test/fix with a maximum of five attempts, review diff/guardrails, then make atomic commits. Record the session in `.ai/PROJECT_MEMORY.md`.

## Browser recovery
All Playwright sync objects belong to one dedicated browser worker thread. Browser launch is retried and bounded by timeouts; dead pages are recreated; browser failures return structured errors instead of uncaught HTTP 500s. Runtime-owned state and Playwright cache permissions are repaired during bootstrap when passwordless sudo permits it.

## Persistence rule
Never declare a coding task complete from a temporary sandbox alone. Persist intended changes to GitHub and verify the resulting remote HEAD and changed paths.

## Never substitute another computer
Do not use Orgo, Daytona, Freestyle or another computer backend as a replacement for Airi-PC.
