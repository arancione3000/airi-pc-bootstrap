# Airi-PC Project Context

## Purpose
This repository is the canonical Airi-PC public bootstrap and Computer Mode runtime.

## Structure
- `computer/server.py`: FastAPI Computer Mode server, GUI/browser/MCP endpoints.
- `computer/coding.py`: sandboxed filesystem, search, context, scope, snapshot/rollback, diff and local Git helpers.
- `computer/code_agent.py`: coding-agent orchestration, plans, task state, edit/test/fix and commit preparation.
- `computer/skills.py`: existing skill and Project Memory subsystem, extended with task tracking and session logging.
- `computer/cleanup.py`: conservative cleanup and quarantine.
- `computer/security.py`: path and destructive-operation guards.
- `skills/*/SKILL.md`: reusable skill definitions; extend these instead of building a parallel skill system.
- `scripts/airi-*`: bootstrap, control and regression/self-test entrypoints.

## Required workflow
1. Read this file before any coding task.
2. Analyze/read before editing.
3. Declare the task scope before autonomous edits.
4. Snapshot changed files before modifying them.
5. Run edit -> test -> fix with at most five attempts; restore snapshots after failed verification.
6. Review a human-readable diff and guardrails before committing.
7. Use atomic logical commits; never stage unrelated files.
8. Record meaningful task events in `.ai/PROJECT_MEMORY.md`.
9. Persist successful changes to the canonical GitHub repository and verify the remote HEAD before declaring completion.

## Commands
- Runtime: `/home/user/airi/scripts/airi-next-session`
- Control: `/home/user/airi/scripts/airi-control`
- Full regression: `/home/user/airi/scripts/airi-selftest`
- Python syntax: `python3 -m py_compile computer/*.py scripts/*`

## Never
- Do not use Orgo, Daytona, Freestyle or another computer as a replacement for Airi-PC.
- Do not modify files outside the declared task scope.
- Do not delete or weaken tests/self-tests.
- Do not remove, bypass or weaken security/cleanup guardrails.
- Never commit secrets, tokens, cookies, `.venv`, logs or runtime-only state.
