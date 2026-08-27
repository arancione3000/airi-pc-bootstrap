# Airi-PC Coding Agent

Airi-PC provides a local coding-agent runtime on top of the verified Computer Mode.

## Mandatory workflow
`context -> plan/todo -> analyze/search -> scoped edit -> test -> fix/rollback -> diff summary -> guardrails -> atomic commit -> persistent session log -> remote persistence verification`

Before any coding change, the agent reads the nearest `AGENTS.md` (or `CLAUDE.md`) automatically. Autonomous changes require an explicit declared scope.

## Existing systems to extend
- Skills: `/home/user/airi/skills/*/SKILL.md`; `coding-task` contains the structured todo workflow.
- Project Memory: `.ai/PROJECT_MEMORY.md`; session events are appended here.
- Coding engine: `computer/coding.py`.
- Orchestrator: `computer/code_agent.py`.

## Coding guarantees
- Structured search returns path, line number and matching text.
- Every autonomous change is snapshotted and may run at most 5 repair attempts.
- Failed verification restores the task snapshot automatically.
- Diff summaries and guardrail checks run before commit.
- Git commits stage only declared logical paths; unrelated files are not staged.
- Test/self-test removal or weakening is blocked by default.
- `computer/security.py` and `computer/cleanup.py` changes require explicit opt-in.
- Direct mutation endpoints require a declared scope and snapshot/verification guardrails.
- Successful local changes record persistent recovery checkpoints and structured decisions.
- Browser/GUI security controls must not be removed to make tests pass.

## Persistence
A task is not complete until intended changes are saved to `arancione3000/airi-pc-bootstrap:main`, the resulting commit SHA is known, remote HEAD matches it, and important changed paths are readable from that commit. The runtime exposes `computer_persistence_status` and `computer_persist`; absence of a verified remote is reported as not persistent rather than success.

## Scope
Never modify paths outside the declared task scope. Never commit secrets, runtime cache, logs or virtual environments.
