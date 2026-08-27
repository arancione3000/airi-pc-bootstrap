# Airi-PC Persistence Rule

**Mandatory:** never declare an Airi-PC coding/fix task complete merely because a temporary runtime passes tests.

Before completion:
1. Save intended source, script and documentation changes to `arancione3000/airi-pc-bootstrap:main`.
2. Record the exact commit SHA returned by the GitHub write operation.
3. Re-read remote `main` and verify its HEAD equals that SHA.
4. Re-read every important changed path at that commit.
5. Rebuild from the persisted source and rerun relevant tests when practical.
6. Record the result in `.ai/PROJECT_MEMORY.md`.

If the commit cannot be verified, report **NOT PERSISTENTLY COMPLETE**.

## Runtime enforcement

Direct file-write/patch/terminal mutation endpoints require a declared scope. Mutations create a snapshot and verification record. Remote persistence uses the canonical Git remote and fails closed when push or remote-HEAD verification cannot be completed.
