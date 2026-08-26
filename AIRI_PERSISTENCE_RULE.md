# Airi-PC Persistence Rule

**Mandatory:** never declare an Airi-PC coding/fix task complete merely because a temporary runtime passes tests.

Before completion:
- save intended source, script and documentation changes to the canonical Git repository;
- record the resulting commit SHA;
- re-read the branch HEAD from GitHub and verify it equals the saved commit;
- verify important changed paths exist at that commit;
- rerun the relevant Airi-PC self-test after restoring from persisted source when practical.

If the commit cannot be verified, report the task as **not persistently completed**.

Canonical target: `arancione3000/airi-pc-bootstrap:main`.
