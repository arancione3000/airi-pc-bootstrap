# Airi-PC Persistence Rule

**Mandatory:** never declare an Airi-PC task complete merely because a temporary runtime passes tests.

Before completion:
1. Save intended source, script and documentation changes to the canonical Git repository.
2. Record the resulting commit SHA.
3. Re-read the branch HEAD from GitHub and verify it equals the saved commit.
4. Verify every important changed path exists at that commit.
5. For runtime/browser changes, recreate from the persisted source and rerun the relevant self-test when practical.

No GitHub commit verification = **not persistently completed**.

Canonical target: `arancione3000/airi-pc-bootstrap:main`.
Canonical runtime: `/home/user/airi`.
