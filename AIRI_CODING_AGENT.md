# Airi-PC Coding Agent

Airi-PC now exposes a local coding-agent runtime on top of the verified Computer Mode. The model remains the planner/reasoner; Airi-PC provides project inspection, file edits, terminal/test/build/lint execution, Git inspection, skills and project memory.

## Core workflow

`analyze -> read -> patch/write -> test -> verify -> diff -> commit`

Failed verification should trigger rollback before another attempt. Risky shell commands require `allow_shell=true`.

## Project tools

- `computer_project_analyze`
- `computer_project_tree`
- `computer_file_read`
- `computer_file_search`
- `computer_file_write`
- `computer_file_patch`
- `computer_terminal_run`
- `computer_test_run`
- `computer_build_run`
- `computer_lint`
- `computer_git_status`
- `computer_git_diff`
- `computer_git_log`
- `computer_git_commit`
- `computer_code_apply_fix`
- `computer_code_verify_change`
- `computer_code_agent`

## Skills

Skills are stored under `/home/user/airi/skills/<name>/SKILL.md` and can be listed, loaded, created, updated, tested and deleted. Deletions are backed up under `.ai/skill-trash`.

## Project memory

Long-lived project notes are stored in `.ai/PROJECT_MEMORY.md`.

## Safety

The coding layer is intentionally conservative: workspace paths are sandboxed to Airi-PC, risky shell patterns are blocked unless explicitly allowed, and failed verified patches roll back to the previous file content.
