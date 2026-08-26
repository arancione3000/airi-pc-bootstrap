# coding-task

## Description
Structured multi-step coding workflow built into Airi-PC's existing skill subsystem.

## Instructions
Always start a task with an explicit goal, declared scope and ordered steps. Keep each step in one of: todo, in_progress, done, blocked, failed. Never skip directly from todo to done. Failed verification rolls the edited file(s) back to their task snapshot before another attempt. The project context file must be loaded before changes.

## Tools
task_start
task_read
task_update
task_finish
project_context
scope_check
diff_summary
guardrail_check
