# Task Engine

`task_engine.py` stores task graphs in `.ai/control_plane/tasks.json`. Nodes have IDs, dependencies, state, inputs/outputs, errors, retry counts and checkpoints. Completed nodes unlock dependent nodes, allowing later sessions to resume without repeating completed work.
