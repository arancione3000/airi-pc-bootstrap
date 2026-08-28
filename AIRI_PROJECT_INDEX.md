# Project Index

`project_index.py` maintains `.ai/control_plane/project-index.json`. It incrementally fingerprints files and indexes Python-style `def` and `class` symbols. Re-running the index skips unchanged files. Search is local and fast, providing useful context without re-reading the repository on every task.
