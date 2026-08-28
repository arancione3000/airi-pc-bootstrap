# Audit Engine

`audit_engine.py` appends structured JSON Lines under `.ai/control_plane/audit.jsonl`. Important records can include goal, plan, decision, tools, inputs, outputs, files, tests, verification and commit. Common token/secret/password/cookie/API-key patterns are redacted before persistence.
