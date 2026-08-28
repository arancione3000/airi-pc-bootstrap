# Skill Sandbox

`skill_manager.py` treats `skills/*/SKILL.md` as controlled extensions. The registry stores version, description, required tools, workspace-scoped permissions, checksum, compatibility/status metadata and verification time. Skill code is not allowed to silently rewrite the core; mutation remains subject to existing task scope and guardrails.
