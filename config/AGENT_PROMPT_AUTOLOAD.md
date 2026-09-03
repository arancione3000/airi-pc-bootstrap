# Agent prompt autoload

At bootstrap, load these files in order:

1. `config/AIRI_SYSTEM_MANIFEST.json`
2. `config/AIRI_CHATGPT_ONLY.json`
3. `config/DEFAULT_REASONING_DIRECTIVE.md`
4. `config/AGENT_PROMPT_CONFIG.json`

Then verify runtime status, readiness, MCP tools, source SHA, and the declared task scope before executing work.
