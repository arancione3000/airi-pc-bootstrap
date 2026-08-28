# Airi-PC Local Autonomous Development

Airi-PC can run an autonomous development loop without a paid OpenAI API by using a local Ollama-compatible model.

## Architecture

`goal -> local model -> unified diff -> git apply --check -> apply -> tests -> retry with feedback -> commit`

The loop is bounded by `AIRI_AUTONOMOUS_ITERATIONS` (default 5). Patch paths are constrained to the repository and sensitive locations are rejected. The model never receives secrets automatically.

## Configuration

- `AIRI_ROOT=/home/user/airi`
- `AIRI_LOCAL_MODEL=qwen2.5-coder:7b`
- `AIRI_OLLAMA_URL=http://127.0.0.1:11434/api/chat`
- `AIRI_AUTONOMOUS_TEST='python3 -m pytest -q'`
- `AIRI_AUTONOMOUS_ITERATIONS=5`

This mode has **zero API usage cost**. It does require local CPU/RAM (or a local GPU) and disk space for the model. Ollama/model installation itself does not require an API subscription.

## Usage

Run from Airi-PC:

```sh
scripts/airi-local-autonomous "Improve the browser screenshot recovery tests"
```

Use `--iterations N` to bound the run further or `--no-commit` to leave a verified worktree without creating a Git commit.

## Startup safety

`airi-next-session` remains the runtime bootstrap. The local agent is not started automatically on every session; autonomy is explicitly enabled by invoking `airi-local-autonomous`. This avoids a background agent modifying the repository without a concrete goal.

## Relation to ChatGPT

This free mode deliberately does **not** automate the ChatGPT website or reuse a ChatGPT login session as an API. The browser may still be used for ordinary Airi-PC tasks. The autonomous software-engineering loop is local and provider-agnostic.
