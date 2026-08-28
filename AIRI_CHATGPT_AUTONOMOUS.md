# Airi-PC + ChatGPT/Codex autonomous development

Airi-PC can open the official ChatGPT/Codex web experience in its own browser. Authentication is performed manually by the user in the browser; passwords, cookies, tokens, and browser storage are never committed to the public repository.

## Start

Run:

`sh scripts/airi-chatgpt-start`

This opens `https://chatgpt.com/codex` using the existing Airi-PC browser. Complete any sign-in in the browser itself. Do not paste credentials into chat or project files.

## Autonomous development model

ChatGPT/Codex is the programming/review brain. Airi-PC remains the controlled execution environment: code changes, tests, browser actions, checkpoints, rollback and Git are mediated by Airi-PC.

The development loop is bounded and checkpointed rather than an uncontrolled message stream. A typical cycle is:

`goal -> analysis -> change -> test -> verification -> commit -> next iteration`

Airi-PC should not execute new work after an emergency stop is requested.

## Emergency stop

Run:

`sh scripts/airi-stop`

This creates `.ai/STOP`, asks the autonomous worker to terminate, and preserves the current checkpoint. To resume later, remove the stop flag only after inspecting the saved state.

## Security

The repository is public, so these paths are deliberately ignored by Git:

- `.ai/auth/`
- `.ai/STOP`
- `.ai/control_plane/local-agent-checkpoint.json`

No script in this feature reads or writes ChatGPT passwords. Browser authentication stays inside the browser session. Private session state is never treated as project source.

## Thinking / Computer mode

Airi-PC may open the official ChatGPT/Codex page, but it must not assume that a UI click successfully enabled a platform-level feature. The operator should verify the active mode in the ChatGPT interface. The bridge does not attempt to collect credentials or bypass platform authentication.

## Cost

This browser bridge does not require an OpenAI API key. Actual ChatGPT/Codex usage remains subject to the limits and availability of the user's ChatGPT plan.
