# Airi-PC

> Reproducible local Computer Mode runtime with a separate Windows Companion for end users.

Airi-PC is an open-source project by **[@arancione3000](https://github.com/arancione3000)**. It focuses on rebuilding and verifying a local computer-use runtime from a canonical GitHub source instead of relying on unreproducible session state.

## Try Airi-PC

**Windows users:** the easiest way to start is the published **Airi-PC Companion** release.

- **[Download the Windows installer](https://github.com/arancione3000/airi-pc-bootstrap/releases/download/companion-v0.2.1/AiriPC-Companion-Setup.exe)**
- **[Download the portable package](https://github.com/arancione3000/airi-pc-bootstrap/releases/download/companion-v0.2.1/AiriPC-Companion-portable-windows.zip)**
- **[Open the latest Companion release](https://github.com/arancione3000/airi-pc-bootstrap/releases/tag/companion-v0.2.1)**
- **[Read the End User Guide](docs/END_USER.md)**

The Companion is a Windows desktop control surface. It is not the complete Airi-PC core runtime and it does not bundle a public relay or an AI-provider API key.

## What Airi-PC does

At a high level, the project combines:

- a canonical bootstrap/rebuild path from `main`;
- a local HTTP/MCP Computer Mode runtime;
- browser and GUI automation with recovery and verification paths;
- a Control Plane for orchestration, task/job management, persistence, verification and audit;
- a persistent Reasoning Engine integrated with the Control Plane;
- coding/Git integration and repository-level verification;
- a separate Windows Companion for local physical-PC safety controls.

The public repository is the source of the bootstrap/runtime implementation. Runtime-owned state, authentication material and other local state are intentionally kept outside the public source tree.

## What it does not claim

Airi-PC is **not** presented here as a hosted cloud service, a universal autonomous agent, production-ready software, or a guarantee of security against every possible threat. Remote MCP exposure requires its own authenticated transport and deployment controls; the public repository does not provide a bundled public relay.

The current model-routing implementation is intentionally **ChatGPT-only**. It does not provide an active multi-provider fallback system.

## Quick developer setup

The repository has separate dependency entry points:

- `requirements.txt` contains the core runtime dependency surface.
- `computer/requirements.txt` contains the core runtime/test stack used by CI, including **FastAPI**.
- `airi-pc-companion/requirements.txt` contains Companion dependencies.
- `requirements-dev.txt` combines the runtime and Companion test dependencies for a complete local test environment.

From a fresh clone on Linux/macOS:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For the core runtime verification path, see [Developer Guide](docs/DEVELOPER.md). A bare `pytest` in an unprepared Python environment is not considered a valid full-suite verification because required runtime dependencies such as FastAPI may not be installed.

## Core runtime

From the repository root, the main entry points are:

```sh
./scripts/airi-next-session
./scripts/airi-session-rebuild
./scripts/airi-rebuild
./scripts/airi-selftest
./scripts/airi-coding-selftest
./scripts/airi-runtime-verify
```

The canonical local server surface used by the verification workflow is `http://127.0.0.1:9010`, with readiness exposed through `/ready` and status through `/status`.

## Architecture

```text
GitHub main
   |
   v
bootstrap / rebuild scripts
   |
   v
local Airi-PC runtime
   |
   +--> MCP / Computer Mode
   +--> Browser / GUI automation
   +--> Control Plane
   |      +--> Reasoning
   |      +--> Tasks / Jobs
   |      +--> Persistence
   |      +--> Verification / Audit
   |      +--> Recovery / Reliability
   |      +--> Git integration
   |
   +--> Windows Companion (separate local desktop control surface)
```

See [Architecture](docs/ARCHITECTURE.md) for the verified component boundaries.

## Security

The public repository is designed to contain implementation and documentation, not local secrets or runtime state.

Never commit API keys, PATs, bearer/session tokens, passwords, cookies, private keys, OAuth secrets, personal runtime state or local logs. The repository uses `.gitignore`, GitHub Secret Scanning and Push Protection, and the project includes explicit security boundaries.

Read [SECURITY.md](SECURITY.md) before exposing the runtime remotely.

## Documentation

- [End User Guide](docs/END_USER.md) — download and use the Windows Companion.
- [Developer Guide](docs/DEVELOPER.md) — prepare an environment, run tests and verify the runtime.
- [Architecture](docs/ARCHITECTURE.md) — understand the major components.
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common setup and runtime failures.
- [Contributing](CONTRIBUTING.md) — contribution expectations.
- [Chat/bootstrap notes](README_CHAT_BOOTSTRAP.md) — maintainer-oriented session/bootstrap details.

## Demo status

There is currently **no bundled video or screenshot demo** in the repository. Do not infer a demo from test screenshots or runtime logs. A future 30–60 second demo should show, in order:

1. the GitHub landing page and the simple Windows download path;
2. the Companion starting locally;
3. its connection/auth state and safety controls;
4. one representative Airi-PC computer-use action;
5. verification/readiness completing successfully;
6. the repository/release page as the official source.

A short README caption for that future demo can be: **“Airi-PC rebuilds a local Computer Mode runtime from a canonical GitHub source, then verifies the runtime before use.”**

## Releases

The published Windows Companion release is **`companion-v0.2.1`**. Its assets are the installer and portable package linked above. Release assets include SHA-256 digests on GitHub; verify the digest when distributing the binaries through another channel.

## License

This repository currently has **no `LICENSE` file**. No licensing terms have been invented here. Before encouraging external redistribution or reuse, the maintainer should choose and add an explicit open-source license.

## Maintainer

**[@arancione3000](https://github.com/arancione3000)** — creator and maintainer of Airi-PC.

The canonical public source is this repository and its `main` branch.
