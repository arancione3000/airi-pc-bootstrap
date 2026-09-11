# Airi-PC

> Reproducible local Computer Mode runtime for Airi-PC, with a separate Windows Companion for end users.

**Creator / maintainer:** [@arancione3000](https://github.com/arancione3000)

Airi-PC rebuilds and verifies a local computer-use runtime from the canonical `main` branch instead of depending on unreproducible session state.

## Try Airi-PC

**Windows users:** the quickest path is the packaged **Airi-PC Companion** release.

- **[Download the Windows installer](https://github.com/arancione3000/airi-pc-bootstrap/releases/tag/companion-v0.2.1)**
- **[Download the portable Windows package](https://github.com/arancione3000/airi-pc-bootstrap/releases/tag/companion-v0.2.1)**
- **[Open the release page](https://github.com/arancione3000/airi-pc-bootstrap/releases/tag/companion-v0.2.1)**
- **[Read the End User Guide](docs/END_USER.md)**

There is **no bundled demo video or screenshot yet**. The repository deliberately does not imply a demo exists when it does not; the planned 30–60 second sequence is documented in [docs/DEMO.md](docs/DEMO.md).

## What can Airi-PC do?

At a high level, the project combines:

- a canonical bootstrap/rebuild path from `main`;
- a local HTTP/MCP Computer Mode runtime;
- browser and GUI automation with recovery/verification paths;
- a Control Plane for orchestration, tasks/jobs, persistence, verification and audit;
- a persistent Reasoning Engine integrated with the Control Plane;
- coding/Git integration and repository-level verification;
- a separate Windows Companion for local PC safety controls and connection state.

The public repository is the source for the reproducible bootstrap/runtime implementation. Local runtime state, authentication material and other machine-specific state stay outside the public source tree.

## How it is structured

```text
GitHub / main
     |
     v
bootstrap + rebuild scripts
     |
     v
local Airi-PC runtime
  |       |        |
 MCP   Browser   GUI
  |       |        |
  +-------+--------+
          v
     Control Plane
      |    |    |
 Reasoning Jobs  Verification/Audit
      |
 Persistence / Git

Windows Companion
(separate local control surface)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component boundaries.

## Quick start for developers

From a fresh Linux/macOS clone, use the repository's dependency entrypoints rather than a global Python environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements.txt` is the canonical **runtime** dependency entrypoint. FastAPI belongs there through `computer/requirements.txt` because the core runtime creates FastAPI applications in `computer/server.py`, `computer/contract_server.py` and `api/index.py`.

`requirements-dev.txt` adds repository-wide test/development dependencies such as `pytest` and the Windows Companion test requirements.

Then run the suite after starting the local runtime:

```sh
chmod +x scripts/airi-* computer/start.sh
DISPLAY_NUM=99 AIRI_BROWSER_HEADLESS=0 sh computer/start.sh
python -m pytest -q
```

For the full verification path, follow [docs/DEVELOPER.md](docs/DEVELOPER.md).

## Why the dependency split matters

The previous `ModuleNotFoundError: fastapi` was caused by running `pytest` in an environment that had **not installed the repository's committed dependency set**. FastAPI was already declared by the project; the local environment was incomplete.

The repository now makes the intended model explicit:

- `requirements.txt` → runtime dependencies;
- `requirements-dev.txt` → full developer/test environment;
- `computer/requirements.txt` → detailed core runtime dependencies;
- `airi-pc-companion/requirements.txt` → Companion runtime/test dependencies.

A bare `pytest` in an unprepared Python installation is not a supported substitute for the documented environment setup.

## Security boundaries

Airi-PC is published with implementation and documentation, not local secrets or runtime credentials.

- Do not commit API keys, access tokens, passwords, cookies, private keys, bearer tokens or local runtime state.
- The active model-routing implementation is intentionally **ChatGPT-only**; the project does not ship an active multi-provider fallback router.
- The Windows Companion is a local control surface and does not bundle a public relay.
- The Companion does not expose arbitrary shell execution; destructive filesystem operations are restricted by its implementation and high-risk actions require explicit confirmation.
- The Game Agent foundation is generic and does not implement game-specific exploits or anti-cheat bypasses.

Read [SECURITY.md](SECURITY.md) before exposing the runtime remotely.

## Verification and CI

The canonical GitHub Actions workflow recreates `/home/user/airi` from the current `main` snapshot, installs `requirements-dev.txt`, compiles Python sources, starts the runtime, runs the full pytest suite, executes project verifiers/self-tests, checks representative browser controls, and restarts/reverifies the server.

Useful local checks include:

```sh
python -m compileall -q computer api scripts tests airi-pc-companion
python -m pytest -q
python scripts/airi-rebuild-verify.py
python scripts/airi-coding-selftest
./scripts/airi-runtime-verify
```

## Documentation

- [End User Guide](docs/END_USER.md)
- [Developer Guide](docs/DEVELOPER.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Demo Plan](docs/DEMO.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## Releases

The current packaged Windows Companion release is **`companion-v0.2.1`**. The release page contains the installer and portable package.

For binary redistribution, verify the published SHA-256 digests shown by GitHub Releases.

## Licensing

This repository currently has **no `LICENSE` file**. No licensing terms are being invented here. Before external redistribution or reuse is encouraged, the maintainer should choose and add an explicit open-source license.

## Community

Use **GitHub Issues** for reproducible bugs and concrete changes. Use **GitHub Discussions** for questions, feedback, ideas and project-direction conversations.

Please keep reports safe: never attach credentials, cookies, authentication material or private runtime state.

---

**Airi-PC** is maintained by **[@arancione3000](https://github.com/arancione3000)**. The canonical public source is this repository and its `main` branch.
