# Airi-PC — Developer Guide

This is the developer/maintainer path for the core runtime and the repository test suite.

## 1. Prerequisites

Use a current Python 3 environment, Git, and a POSIX-compatible shell for the core runtime scripts. Some runtime verification also requires a GUI/browser environment; the canonical CI workflow provisions Playwright Chromium and Xvfb.

## 2. Create the reproducible local environment

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` is the convenience entry point for the full repository test environment. It includes `computer/requirements.txt` (where FastAPI is declared for the runtime) and `airi-pc-companion/requirements.txt` for Companion tests.

This is the important difference from the earlier bare local test invocation: **FastAPI is already declared by the project; it simply must be installed into the environment used to run pytest.**

## 3. Run the Python suite

```sh
python -m pytest -q
```

For a syntax-only check:

```sh
python -m compileall -q computer api scripts tests airi-pc-companion
```

A compile-only pass is useful but is not equivalent to the runtime verification.

## 4. Runtime verification

The canonical project commands are:

```sh
./scripts/airi-selftest
./scripts/airi-coding-selftest
./scripts/airi-rebuild-verify.py
./scripts/airi-runtime-verify
```

The local runtime normally uses:

- server: `http://127.0.0.1:9010`
- status: `/status`
- readiness: `/ready`
- GUI display in CI: `DISPLAY=:99`

## 5. GitHub Actions

`.github/workflows/airi-runtime.yml` runs the canonical runtime verification on pushes to `main` and on manual dispatch. It installs the runtime dependency set, starts the runtime, executes the Python suite, runs the project self-tests/verifiers, exercises representative browser controls, and restarts/reverifies the server.

`.github/workflows/companion-windows.yml` builds the Windows Companion when a `companion-v*` tag is pushed and publishes the installer and portable archive to a GitHub Release.

## 6. Dependency model

- `requirements.txt` — minimal core dependency entry point (currently points to the core FastAPI runtime requirement set).
- `computer/requirements.txt` — canonical core runtime/test dependencies used by CI.
- `airi-pc-companion/requirements.txt` — Windows Companion test/runtime dependencies.
- `airi-pc-companion/installer/requirements-build.txt` — packaging/build-time Companion dependencies.
- `requirements-dev.txt` — convenience file for developers who want to run the repository-wide suite locally.

Do not solve missing imports by installing packages globally. Update the appropriate dependency file when a dependency is genuinely part of the project, then verify from a clean virtual environment.

## 7. Contribution rules

Keep changes scoped and reproducible. Do not commit local runtime state, credentials or session data. Update tests when behavior changes and rerun the relevant verification after changes.

The canonical source is always `arancione3000/airi-pc-bootstrap:main`.
