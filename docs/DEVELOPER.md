# Airi-PC — Developer Guide

This is the developer/maintainer path for the core runtime and the repository test suite.

## 1. Prerequisites

Use a current Python 3 environment, Git, and a POSIX-compatible shell for the core runtime scripts. Some runtime verification also requires a GUI/browser environment; the canonical CI workflow provisions Playwright Chromium and Xvfb.

## 2. Create the reproducible local environment

From the repository root:

```sh
export AIRIPC_WORKSPACE_ROOT="$PWD"
export AIRI_ROOT="$PWD"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
chmod +x scripts/airi-* computer/start.sh
```

These two workspace variables are important when testing an arbitrary clone: several Control Plane components are also used in the managed `/home/user/airi` runtime and otherwise default there. Binding them to `$PWD` keeps project indexing, transactions, maintenance and related state inside the clone being tested.

`requirements.txt` is the canonical core runtime dependency entry point and includes `computer/requirements.txt`, where FastAPI is declared. `requirements-dev.txt` adds the Companion dependency set so the repository-wide test suite can run.

This is the important distinction from the earlier bare local test invocation: **FastAPI was already declared by the project; the Python environment running pytest simply had not installed the committed dependency set.**

## 3. Start the local runtime before the full suite

Some Control Plane tests intentionally probe `/status` and `/ready`, so a full local suite needs the runtime available.

```sh
DISPLAY_NUM=99 AIRI_BROWSER_HEADLESS=0 sh computer/start.sh
```

Expected readiness is exposed at:

```text
http://127.0.0.1:9010/ready
```

## 4. Run the Python suite

```sh
python -m pytest -q
```

A clean verification run on the current repository should complete the suite after the runtime has started and the workspace variables point to the clone. Running pytest in a completely empty environment, or against a different workspace root, is not a substitute for this setup.

For a syntax-only check:

```sh
python -m compileall -q computer api scripts tests airi-pc-companion
```

## 5. Project verification commands

The verification programs are Python scripts in the current tree, so invoke them with the environment's interpreter:

```sh
python scripts/airi-rebuild-verify.py
python scripts/airi-coding-selftest
python scripts/airi-runtime-verify
python scripts/airi-selftest
```

Shell session/bootstrap entry points can be invoked with `sh`:

```sh
sh scripts/airi-next-session
sh scripts/airi-session-rebuild
sh scripts/airi-rebuild
```

## 6. Dependency model

- `requirements.txt` — canonical core runtime dependency entry point; it includes `computer/requirements.txt`.
- `computer/requirements.txt` — detailed core runtime/test dependencies used by CI, including FastAPI.
- `requirements-dev.txt` — complete repository test environment, combining core and Companion requirements.
- `airi-pc-companion/requirements.txt` — Windows Companion runtime/test dependencies.
- `airi-pc-companion/installer/requirements-build.txt` — packaging/build-time Companion dependencies.

Do not solve missing imports by installing packages globally. Update the appropriate dependency file when a dependency is genuinely part of the project, then verify from a clean virtual environment.

## 7. GitHub Actions

`.github/workflows/airi-runtime.yml` runs the canonical runtime verification on pushes to `main` and on manual dispatch. It installs the runtime dependency set, starts the runtime, executes the Python suite, runs the project self-tests/verifiers, exercises representative browser controls, and restarts/reverifies the server.

`.github/workflows/companion-windows.yml` builds the Windows Companion when a `companion-v*` tag is pushed and publishes the installer and portable archive to a GitHub Release. A manual dispatch runs the packaging path without creating a tag release.

## 8. Contribution rules

Keep changes scoped and reproducible. Do not commit local runtime state, credentials or session data. Update tests when behavior changes and rerun the relevant verification after changes.

The canonical source is always `arancione3000/airi-pc-bootstrap:main`.
