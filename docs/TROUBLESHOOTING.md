# Airi-PC — Troubleshooting

Use this page before opening an issue. Never include credentials, tokens, cookies, private keys, passwords or private runtime state in an issue.

## `ModuleNotFoundError: No module named 'fastapi'`

This error means the Python environment running `pytest` does not have the project's committed dependency set installed. **FastAPI is already declared by the repository in `computer/requirements.txt`.**

Create a fresh virtual environment and install the repository's developer dependency entry point:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Then, because the full suite includes runtime health checks, start Airi-PC before pytest:

```sh
chmod +x scripts/airi-* computer/start.sh
DISPLAY_NUM=99 AIRI_BROWSER_HEADLESS=0 sh computer/start.sh
python -m pytest -q
```

Do not install FastAPI globally and treat that as a repository fix. The reproducible fix is to use the dependency files committed to the project.

## Python tests fail during collection

First confirm the environment was created from `requirements-dev.txt`. Then run:

```sh
python -m compileall -q computer api scripts tests airi-pc-companion
python -m pytest -q
```

If collection still fails, record the exact import error and the Python version. Do not attach `.venv`, credential files, auth state or private runtime logs.

## Core runtime is not ready

The canonical local endpoints are:

```text
http://127.0.0.1:9010/status
http://127.0.0.1:9010/ready
```

Start the runtime with:

```sh
DISPLAY_NUM=99 AIRI_BROWSER_HEADLESS=0 sh computer/start.sh
```

Then use the project verifiers:

```sh
python scripts/airi-rebuild-verify.py
python scripts/airi-coding-selftest
python scripts/airi-runtime-verify
python scripts/airi-selftest
```

If `/home/user/airi` is missing in a canonical session environment, use the repository bootstrap/rebuild entry points described in the Developer Guide.

## Windows Companion shows offline

1. Confirm the Companion is running.
2. Confirm the intended Airi-PC transport is actually available.
3. Check the Companion's displayed connection/auth state.
4. Remember that the public repository does not bundle a public relay or remote credential.

The Companion is a local safety/control surface, not the entire core runtime.

## Reproducible issue reports

Include:

- Airi-PC commit or Companion release version;
- operating system;
- exact command/action that failed;
- expected result;
- actual result;
- safe diagnostic output.

Do not include authentication material or private runtime state.
