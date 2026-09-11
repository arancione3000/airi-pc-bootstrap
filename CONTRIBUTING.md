# Contributing to Airi-PC

Thanks for helping improve Airi-PC. The goal is to keep the public project reproducible, understandable and honest about what has actually been verified.

## Before opening an issue

Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) first.

For bugs, include a reproducible description and safe diagnostic output. Do not attach credentials, cookies, tokens, private keys or private runtime state.

## Pull requests

Prefer focused changes. A good PR should explain:

- what changed;
- why it changed;
- how it was tested;
- any known limitations.

## Development environment

Use the repository dependency entrypoints:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Do not treat a bare global `pytest` invocation as the project environment. `FastAPI` is a runtime dependency and `pytest` is a developer/test dependency; both are installed by `requirements-dev.txt` through the runtime and test dependency chain.

## Verification

Use the checks relevant to your change. For core Python/runtime work:

```sh
python -m compileall -q computer api scripts tests airi-pc-companion
python -m pytest -q
python scripts/airi-rebuild-verify.py
./scripts/airi-runtime-verify
```

For Windows Companion changes, also run the Companion tests and exercise the packaging workflow when practical.

## Documentation accuracy

Do not add claims such as `production ready`, `fully autonomous` or `secure` unless the repository contains evidence that supports them. Keep user-facing documentation aligned with the implementation and tests.

## Security

Never commit secrets or local runtime state. Read [SECURITY.md](SECURITY.md) before changing authentication, remote MCP exposure or transport behavior.
