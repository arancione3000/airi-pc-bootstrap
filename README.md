# Airi-PC Public Bootstrap

Airi-PC è un runtime Computer Mode riproducibile.

Ogni nuova sessione deve ricostruire il runtime dal branch `main`.

Bootstrap:

```sh
curl -fsSL https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/scripts/airi-session-rebuild | sh
```

Il runtime deve avere:

- GUI 1280x800;
- Chromium;
- Playwright;
- MCP;
- esattamente 83 tool;
- inputSchema MCP per ogni tool;
- Computer Mode;
- coding agent;
- memory;
- task state;
- recovery;
- persistence;
- research;
- scheduler;
- cleanup;
- browser auth/human verification.

La readiness è valida soltanto quando il verifier finale restituisce:

`AIRI_RUNTIME_ALL= True`

Il repository contiene inoltre il coding-agent runtime, il sistema di skill/memory, i servizi di recovery/persistence, ricerca, scheduler e Smart Cleanup. I controlli di sicurezza, scope dichiarato, snapshot/rollback e OAuth/PKCE restano parte del runtime.

Per la diagnosi locale:

```sh
./scripts/airi-selftest
./scripts/airi-coding-selftest
./scripts/airi-runtime-verify
```

Il public bootstrap non deve contenere credenziali, token, cookie, `.venv`, log o stato runtime sensibile.
