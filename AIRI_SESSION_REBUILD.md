# AIRI-PC — SESSION REBUILD CONTRACT

Il repository `main` è la sorgente canonica del runtime.

Ogni nuova sessione deve poter ricostruire Airi-PC senza dipendere da:

- vecchie sessioni;
- vecchi processi;
- vecchi browser;
- vecchi tunnel;
- vecchi computer.

Workspace:

`/home/user/airi`

GUI:

`DISPLAY=:99`

Risoluzione:

`1280x800`

MCP:

`127.0.0.1:9010/mcp`

Browser:

Chromium + Playwright

Tool canonici:

83

Fonte:

`AIRI_TOOL_MANIFEST.json`

La readiness deve essere verificata realmente.

Sono obbligatori:

- `airi-selftest`
- `airi-coding-selftest`
- `airi-runtime-verify`
- MCP `initialize`
- MCP `tools/list`
- MCP `computer_status`

Output finale richiesto:

`AIRI_RUNTIME_ALL= True`

Lo stato `.ai` può essere preservato, ma segreti e credenziali non devono mai essere committati.
