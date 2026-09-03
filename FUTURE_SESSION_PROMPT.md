# AIRI-PC — RICOSTRUZIONE CANONICA FUTURA

Usa esclusivamente:

`arancione3000/airi-pc-bootstrap`

branch:

`main`

Non usare una vecchia macchina.
Non usare una vecchia sessione.
Non assumere che Airi-PC esista già.
Non sostituire Airi-PC con un altro computer.

Per una nuova sessione normale esegui:

```sh
/home/user/airi/scripts/airi-next-session
```

Per una ricostruzione completa dalla sorgente canonica esegui:

```sh
/home/user/airi/scripts/airi-rebuild
```

Il rebuild deve:
ricreare `/home/user/airi` dalla sorgente canonica;
installare tutte le dipendenze Python;
installare Chromium/Playwright;
creare `DISPLAY=:99`;
usare 1280x800;
avviare il server MCP su 127.0.0.1:9010;
caricare `AIRI_TOOL_MANIFEST.json`;
esporre esattamente 83 tool;
fornire inputSchema per tutti i tool;
eseguire `airi-selftest`;
eseguire `airi-coding-selftest`;
eseguire `airi-runtime-verify`;
eseguire realmente `computer_status`.

Non dichiarare Airi-PC pronto finché:
`AIRI_RUNTIME_ALL= True`

NON inferire la readiness dai soli file.
La Computer Mode deve essere realmente funzionante. Il browser deve essere realmente disponibile. MCP deve essere realmente operativo.
