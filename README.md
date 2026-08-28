# Airi-PC Public Bootstrap

Airi-PC è un runtime Computer Mode riproducibile.

Ogni nuova sessione deve ricostruire il runtime dal branch `main`.

## Bootstrap sicuro

Il bootstrap usa download, verifica e poi esecuzione separata. Non viene mai usata la forma `curl | sh`.

```sh
curl -fL https://github.com/arancione3000/airi-pc-bootstrap/archive/refs/heads/main.tar.gz -o /tmp/airi-bootstrap.tar.gz
sha256sum /tmp/airi-bootstrap.tar.gz
tar -tzf /tmp/airi-bootstrap.tar.gz | head -n 20
/bin/sh scripts/airi-rebuild
```

Il runtime contiene 83 tool base e il facade `computer_control_plane` per l'orchestrazione autonoma.

La readiness è valida soltanto quando il verifier finale restituisce `AIRI_RUNTIME_ALL= True`.

Per la diagnosi locale: `scripts/airi-selftest`, `scripts/airi-coding-selftest`, `scripts/airi-runtime-verify`.

Il public bootstrap non deve contenere credenziali, token, cookie, `.venv`, log o stato runtime sensibile.
