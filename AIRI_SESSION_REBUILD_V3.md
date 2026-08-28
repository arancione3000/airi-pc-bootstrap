# Airi-PC V3 session reconstruction

Canonical entrypoint:

```sh
/home/user/airi/scripts/airi-next-session
```

Explicit full reconstruction:

```sh
/home/user/airi/scripts/airi-rebuild
```

`airi-rebuild` replaces the runtime from the canonical GitHub branch while preserving runtime-owned `.ai` state, records a source fingerprint, starts the runtime, verifies `/ready`, and runs the canonical runtime verification script. Checkpoints and the final report are stored under `.ai/state` so a later session can audit or resume reconstruction.

This is source-first: a broken or stale local workspace is disposable; the GitHub branch is the authoritative reconstruction source.
