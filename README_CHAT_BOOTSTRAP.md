# Airi-PC chat bootstrap

For a fresh Composio session, run the canonical one-shot entrypoint:

```sh
/home/user/airi/scripts/airi-next-session
```

If `/home/user/airi` is missing, first fetch the public `main` archive from:
`https://github.com/arancione3000/airi-pc-bootstrap/archive/refs/heads/main.tar.gz`
then place it at `/home/user/airi` and rerun the entrypoint.

Canonical runtime: `/home/user/airi`
Server: `http://127.0.0.1:9010`
GUI: `DISPLAY=:99` (`1280x800`)
Control: `/home/user/airi/scripts/airi-control`
Self-test: `/home/user/airi/scripts/airi-selftest`

The browser implementation uses a dedicated single worker thread for Playwright sync objects, bounded retries/timeouts, stale-browser recovery, and structured errors. Runtime-owned state and the Playwright cache are repaired when older sessions left root-owned files and passwordless sudo permits recovery.

**Persistence rule:** an Airi-PC task is not complete until intended changes are committed to `arancione3000/airi-pc-bootstrap:main`, the new commit SHA is recorded, remote HEAD is reread and matches, and important changed paths are verified remotely.
