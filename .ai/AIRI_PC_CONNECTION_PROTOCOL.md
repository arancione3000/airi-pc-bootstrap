# Airi-PC automatic connection protocol

Canonical trigger: when the user says `usa Airi-PC`, execute this procedure whenever Remote Bash is available.

## A — normal path
Run `/home/user/airi/scripts/airi-auto-connect`.

The wrapper first runs `scripts/airi-next-session`, which:
- fetches the canonical `main` archive to a file (never `curl | sh`);
- verifies/synchronizes the Git repository;
- preserves `.ai/PROJECT_MEMORY.md` when possible;
- reuses the runtime when `/ready` responds;
- otherwise starts/restarts the runtime;
- checks status, browser state and the Airi-PC self-test.

## B — automatic recovery
If A exits non-zero, immediately run `scripts/airi-session-rebuild`.

The rebuild fetches the canonical GitHub `main` archive to a file, replaces the runtime while preserving `.ai` state when possible, reinstalls dependencies, starts the runtime, waits for `/ready`, then runs the self-tests.

## Final gate
A connection is reported as successful only when:
- `http://127.0.0.1:9010/ready` responds successfully;
- `airi-control status` reports `ok: true` and `gui_available: true`;
- browser status/state checks succeed;
- `airi-selftest` succeeds.

If both A and B fail, report the actual failure and do not claim Airi-PC is connected.

## Canonical sources
- Bootstrap repository: `arancione3000/airi-pc-bootstrap`
- Branch: `main`
- Runtime: `/home/user/airi`
- Local server: `http://127.0.0.1:9010`
