# Airi-PC — Next Session Canonical Procedure

Canonical repository: `arancione3000/airi-pc-bootstrap`
Branch: `main`
Runtime root: `/home/user/airi`
Server: `http://127.0.0.1:9010`
GUI: `DISPLAY=:99`, expected `1280x800`
Control: `/home/user/airi/scripts/airi-control`
Bootstrap: `/home/user/airi/scripts/airi-chat-bootstrap`
One-shot: `/home/user/airi/scripts/airi-next-session`
Self-test: `/home/user/airi/scripts/airi-selftest`

## Required sequence
1. Run `scripts/airi-next-session`; it bootstraps from the canonical GitHub archive when the runtime is missing.
2. It repairs stale root-owned runtime/cache state when passwordless sudo permits it.
3. Require status and browser-state to succeed before a user task.
4. Run `scripts/airi-selftest`; require `ALL=True` and exit code 0.
5. Execute the requested task only after verification.
6. Before declaring the task complete, persist intended changes to `main`, record the commit SHA, re-read remote HEAD, and verify the changed paths.

## Browser guarantee
All Playwright sync API objects are owned by one dedicated browser worker thread. FastAPI request threads never directly reuse Playwright objects. Browser failures are bounded, recoverable, and returned as structured results rather than causing unhandled HTTP 500 failures.

## Never substitute another computer
Airi-PC tasks must use this runtime. Do not replace it with Orgo, Daytona, Freestyle or another computer backend.
