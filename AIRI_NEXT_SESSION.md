# Airi-PC — Next Session Canonical Procedure

Canonical repository: `arancione3000/airi-pc-bootstrap`
Branch: `main`
Runtime root: `/home/user/airi`
Control: `/home/user/airi/scripts/airi-control`
Bootstrap: `/home/user/airi/scripts/airi-chat-bootstrap`
Self-test: `/home/user/airi/scripts/airi-selftest`
One-shot recovery/verification: `/home/user/airi/scripts/airi-next-session`

## Completion rule
An Airi-PC task is not complete until intended project changes are saved to Git and the resulting repository state is verified. A temporary sandbox test alone is not persistence evidence.

## Cold-session sequence
1. Ensure `/home/user/airi/computer/start.sh` exists.
2. Run `scripts/airi-next-session`.
3. Require runtime, GUI and browser status to be healthy.
4. Run `scripts/airi-selftest`; require `ALL=True` and exit code 0.
5. Execute the requested computer task only after verification.
6. Before reporting completion, verify Git branch HEAD and changed paths remotely.

## Browser recovery
The verified incident was Playwright `cannot switch to a different thread`, combined with runtime workspace/cache files owned by root. Recovery requires restarting Airi-PC Computer Mode after making runtime-owned state writable; never substitute another computer runtime.

## Runtime
`DISPLAY=:99`, expected resolution `1280x800`, server `127.0.0.1:9010`.
