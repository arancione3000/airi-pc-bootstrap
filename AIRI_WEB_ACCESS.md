# Airi-PC web access

Airi-PC remains the canonical desktop runtime. The GUI browser may be restricted by sandbox network policy.

When a public URL returns ERR_BLOCKED_BY_ADMINISTRATOR:
1. Keep Airi-PC active.
2. Do not bypass the policy.
3. Use the authorized Composio Browser Tool for the Internet step.
4. Use its screenshots as the page visual result.
5. Return to Airi-PC for desktop actions and verification.

Run scripts/airi-web-check to detect whether GUI Chromium has Internet access.

Never use proxies, DNS tricks, VPNs, tunnels, or other policy bypasses.
