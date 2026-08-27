# human-in-the-loop

## Description
Safe handling of CAPTCHA and human-verification pages without bypassing site protection.

## Instructions
When a human-verification challenge is detected, stop autonomous interaction with the challenge. Report the current page URL and require manual completion in the Airi-PC browser. Use the human status and wait controls to pause and poll until the challenge disappears. Resume the original workflow only after the browser reports no active human challenge. Never attempt to solve, evade, or automate a CAPTCHA.

## Tools
- computer_browser_human_status
- computer_browser_human_wait
- computer_browser_open
- computer_browser_state
