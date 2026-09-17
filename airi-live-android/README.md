# Airi Live for Android

A zero-configuration Android dashboard for watching Airi-PC task activity in near real time.

## How it works

- Airi-PC emits **operational metadata only** (task state, step title/operation, job lifecycle).
- The relay is `ntfy.sh`; there is no server for the user to configure or pay for.
- The topic is derived automatically from the canonical repository name + the current `main` commit SHA.
- The app resolves `main`, derives the same topic, and opens ntfy's streaming JSON endpoint.
- When `main` changes, reconnecting the app automatically follows the new channel.

No prompts, file contents, screenshots, command output, cookies, credentials, API keys, or auth tokens are intentionally published by the telemetry layer. The public relay can be disabled with `AIRI_LIVE_TELEMETRY=0`.

## Build

```bash
gradle :app:assembleDebug
```

The repository workflow `.github/workflows/airi-live-android.yml` builds an installable debug APK and uploads it as a GitHub Actions artifact.
