# Airi Live for Android

Airi Live is the zero-configuration Android viewer for the current Airi-PC
runtime. Its primary purpose is to let the user see the real graphical Airi-PC
desktop while a task is running, even when Airi-PC was reconstructed from a
different ChatGPT conversation.

## Rebuild-safe discovery

- The app and every Airi-PC reconstruction use one stable rendezvous topic
  configured in `.ai/airi_live.json`.
- A reconstruction owns one session ID shared by the runtime, telemetry and POV
  publisher.
- `computer/start.sh` launches the resident live runtime automatically.
- If the public POV tunnel dies, the live runtime recreates it and publishes a
  fresh offer without user setup.
- CI and emulator tests use isolated topics and cannot become the production
  session. The Android client also rejects historical `ci-smoke` events from
  older builds.

## POV transport

The Android app creates an RSA key in Android Keystore and publishes only the
viewer public key. Airi-PC starts a read-only graphical POV endpoint and sends
the temporary viewer descriptor encrypted for that key. Frames are rendered
natively in the app over a persistent MJPEG connection, with frame polling only
as a fallback.

The rendezvous relay is `ntfy.sh`; there is no account or paid server to
configure. Telemetry does not intentionally publish prompts, file contents,
cookies, credentials, API keys or auth tokens. The graphical frames themselves
are transported by the temporary POV endpoint rather than embedded in ntfy
messages.

Set `AIRI_LIVE_TELEMETRY=0` to disable public live announcements.

## Verification

The GitHub Actions workflow `.github/workflows/airi-live-android.yml` builds
the APK and runs an Android-emulator end-to-end test that verifies encrypted
viewer rendezvous, real remote pixels, continuous-stream FPS, visible motion and
full-screen POV before the APK is published.
