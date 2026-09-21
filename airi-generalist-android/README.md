# AIRI Generalist Lab for Android

Android chat client for the autonomous AIRI Generalist research model.

## What it does

- downloads the latest mobile model bundle from the repository's `generalist-mobile` branch;
- validates every downloaded file with SHA-256;
- runs inference locally on Android with ONNX Runtime;
- supports the stable **Champion** and the isolated **Latest Research** finalist;
- shows cycle, parameter count, tokenizer size, NLL/byte and generation similarity;
- checks for a new model on launch, manually with **Aggiorna**, and every two minutes while open;
- keeps the last valid local model if GitHub or a new export is temporarily unavailable.

The Latest Research slot is deliberately research-only. It does not bypass the normal
promotion verifier and is never treated as production merely because the Android app can test it.

## Build

```bash
cd airi-generalist-android
gradle :app:testDebugUnitTest :app:assembleDebug
```

The repository workflow publishes an emulator-smoke-tested APK as a GitHub release.
