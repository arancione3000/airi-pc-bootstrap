# Airi-PC Bootstrap Contract

Airi-PC is ChatGPT-only. The bootstrap provides execution, observation, GUI, browser, filesystem, persistence hooks, verification and recovery. It does not start or call a second model provider.

The canonical private repository is `arancione3000/airi-pc`. A fresh ChatGPT session should use its `AIRI_AGENT_ENTRYPOINT.md` and `config/AIRI_SYSTEM_MANIFEST.json` as the authoritative reconstruction instructions.

The bootstrap must remain safe to use as a runtime base: `computer/start.sh` does not launch a model gateway, and `computer/control_plane/model_gateway.py` is a disabled compatibility shim.
