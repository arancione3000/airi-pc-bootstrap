# Capability Manager

`capability_manager.py` maintains `.ai/control_plane/capabilities.json`. Entries track name, category, description, input schema, dependencies, availability, health, last verification, latency, error rate and fallback metadata. Discovery registers tools; probes update reliability; routing scores healthy capabilities and rejects invalidated ones.
