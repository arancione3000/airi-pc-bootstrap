# Airi-PC Reliability Layer

Airi-PC registra metriche per capability e componenti, calcola error rate e latenza media e usa un circuit breaker semplice per evitare di instradare ripetutamente verso una capability in errore.

Stati breaker: `closed`, `open`, `half_open`.

Tre errori consecutivi aprono il breaker; dopo 30 secondi viene consentito un probe half-open. Un successo richiude il breaker.

Lo stato persistente è `.ai/control_plane/reliability.json`, esclusivamente runtime-owned e non destinato al repository pubblico.
