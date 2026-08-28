# Airi-PC Supervisor

Il supervisor coordina i controlli di runtime senza diventare un secondo server operativo. Verifica `/ready`, `/status` e i processi principali e può delegare un restart al `computer/start.sh` quando il runtime non è disponibile.

Lo snapshot persistente è `.ai/control_plane/supervisor.json`.

`start.sh` e `airi-next-session` avviano il supervisor in modo idempotente.
