# Transaction Engine

`transaction_engine.py` creates file-level snapshots under `.ai/control_plane/transactions/<id>/`, records BEGIN/STEP/COMMIT/ROLLBACK state, and can restore prior contents or remove newly created files. It is designed for reversible workspace mutations where technically possible.
