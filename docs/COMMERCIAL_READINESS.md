# AIRI Commercial Readiness

This document is an engineering checklist, not legal advice.

AIRI can be developed as a commercial product, but the repository must not
pretend that source ownership automatically clears every dependency, training
source, trademark, privacy obligation or redistribution right.

## Current fail-closed rule

Run:

```bash
python scripts/airi-commercial-audit.py --repo .
```

To include persisted Generalist data provenance:

```bash
python scripts/airi-commercial-audit.py --repo . --state-dir /path/to/generalist-state
```

Use `--enforce` in a release pipeline only after the maintainer has chosen an
explicit repository license and completed the human legal review.

The audit checks:

- whether the repository has an explicit LICENSE/COPYING file;
- whether runtime/development dependency manifests are present;
- whether the Generalist data-growth provenance policy exists;
- whether a persisted autodata manifest contains ambiguous license rows;
- whether SECURITY/privacy documentation exists.

## Required before selling or redistributing

1. Choose the repository/product license deliberately. The automated tooling
   does not choose one.
2. Review every third-party dependency and bundled binary license.
3. Review training-data provenance and redistribution/training rights.
4. Define privacy, retention and telemetry behavior for Airi-PC.
5. Review branding/trademark rights.
6. Keep computer-use permissions explicit and auditable.
7. Sign release binaries and publish reproducible hashes.
8. Obtain human legal review for the intended countries and business model.

A green automated audit means only that the engineering prerequisites checked
by the script are present. It is not a legal opinion or a warranty of
commercial clearance.
