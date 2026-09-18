# neuroevolution

## Description
Evolve and operate Airi-PC's optional lightweight fake-vs-real reliability classifier using a population-based NAS / neuroevolution engine.

## Instructions
Use `scripts/airi-evolve` through the existing terminal tools. Keep verified data and model state under `.ai/evolution/`. Never label an example from the model's own prediction and feed it back as truth. For online improvement, ingest only externally verified labels with source/evidence when available. Prefer `safe` mode by default. Use `experimental` mode only when broader architecture search and extra compute are acceptable. Check `status` after starting a run, and use `history` to compare F1, parameters and latency across promotions.

The classifier estimates learned reliability; it does not prove factual truth. For important claims, combine its output with Airi-PC browser research and primary-source verification.

## Tools
- computer_terminal_run
- computer_terminal_start
- computer_terminal_status
- computer_terminal_attach
- computer_terminal_cancel
- computer_research
