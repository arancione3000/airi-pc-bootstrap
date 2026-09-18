# neuroevolution

## Description
Operate Airi-PC's autonomous lightweight reliability classifier: verified-data acquisition, ClaimReview fact-checking, population-based NAS/neuroevolution, champion promotion, progress reporting and edge export.

## Instructions
Use the dedicated evolution commands/tools rather than editing model files manually.

On first real-data setup, run `scripts/airi-evolve bootstrap-liar --mode safe`. The importer downloads LIAR at runtime, keeps only unambiguous labels (true/mostly-true vs false/pants-fire), records provenance and starts evolution automatically. LIAR is research-use-only; do not commit or redistribute its downloaded dataset.

For an online claim, prefer the one-shot `scripts/airi-evolve factcheck "<claim>"`. It uses Airi-PC's existing multi-source research, queues the claim, looks for structured ClaimReview metadata, requires at least two independent domains with the same unambiguous verdict, ingests only verified consensus and triggers a new evolution when the verified-data threshold is reached.

If automatic research cannot establish consensus, leave the claim pending. Use Airi-PC browser/research to collect additional fact-check URLs, then call `queue-verify`. Never turn the classifier's own prediction into a training label. Never force ambiguous verdicts such as mixed, half-true, misleading, unsupported or missing-context into the binary dataset.

Use `safe` mode by default. Use `experimental` only when broader architecture search and additional compute are acceptable. Champion promotion is multi-seed and majority-gated. Conflicting verified labels are quarantined instead of learned.

Use `report` for macro-F1, per-class F1, parameter count, model size and CPU latency history. Use `export --torchscript` for a portable champion bundle; if TorchScript tracing is unsupported for the evolved architecture, keep the regular state-dict bundle and report the tracing error.

The output is a learned reliability estimate, not proof of factual truth. For consequential claims, surface the supporting fact-check sources to the user.

## Tools
- computer_evolution_status
- computer_evolution_bootstrap_liar
- computer_evolution_factcheck
- computer_evolution_predict
- computer_evolution_report
- computer_evolution_export
- computer_evolution_queue_add
- computer_evolution_queue_list
- computer_evolution_queue_verify
- computer_evolution_start
- computer_evolution_stop
- computer_evolution_maintenance
- computer_research
- computer_browser_open
- computer_browser_text
