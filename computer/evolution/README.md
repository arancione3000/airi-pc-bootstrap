# Airi-PC NeuroEvolution

This optional subsystem evolves compact text architectures for **reliability estimation / likely fake vs likely real**. PyTorch remains optional for the core Airi-PC runtime and is installed only when evolution is activated.

## End-to-end pipeline

```text
real verified data (LIAR bootstrap + ClaimReview)
                    |
                    v
             verified.jsonl
                    |
          +---------+---------+
          |                   |
          v                   v
      online queue        population NAS
          |                   |
  Airi multi-source       mutate/crossover
      research                |
          |                   v
 ClaimReview consensus   train + evaluate
          |                   |
          +---------> champion/challenger
                              |
                     multi-seed vote gate
                              |
                              v
                         champion model
                              |
                    predict / report / export
```

The model never turns its own prediction into a training label.

## Real-data bootstrap

`airi-evolve bootstrap-liar` downloads LIAR at runtime instead of redistributing it in the repository. Only strong binary labels are used:

- `true`, `mostly-true` -> real
- `false`, `pants-fire` -> fake
- `half-true`, `barely-true` -> excluded

LIAR is research-use-only and its original sources retain copyright.

## Online fact-check learning

`airi-evolve factcheck "<claim>"` reuses Airi-PC's existing multi-source research. The verifier looks for Schema.org `ClaimReview` metadata and requires at least two independent domains with matching, unambiguous verdicts before the claim can be ingested.

Ambiguous ratings such as mixed, half-true, misleading, unsupported or missing-context are not coerced into binary labels. Conflicting sources leave the claim quarantined.

Every accepted online claim keeps the source domains and ClaimReview evidence in the dataset record.

## Evolution and promotion

Genomes can mutate:

- Conv1D, self-attention, GRU and MLP blocks;
- block count and ordering;
- widths and activations;
- residual connections;
- convolution kernels and attention heads;
- dropout;
- learning rate, weight decay and batch size;
- mutation/structural-mutation rates in experimental mode.

Search fitness combines macro-F1, accuracy, calibration, parameter count and measured CPU latency. Deployment is more conservative: the finalist is independently retrained several times and must win a majority of champion-promotion votes.

Normalized duplicate texts cannot leak between train/validation/test, and contradictory labels for the same normalized text are rejected.

## Autopilot

The bootstrap enables the Airi-PC scheduler job `neuroevolution-maintenance` by default. It performs a lightweight periodic check (default hourly). Training starts only if enough new verified records are waiting, so idle machines do not continually retrain.

```sh
./scripts/airi-evolve autopilot
./scripts/airi-evolve autopilot --interval-seconds 7200
./scripts/airi-evolve autopilot --disable
```

## Commands

```sh
./scripts/airi-evolve setup
./scripts/airi-evolve bootstrap-liar
./scripts/airi-evolve status
./scripts/airi-evolve pipeline-status
./scripts/airi-evolve factcheck "claim to check"
./scripts/airi-evolve queue-list
./scripts/airi-evolve queue-verify <id> --url <factcheck-1> --url <factcheck-2>
./scripts/airi-evolve predict "text"
./scripts/airi-evolve start --mode safe
./scripts/airi-evolve start --mode experimental
./scripts/airi-evolve maintenance
./scripts/airi-evolve report
./scripts/airi-evolve export --torchscript
./scripts/airi-evolve history
./scripts/airi-evolve stop
```

Airi-PC also exposes the same operations as native MCP tools named `computer_evolution_*`, so agents do not need to shell out.

## Edge/export

`export` creates an integrity-checked ZIP containing the champion genome, state dict, metrics, provenance and SHA-256 manifest. `--torchscript` additionally attempts a traced model; tracing failure does not destroy the normal bundle and is reported in the manifest.

The classifier remains a learned reliability estimator, not proof of truth. Consequential claims should be shown together with the external fact-check evidence.


## V4: uncertainty, drift and measured edge optimization

Champion predictions now support an abstention state. By default a prediction below 0.65 confidence returns `uncertain` while still exposing the underlying binary preference and probabilities. The threshold is stored with champion provenance and can evolve independently from the model code.

Drift monitoring evaluates only verified examples that arrived **after** the current champion was promoted. It compares recent macro-F1, Brier score and class balance against the champion baseline. A drift signal plus enough new verified examples can trigger evolution earlier than the normal sample-count threshold.

```sh
./scripts/airi-evolve drift
./scripts/airi-evolve predict "claim"
```

Edge optimization is measured, not assumed. `edge-quantize` applies dynamic INT8 quantization to supported Linear/GRU layers, then benchmarks serialized size, CPU latency, probability delta and binary-label agreement against the float champion. The INT8 model is saved only when quality stays inside the configured tolerance and it demonstrates a real size or latency gain.

```sh
./scripts/airi-evolve edge-quantize
./scripts/airi-evolve predict-edge "claim"
```

Every edge artifact is tied to the current champion `genome_id`. Promoting a new champion automatically deletes the previous edge artifact, and stale INT8 metadata is rejected on load. Autopilot attempts edge optimization once for each new champion; rejected attempts are recorded so the system does not waste time retrying every maintenance cycle.

Native MCP equivalents are available as:

- `computer_evolution_drift`
- `computer_evolution_edge_quantize`
- `computer_evolution_predict_edge`
