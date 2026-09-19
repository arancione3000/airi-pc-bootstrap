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

- `true` -> real
- `false`, `pants-fire` -> fake
- `mostly-true`, `half-true`, `barely-true` -> excluded as non-binary/ambiguous

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
./scripts/airi-evolve audit
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


## V5: golden canary and provenance-aware replay

A persistent golden canary set is reserved before NAS/training and never enters the training, validation or normal test pools. Challenger promotion now requires both the normal multi-seed gate and the hidden canary gate. The canary IDs persist across later generations so a model cannot gradually train on its own promotion exam.

The training sampler is provenance-aware. When multiple source families exist (for example LIAR bootstrap plus online ClaimReview consensus), sampling weights reduce domination by the largest family while preserving broad replay of older data. This gives new verified evidence meaningful training influence without replacing long-term memory.

Final evaluation records per-source-family metrics. Champion provenance also stores source-family counts and canary metadata.

Promotion trials are now auditable artifacts. The exact state dict of each independent trial is saved under the run directory, and if a challenger wins, the deployed `champion/model.pt` is copied from the selected evaluated trial itself. This guarantees that published champion metrics correspond to the actual deployed weights.


## V6: audit hardening and long-running stability

The evaluation partitions are now persistent across generations. The first train/validation/test assignment is written to a manifest, and later verified records are assigned deterministically without moving older records between splits. This prevents a sample that was once held out for testing from silently becoming training data in a later evolution.

The golden canary set is frozen after its first creation. It can no longer grow by taking older records that a previous champion may already have seen.

LIAR bootstrap ingestion is batched: the existing JSONL dataset is indexed once per batch rather than rescanned for every row. This removes the previous quadratic import path.

Verified dataset writes use a cross-process lock with stale-lock recovery, preventing concurrent MCP/HTTP fact-check requests from racing while appending online training data.

Promotion trial storage is bounded. Only the selected trial weights are retained for the current run, loser state dicts are deleted, and old selected-trial weights are pruned while JSON metrics/history remain.

ClaimReview source independence is stricter: subdomains of the same registrable domain count as one organization, and HTTP redirects are revalidated before following so a public URL cannot redirect the verifier into localhost or a private network. Verdict normalization now conservatively recognizes common Italian and other European true/false labels while still excluding ambiguous ratings.

Process start/stop is platform-aware on Windows and POSIX.

Use the built-in self-audit at any time:

```sh
./scripts/airi-evolve audit
```

The audit checks dataset label conflicts, duplicate rows, canary/split overlap, missing/stale split assignments, champion completeness, edge/champion genome consistency, stale dataset locks, invalid queue files and redundant promotion-trial weights. The same operation is available to Airi as `computer_evolution_audit`.


### INT8 backend compatibility note

PyTorch is moving quantization development from legacy `torch.ao.quantization` APIs to TorchAO. Airi-PC currently keeps `torch.ao.quantization.quantize_dynamic` as a compatibility backend because this project still supports PyTorch 2.7+, while newer TorchAO releases target newer PyTorch bases. Every INT8 attempt records the active backend, PyTorch version and migration target (`torchao.quantization.quantize_`) so the backend can be migrated explicitly when the project's minimum PyTorch version is raised.


## Shadow Evolution Lab: perpetual route learning without production control

Airi-PC now has a second, deliberately isolated use of neuroevolution: the **Shadow Evolution Lab**. It learns from the Control Plane's real execution outcomes and estimates which candidate tool is likely to succeed for a task.

This is useful because Airi-PC already accumulates a continual stream of real examples such as:

```text
sanitized goal + operation + selected tool + argument schema
                                ↓
                         success / failure
                                ↓
                    isolated evolutionary NAS
                                ↓
                  advisory route-success scores
```

The lab is intentionally not connected to production routing. Its score is never used by `ControlPlane.route()` to select, execute, reorder or block tools. There is no lab-to-production promotion command.

### Perpetual, but not wasteful

The scheduler keeps the lab enabled across restarts by default. Every 15 minutes it checks for new observations. A training process starts only after enough new observations have accumulated (default 40). With no new evidence the lab remains idle rather than repeatedly fitting the same benchmark.

This provides indefinite continual learning while avoiding permanent CPU usage and repeated overfitting.

### Isolation

The worker runs in a separate CPU-only process with several independent barriers:

- the evolved genome can mutate only bounded architecture/hyperparameter fields;
- no generated code is executed;
- API keys and common credential environment variables are not inherited;
- socket/DNS operations are blocked by a Python audit hook;
- child subprocess creation is blocked inside the worker;
- writes, deletes and renames outside `.ai/evolution-lab/shadow-router` are blocked;
- symlinks escaping the lab directory make the sandbox audit fail;
- on Linux, a network namespace is used as an additional kernel-level barrier when `unshare -n` is available;
- CPU threads, CPU time, address space, file size and file descriptors are bounded where the OS exposes `prlimit`;
- the production router cannot be modified by the lab.

The Python audit hook is defense-in-depth rather than a claim of VM-grade containment. The strongest guarantee comes from the combination of shadow-only architecture, separate process, minimal environment, filesystem boundary, optional network namespace and the fact that the lab has no production execution path.

### Privacy and retention

Training traces do not persist argument values. They contain a sanitized goal, operation, tool name and argument **keys/types**. URLs, emails, filesystem paths, credential-like assignments and long identifiers in goals are redacted.

Long-running storage is bounded:

- raw observations: maximum 50,000 retained;
- training view: latest 20,000 observations;
- lab run directories: maximum 50;
- history: maximum 200 lines;
- worker log: rotated after 10 MiB.

### Native Airi tools

- `computer_evolution_lab_status`
- `computer_evolution_lab_audit`
- `computer_evolution_lab_score`
- `computer_evolution_lab_start`
- `computer_evolution_lab_stop`
- `computer_evolution_lab_maintenance`
- `computer_evolution_lab_autopilot`

`computer_evolution_lab_score` is advisory only. A future production integration would require a separately reviewed change and should not be enabled merely because a lab benchmark looks good.
