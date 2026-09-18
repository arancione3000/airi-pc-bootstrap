# Airi-PC NeuroEvolution

This optional subsystem evolves compact text-classification architectures for **reliability estimation / likely fake vs likely real**. It is intentionally isolated from the core Airi-PC runtime: importing Airi-PC does not import PyTorch, and PyTorch is installed only when `airi-evolve setup` or the first evolution run needs it.

## Safety model

The model is not a truth oracle. Predictions are statistical estimates learned from verified examples. Online learning accepts only labelled examples that include an externally verified label; the system never treats its own prediction as ground truth.

A champion/challenger gate protects the deployed model. A challenger is promoted only when it is within the configured macro-F1 regression bound and either improves macro-F1 or is materially smaller/faster at essentially the same macro-F1.

## Search space

Genomes can mutate:

- block type: Conv1D, self-attention, GRU, MLP;
- number and order of blocks;
- widths;
- activation functions;
- residual connections;
- convolution kernels;
- attention head count;
- dropout;
- learning rate, weight decay, batch size;
- mutation and structural mutation rates (strong self-adaptation in experimental mode).

## Commands

```sh
./scripts/airi-evolve setup
./scripts/airi-evolve status
./scripts/airi-evolve ingest --text "..." --label real --source "..." --evidence "..."
./scripts/airi-evolve start --mode safe
./scripts/airi-evolve start --mode experimental
./scripts/airi-evolve predict "claim or article text"
./scripts/airi-evolve history
./scripts/airi-evolve stop
```

Ingestion can automatically launch another safe evolution after enough new verified examples have accumulated. State lives in `.ai/evolution/`, outside the canonical source tree.
