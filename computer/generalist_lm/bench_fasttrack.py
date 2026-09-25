from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig, parameter_count


def bench(name: str, *, layers: int, d_ff: int, context: int, batch: int, steps: int, compile_model: bool = False) -> dict:
    cfg = GeneralistLMConfig(
        vocab_size=384,
        context_length=context,
        d_model=96,
        n_heads=4,
        n_layers=layers,
        d_ff=d_ff,
        dropout=0.0,
        tokenizer_version="bpe-v1",
        ff_variant="swiglu",
    ).validate()
    model = CausalTransformerLM(cfg)
    compile_seconds = 0.0
    if compile_model:
        compile_start = time.perf_counter()
        model = torch.compile(model)
        compile_seconds = time.perf_counter() - compile_start
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(12345)
    x = torch.randint(8, cfg.vocab_size, (batch, context), generator=gen)

    def one_step() -> float:
        opt.zero_grad(set_to_none=True)
        logits = model(x)["logits"]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, cfg.vocab_size),
            x[:, 1:].reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach())

    warmup_start = time.perf_counter()
    for _ in range(2):
        one_step()
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    loss = 0.0
    for _ in range(steps):
        loss = one_step()
    elapsed = time.perf_counter() - start
    trained_tokens = steps * batch * (context - 1)
    return {
        "name": name,
        "parameters": parameter_count(model),
        "layers": layers,
        "d_ff": d_ff,
        "context": context,
        "batch": batch,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens": trained_tokens,
        "tokens_per_second": trained_tokens / elapsed,
        "last_loss": loss,
        "compile_enabled": bool(compile_model),
        "compile_wrapper_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "estimated_1m_total_seconds": warmup_seconds + (1_000_000 / (trained_tokens / elapsed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    rows = [
        bench("eager-current-b32", layers=12, d_ff=1888, context=128, batch=32, steps=args.steps),
        bench("compile-current-b32", layers=12, d_ff=1888, context=128, batch=32, steps=args.steps, compile_model=True),
    ]
    baseline = rows[0]["tokens_per_second"]
    for row in rows:
        row["speedup_vs_current"] = row["tokens_per_second"] / baseline
    print(json.dumps({"ok": True, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
