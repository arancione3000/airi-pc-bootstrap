from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from generalist_lm.model import CausalTransformerLM, GeneralistLMConfig, parameter_count


def bench(name: str, *, layers: int, d_ff: int, context: int, batch: int, steps: int) -> dict:
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
        opt.step()
        return float(loss.detach())

    for _ in range(2):
        one_step()

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
        bench("current-12x1888-c128-b32", layers=12, d_ff=1888, context=128, batch=32, steps=args.steps),
        bench("current-12x1888-c128-b64", layers=12, d_ff=1888, context=128, batch=64, steps=args.steps),
        bench("current-12x1888-c128-b128", layers=12, d_ff=1888, context=128, batch=128, steps=max(2, args.steps // 2)),
        bench("wide10-10x2304-c128-b32", layers=10, d_ff=2304, context=128, batch=32, steps=args.steps),
        bench("wide10-10x2304-c128-b64", layers=10, d_ff=2304, context=128, batch=64, steps=args.steps),
        bench("wide10-10x2304-c256-b32", layers=10, d_ff=2304, context=256, batch=32, steps=args.steps),
        bench("wide10-10x2304-c256-b16", layers=10, d_ff=2304, context=256, batch=16, steps=args.steps),
    ]
    baseline = rows[0]["tokens_per_second"]
    for row in rows:
        row["speedup_vs_current"] = row["tokens_per_second"] / baseline
    print(json.dumps({"ok": True, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
