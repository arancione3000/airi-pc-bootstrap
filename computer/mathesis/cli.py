from __future__ import annotations

import argparse
import json

from .engine import MathesisOmega


def main() -> int:
    parser = argparse.ArgumentParser(description="MATHESIS-Ω proof-gated mathematical agent")
    parser.add_argument("prompt", nargs="*", help="request to formalize/solve/prove/synthesize")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    engine = MathesisOmega()
    if args.status:
        output = engine.status()
    else:
        if not args.prompt:
            parser.error("provide a prompt or --status")
        output = engine.answer(" ".join(args.prompt))
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if output.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
