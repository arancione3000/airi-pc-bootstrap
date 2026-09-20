from __future__ import annotations

import argparse
import json
import os

from .qualification import (
    qualify_checkpoint,
    qualify_transformers_model,
    qualification_status,
    transformers_qualification_status,
)
from .research_cycle import run_research_cycle
from .research_health import research_health


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIRI Generalist LM research and qualification CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status")
    status.add_argument("--state", default=os.environ.get("AIRI_GENERALIST_STATE", ".ai/generalist-lm/champion"))

    qn = sub.add_parser("qualify-native")
    qn.add_argument("state")
    qn.add_argument("--minimum-score", type=float, default=85.0)

    qt = sub.add_parser("qualify-transformers")
    qt.add_argument("model_dir")
    qt.add_argument("--attestation")
    qt.add_argument("--minimum-score", type=float, default=85.0)

    ts = sub.add_parser("transformers-status")
    ts.add_argument("model_dir")
    ts.add_argument("--attestation")

    rc = sub.add_parser("research-cycle")
    rc.add_argument("--state", default=os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research"))

    rh = sub.add_parser("research-health")
    rh.add_argument("--state", default=os.environ.get("AIRI_GENERALIST_RESEARCH_STATE", ".ai/generalist-research"))
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "status":
        result = qualification_status(args.state)
    elif args.cmd == "qualify-native":
        result = qualify_checkpoint(args.state, minimum_score=args.minimum_score)
    elif args.cmd == "qualify-transformers":
        result = qualify_transformers_model(
            args.model_dir,
            attestation_path=args.attestation,
            minimum_score=args.minimum_score,
        )
    elif args.cmd == "transformers-status":
        result = transformers_qualification_status(
            args.model_dir,
            attestation_path=args.attestation,
        )
    elif args.cmd == "research-cycle":
        result = run_research_cycle(args.state)
    elif args.cmd == "research-health":
        result = research_health(args.state)
    else:  # pragma: no cover
        raise AssertionError(args.cmd)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok", result.get("qualified", True)) is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
