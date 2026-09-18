from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import runtime


def emit(value, code: int = 0):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(code)


def _run(args):
    from .engine import EvolutionConfig, run_evolution
    overrides = {
        "population": args.population,
        "generations": args.generations,
        "candidate_epochs": args.candidate_epochs,
    }
    cfg = EvolutionConfig.for_mode(args.mode, **overrides)
    runtime._json_write(runtime.STATUS, {"state": "running", "pid": os.getpid(), "mode": cfg.mode, "started_at": time.time()})
    try:
        result = run_evolution(runtime.STATE, cfg)
        runtime._json_write(runtime.STATUS, {"state": "completed", "pid": os.getpid(), "finished_at": time.time(), "result": result})
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        runtime._json_write(runtime.STATUS, {"state": "failed", "pid": os.getpid(), "finished_at": time.time(), "error": repr(exc)})
        print(json.dumps({"ok": False, "error": repr(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    finally:
        try:
            if runtime.PID.exists() and runtime.PID.read_text().strip() == str(os.getpid()):
                runtime.PID.unlink()
        except Exception:
            pass


def parser():
    p = argparse.ArgumentParser(prog="airi-evolve", description="Airi-PC neuroevolution / NAS engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("status")
    h = sub.add_parser("history"); h.add_argument("--limit", type=int, default=10)
    pred = sub.add_parser("predict"); pred.add_argument("text")
    ing = sub.add_parser("ingest")
    ing.add_argument("--text", required=True); ing.add_argument("--label", required=True)
    ing.add_argument("--source", default=""); ing.add_argument("--evidence", default="")
    ing.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    ing.add_argument("--no-auto", action="store_true"); ing.add_argument("--trigger-samples", type=int, default=runtime.DEFAULT_TRIGGER)
    start = sub.add_parser("start")
    start.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    start.add_argument("--population", type=int); start.add_argument("--generations", type=int); start.add_argument("--candidate-epochs", type=int)
    sub.add_parser("stop")
    run = sub.add_parser("_run")
    run.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    run.add_argument("--population", type=int); run.add_argument("--generations", type=int); run.add_argument("--candidate-epochs", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cmd == "setup": emit(runtime.setup(), 0)
    if args.cmd == "status": emit(runtime.status(), 0)
    if args.cmd == "history": emit(runtime.history(args.limit), 0)
    if args.cmd == "predict":
        result = runtime.predict(args.text); emit(result, 0 if result.get("ok") else 2)
    if args.cmd == "ingest":
        result = runtime.ingest({"text": args.text, "label": args.label, "source": args.source, "evidence": args.evidence}, auto_evolve=not args.no_auto, mode=args.mode, trigger_samples=args.trigger_samples)
        emit(result, 0)
    if args.cmd == "start":
        result = runtime.start(mode=args.mode, population=args.population, generations=args.generations, candidate_epochs=args.candidate_epochs)
        emit(result, 0 if result.get("ok") else 2)
    if args.cmd == "stop": emit(runtime.stop(), 0)
    if args.cmd == "_run": return _run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
