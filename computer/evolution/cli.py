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
    sub.add_parser("pipeline-status")
    boot = sub.add_parser("bootstrap-liar")
    boot.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    boot.add_argument("--no-evolve", action="store_true")
    boot.add_argument("--no-autopilot", action="store_true")
    qa = sub.add_parser("queue-add")
    qa.add_argument("claim")
    ql = sub.add_parser("queue-list")
    ql.add_argument("--status")
    ql.add_argument("--limit", type=int, default=100)
    qv = sub.add_parser("queue-verify")
    qv.add_argument("id")
    qv.add_argument("--url", action="append", required=True)
    qv.add_argument("--min-sources", type=int, default=2)
    qv.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    qv.add_argument("--no-auto", action="store_true")
    qv.add_argument("--trigger-samples", type=int, default=runtime.DEFAULT_TRIGGER)
    rep = sub.add_parser("report")
    rep.add_argument("--history-limit", type=int, default=20)
    fc = sub.add_parser("factcheck")
    fc.add_argument("claim")
    fc.add_argument("--max-sources", type=int, default=8)
    fc.add_argument("--min-sources", type=int, default=2)
    fc.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    fc.add_argument("--no-auto", action="store_true")
    maint = sub.add_parser("maintenance")
    maint.add_argument("--mode", choices=("safe", "experimental"), default="safe")
    ap = sub.add_parser("autopilot")
    ap.add_argument("--disable", action="store_true")
    ap.add_argument("--interval-seconds", type=int, default=3600)
    exp = sub.add_parser("export")
    exp.add_argument("--out")
    exp.add_argument("--torchscript", action="store_true")
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
    if args.cmd == "pipeline-status": emit(runtime.pipeline_status(), 0)
    if args.cmd == "bootstrap-liar":
        result = runtime.bootstrap_liar(auto_evolve=not args.no_evolve, mode=args.mode, enable_autopilot=not args.no_autopilot)
        emit(result, 0 if result.get("ok") else 2)
    if args.cmd == "queue-add": emit(runtime.queue_add(args.claim), 0)
    if args.cmd == "queue-list": emit(runtime.queue_items(args.status, args.limit), 0)
    if args.cmd == "queue-verify":
        result = runtime.queue_verify(args.id, args.url, min_sources=args.min_sources, auto_evolve=not args.no_auto, mode=args.mode, trigger_samples=args.trigger_samples)
        emit(result, 0 if result.get("status") == "verified" else 2)
    if args.cmd == "report": emit(runtime.report(args.history_limit), 0)
    if args.cmd == "factcheck":
        result = runtime.factcheck(args.claim, max_sources=args.max_sources, min_sources=args.min_sources, auto_evolve=not args.no_auto, mode=args.mode)
        emit(result, 0 if result.get("ok") else 2)
    if args.cmd == "maintenance": emit(runtime.maintenance(mode=args.mode), 0)
    if args.cmd == "autopilot": emit(runtime.autopilot(not args.disable, args.interval_seconds), 0)
    if args.cmd == "export":
        result = runtime.export(args.out, include_torchscript=args.torchscript)
        emit(result, 0 if result.get("ok") else 2)
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
