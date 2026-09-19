from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .architecture import architecture_report, default_genome, mutate_genome
from .benchmark import TRAINING_PHRASES, evaluate_genome
from .kernel import IntegrityKernel, atomic_json
from .knowledge import default_state_dir
from .neural_graph import GrowingNeuralRouter
from .types import ArchitectureGenome, EvolutionResult


class SelfEvolutionEngine:
    """Transactional champion/challenger self-improvement.

    It rewrites only learned/model architecture state. The verifier and parsing
    kernel are immutable during a cycle and checked by hash before promotion.
    """

    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()
        self.champion_path = self.state_dir / "champion.json"
        self.router_path = self.state_dir / "router.json"
        self.history_path = self.state_dir / "history.jsonl"
        self.kernel = IntegrityKernel()

    def load_champion(self) -> ArchitectureGenome:
        try:
            value = json.loads(self.champion_path.read_text(encoding="utf-8"))
            return ArchitectureGenome.from_dict(value)
        except Exception:
            return default_genome()

    def load_router(self, genome: ArchitectureGenome) -> GrowingNeuralRouter:
        try:
            value = json.loads(self.router_path.read_text(encoding="utf-8"))
            router = GrowingNeuralRouter.from_dict(value)
            if router.hidden_size == genome.neural_hidden:
                return router
        except Exception:
            pass
        router = GrowingNeuralRouter(hidden_size=genome.neural_hidden)
        router.train(TRAINING_PHRASES, epochs=80, lr=0.07)
        return router

    def _append_history(self, row: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def evolve_once(self) -> EvolutionResult:
        before = self.kernel.snapshot()
        champion = self.load_champion()
        champion_router = self.load_router(champion)
        champion_bench = evaluate_genome(champion, champion_router)

        candidate = mutate_genome(champion)
        grow_by = max(1, candidate.neural_hidden - champion_router.hidden_size)
        candidate_router = champion_router.grow(grow_by)
        candidate_router.train(TRAINING_PHRASES, epochs=50, lr=0.05)
        candidate_bench = evaluate_genome(candidate, candidate_router)

        integrity = self.kernel.verify_snapshot(before)
        no_new_critical = len(candidate_bench["critical_failures"]) <= len(champion_bench["critical_failures"])
        meaningful_gain = candidate_bench["score"] > champion_bench["score"] + 0.05
        promoted = bool(integrity["ok"] and candidate_bench["ok"] and no_new_critical and meaningful_gain)

        if not integrity["ok"]:
            reason = "immutable verifier kernel changed during candidate evaluation"
        elif not candidate_bench["ok"]:
            reason = "candidate failed critical proof/program benchmarks"
        elif not no_new_critical:
            reason = "candidate introduced critical regressions"
        elif not meaningful_gain:
            reason = "candidate did not beat champion by the promotion margin"
        else:
            reason = "candidate improved verified benchmark without critical regressions"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        if promoted:
            atomic_json(self.champion_path, candidate.to_dict())
            atomic_json(self.router_path, candidate_router.to_dict())
            selected = candidate
        else:
            if not self.champion_path.exists():
                atomic_json(self.champion_path, champion.to_dict())
            if not self.router_path.exists():
                atomic_json(self.router_path, champion_router.to_dict())
            selected = champion

        row = {
            "at": time.time(),
            "promoted": promoted,
            "reason": reason,
            "champion": champion.to_dict(),
            "candidate": candidate.to_dict(),
            "champion_benchmark": champion_bench,
            "candidate_benchmark": candidate_bench,
            "integrity": integrity,
            "selected": selected.to_dict(),
        }
        self._append_history(row)

        return EvolutionResult(
            promoted=promoted,
            champion=selected.to_dict(),
            candidate=candidate.to_dict(),
            champion_score=float(champion_bench["score"]),
            candidate_score=float(candidate_bench["score"]),
            reason=reason,
            benchmark={
                "champion": champion_bench,
                "candidate": candidate_bench,
                "kernel_integrity": integrity,
            },
        )

    def status(self) -> dict[str, Any]:
        champion = self.load_champion()
        router = self.load_router(champion)
        benchmark = evaluate_genome(champion, router)
        return {
            "ok": benchmark["ok"],
            "champion": architecture_report(champion),
            "benchmark": benchmark,
            "state_dir": str(self.state_dir),
            "self_rewrite_scope": "architecture/router state only",
            "kernel": self.kernel.snapshot(),
        }
