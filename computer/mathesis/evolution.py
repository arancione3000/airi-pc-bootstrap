from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .architecture import architecture_report, default_genome, generate_challengers
from .benchmark import TRAINING_PHRASES, evaluate_genome
from .kernel import IntegrityKernel, atomic_json
from .knowledge import default_state_dir
from .model_writer import write_model_module
from .neural_graph import INTENTS, GrowingNeuralRouter
from .types import ArchitectureGenome, EvolutionResult


class SelfEvolutionEngine:
    """Transactional champion/challenger architecture evolution.

    The mutable object is a bounded architecture DSL + learned router state.
    Mathematical/verifier kernel files remain immutable during every promotion.
    Multiple challengers are evaluated independently on each cycle.
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

    def _fresh_router(self, genome: ArchitectureGenome) -> GrowingNeuralRouter:
        router = GrowingNeuralRouter(hidden_size=genome.neural_hidden)
        router.train(TRAINING_PHRASES, epochs=90, lr=0.07)
        return router

    def load_router(self, genome: ArchitectureGenome) -> GrowingNeuralRouter:
        try:
            value = json.loads(self.router_path.read_text(encoding="utf-8"))
            router = GrowingNeuralRouter.from_dict(value)
            if (
                router.hidden_size == genome.neural_hidden
                and router.output_size == len(INTENTS)
            ):
                return router
        except Exception:
            pass
        # Intent-vocabulary or topology migrations deliberately retrain rather
        # than attempting to reinterpret incompatible old output weights.
        return self._fresh_router(genome)

    def _candidate_router(
        self,
        champion_router: GrowingNeuralRouter,
        candidate: ArchitectureGenome,
        trial: int,
    ) -> GrowingNeuralRouter:
        if (
            champion_router.output_size == len(INTENTS)
            and candidate.neural_hidden >= champion_router.hidden_size
        ):
            delta = candidate.neural_hidden - champion_router.hidden_size
            router = champion_router.grow(delta) if delta else GrowingNeuralRouter.from_dict(champion_router.to_dict())
        else:
            router = GrowingNeuralRouter(
                hidden_size=candidate.neural_hidden,
                seed=champion_router.seed + 101 + trial,
            )
        router.train(TRAINING_PHRASES, epochs=55, lr=0.05)
        return router

    def _append_history(self, row: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > 2000:
                self.history_path.write_text("\n".join(lines[-2000:]) + "\n", encoding="utf-8")
        except Exception:
            pass

    def evolve_once(self) -> EvolutionResult:
        before = self.kernel.snapshot()
        champion = self.load_champion()
        champion_router = self.load_router(champion)
        champion_bench = evaluate_genome(champion, champion_router)

        requested = int(os.environ.get("MATHESIS_CHALLENGERS", "3"))
        challengers = generate_challengers(
            champion,
            weaknesses=champion_bench.get("weaknesses", []),
            count=max(1, min(6, requested)),
        )

        trials: list[dict[str, Any]] = []
        router_by_id: dict[str, GrowingNeuralRouter] = {}
        for index, candidate in enumerate(challengers):
            candidate_router = self._candidate_router(champion_router, candidate, index)
            candidate_bench = evaluate_genome(candidate, candidate_router)
            router_by_id[candidate.genome_id] = candidate_router
            trials.append({
                "trial": index,
                "genome": candidate.to_dict(),
                "benchmark": candidate_bench,
            })

        selected_trial = max(trials, key=lambda row: float(row["benchmark"]["score"]))
        candidate = ArchitectureGenome.from_dict(selected_trial["genome"])
        candidate_bench = selected_trial["benchmark"]
        candidate_router = router_by_id[candidate.genome_id]

        self.kernel.validate_state_path(self.state_dir, self.state_dir / "candidate_model.py")
        write_model_module(self.state_dir / "candidate_model.py", candidate)

        integrity = self.kernel.verify_snapshot(before)
        no_new_critical = len(candidate_bench["critical_failures"]) <= len(champion_bench["critical_failures"])
        meaningful_gain = candidate_bench["score"] > champion_bench["score"] + 0.02
        promoted = bool(integrity["ok"] and candidate_bench["ok"] and no_new_critical and meaningful_gain)

        if not integrity["ok"]:
            reason = "immutable verifier kernel changed during candidate evaluation"
        elif not candidate_bench["ok"]:
            reason = "candidate failed critical proof/program benchmarks"
        elif not no_new_critical:
            reason = "candidate introduced critical regressions"
        elif not meaningful_gain:
            reason = "best candidate did not beat champion by the promotion margin"
        else:
            reason = "best candidate improved verified benchmark without critical regressions"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        if promoted:
            atomic_json(self.champion_path, candidate.to_dict())
            atomic_json(self.router_path, candidate_router.to_dict())
            self.kernel.validate_state_path(self.state_dir, self.state_dir / "champion_model.py")
            write_model_module(self.state_dir / "champion_model.py", candidate)
            selected = candidate
        else:
            if not self.champion_path.exists():
                atomic_json(self.champion_path, champion.to_dict())
            if not self.router_path.exists():
                atomic_json(self.router_path, champion_router.to_dict())
            if not (self.state_dir / "champion_model.py").exists():
                self.kernel.validate_state_path(self.state_dir, self.state_dir / "champion_model.py")
                write_model_module(self.state_dir / "champion_model.py", champion)
            selected = champion

        row = {
            "at": time.time(),
            "promoted": promoted,
            "reason": reason,
            "champion": champion.to_dict(),
            "candidate": candidate.to_dict(),
            "selected_trial": selected_trial["trial"],
            "trials": trials,
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
                "trials": trials,
                "selected_trial": selected_trial["trial"],
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
            "self_rewrite_scope": "architecture DSL/router state only",
            "candidate_arena": {"max_challengers": 6, "default_challengers": 3},
            "kernel": self.kernel.snapshot(),
        }
