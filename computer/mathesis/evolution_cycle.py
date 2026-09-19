from __future__ import annotations

import json
import os

from .curriculum import MathematicalCurriculum
from .discovery import ConjectureDiscoveryEngine
from .evolution import SelfEvolutionEngine


def main() -> int:
    evolution = SelfEvolutionEngine()
    champion = evolution.load_champion()

    discovery_engine = ConjectureDiscoveryEngine(
        evolution.state_dir,
        counterexample_radius=champion.counterexample_radius,
        symbolic_depth=champion.symbolic_depth,
        discovery_beam=champion.discovery_beam,
    )
    discovery = discovery_engine.discover_once()

    discovery_status = discovery_engine.status()
    base_study_every = max(1, int(os.environ.get("MATHESIS_STUDY_EVERY", "12")))
    study_every = max(1, base_study_every // max(1, champion.research_budget))
    study = None
    if discovery_status["cycle"] % study_every == 0:
        # Web study is intentionally non-fatal: offline periods must never stop
        # mathematical evolution or state persistence.
        try:
            study = MathematicalCurriculum(evolution.state_dir).study_once()
        except Exception as exc:
            study = {"ok": False, "status": "research_error", "error": repr(exc)}

    result = evolution.evolve_once()
    output = {
        "ok": True,
        "discovery": discovery,
        "discovery_status": discovery_engine.status(),
        "study": study,
        "evolution": result.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
