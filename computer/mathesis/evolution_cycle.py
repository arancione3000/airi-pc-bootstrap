from __future__ import annotations

import json

from .evolution import SelfEvolutionEngine


def main() -> int:
    engine = SelfEvolutionEngine()
    result = engine.evolve_once()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
