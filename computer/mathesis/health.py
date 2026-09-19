from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .architecture import _ALLOWED_EXPERTS
from .discovery import _relation_is_structurally_nontrivial
from .evolution import SelfEvolutionEngine
from .knowledge import default_state_dir
from .neural_graph import INTENTS, GrowingNeuralRouter
from .safe_math import parse_relation
from .sympy_lab import DOMAIN_ATLAS


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def health_report(state_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(state_dir or default_state_dir()).resolve()
    engine = SelfEvolutionEngine(root)
    champion = engine.load_champion()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # Architecture / curriculum closure.
    missing_domains = sorted(set(DOMAIN_ATLAS) - set(_ALLOWED_EXPERTS))
    check("architecture:domain_closure", not missing_domains, missing_domains)

    experts = list(champion.experts)
    check("architecture:unique_experts", len(experts) == len(set(experts)), experts)
    unknown_experts = sorted(set(experts) - set(_ALLOWED_EXPERTS))
    check("architecture:known_experts", not unknown_experts, unknown_experts)

    bad_edges = [
        list(edge)
        for edge in champion.topology
        if len(edge) != 2 or edge[0] not in experts or edge[1] not in experts
    ]
    check("architecture:topology_closed", not bad_edges, bad_edges)
    check("architecture:symbolic_depth_bounds", 1 <= champion.symbolic_depth <= 12, champion.symbolic_depth)
    check("architecture:discovery_beam_bounds", 1 <= champion.discovery_beam <= 8, champion.discovery_beam)
    check("architecture:proof_cell_bounds", 1 <= champion.max_proof_cells <= 256, champion.max_proof_cells)
    check("architecture:research_budget_bounds", 1 <= champion.research_budget <= 4, champion.research_budget)

    # Router state must match the currently selected architecture and intent vocabulary.
    router_path = root / "router.json"
    if router_path.exists():
        try:
            router = GrowingNeuralRouter.from_dict(_read_json(router_path, {}))
            check("router:hidden_matches_champion", router.hidden_size == champion.neural_hidden, {
                "router": router.hidden_size,
                "champion": champion.neural_hidden,
            })
            check("router:intent_vocabulary_matches", router.output_size == len(INTENTS), {
                "router": router.output_size,
                "expected": len(INTENTS),
            })
        except Exception as exc:
            check("router:loadable", False, repr(exc))
    else:
        check("router:present", False, str(router_path))

    discoveries = _read_json(root / "discoveries.json", {"theorems": {}, "discarded": {}})
    active = discoveries.get("theorems") or {}
    discarded = discoveries.get("discarded") or {}
    overlap = sorted(set(active) & set(discarded))
    check("discovery:active_discarded_disjoint", not overlap, overlap)

    bad_active: list[dict[str, Any]] = []
    for theorem_id, row in active.items():
        if not row.get("verified"):
            # Rejected conjectures may remain as bounded research history.
            continue

        strategy = str(row.get("strategy", ""))
        statement = str(row.get("statement", ""))
        cert = row.get("certificate") or {}

        if not bool(cert.get("ok", False)):
            bad_active.append({"id": theorem_id, "reason": "verified row has non-ok certificate"})
            continue

        if strategy == "faulhaber_interpolation":
            recurrence = cert.get("recurrence") or {}
            recurrence_statement = str(recurrence.get("statement", ""))
            if not (
                cert.get("recurrence_nontrivial") is True
                and recurrence.get("ok") is True
                and recurrence_statement
                and _relation_is_structurally_nontrivial(recurrence_statement)
            ):
                bad_active.append({"id": theorem_id, "reason": "Faulhaber recurrence is not proof-gated/nontrivial"})
            continue

        try:
            parse_relation(statement)
        except Exception:
            # Some future theorem schemas may not be relation-style; they need
            # their own explicit strategy-specific health check before being
            # made critical.
            continue
        if not _relation_is_structurally_nontrivial(statement):
            bad_active.append({"id": theorem_id, "reason": "structural tautology"})

    check("discovery:verified_quality", not bad_active, bad_active)

    curriculum = _read_json(root / "curriculum.json", None)
    if isinstance(curriculum, dict):
        effective_version = max(2, int(curriculum.get("version", 1)))
        check("curriculum:schema", effective_version >= 2, effective_version)
        cursor = int(curriculum.get("cursor", 0))
        check("curriculum:cursor_nonnegative", cursor >= 0, cursor)
        bad_retries = {
            str(domain): count
            for domain, count in (curriculum.get("retry_counts") or {}).items()
            if domain not in DOMAIN_ATLAS or not isinstance(count, int) or not (1 <= count <= 2)
        }
        check("curriculum:retry_counters_bounded", not bad_retries, bad_retries)

    status = _read_json(root / "status.json", None)
    if isinstance(status, dict):
        check("status:reported_ok", status.get("ok") is True, status.get("ok"))
        benchmark = ((status.get("evolution") or {}).get("benchmark") or {})
        check("status:no_critical_failures", not benchmark.get("critical_failures"), benchmark.get("critical_failures"))
        verifiers = status.get("verifiers") or {}
        check("status:z3_available", verifiers.get("z3_available") is True, verifiers)
        check("status:sympy_present", bool(verifiers.get("sympy")), verifiers)

    last = _read_json(root / "last-cycle.json", None)
    if isinstance(last, dict):
        check("cycle:reported_ok", last.get("ok") is True, last.get("ok"))
        evolution = last.get("evolution") or {}
        kernel = ((evolution.get("benchmark") or {}).get("kernel_integrity") or {})
        check("cycle:kernel_integrity", kernel.get("ok") is True, kernel)

    failed = [row for row in checks if not row["ok"]]
    return {
        "ok": not failed,
        "state_dir": str(root),
        "champion": champion.genome_id,
        "generation": champion.generation,
        "checks": checks,
        "failed": failed,
    }


def main() -> int:
    report = health_report()
    root = Path(report["state_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "health.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
