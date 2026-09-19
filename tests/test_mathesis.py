from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "computer"))

from mathesis.architecture import _ALLOWED_EXPERTS, default_genome
from mathesis.benchmark import TRAINING_PHRASES, VALIDATION_PHRASES
from mathesis.engine import MathesisOmega
from mathesis.discovery import ConjectureDiscoveryEngine
from mathesis.curriculum import MathematicalCurriculum
from mathesis.sympy_lab import SymPyMathLab, DOMAIN_ATLAS
from mathesis.evolution import SelfEvolutionEngine
from mathesis.experience import ExperienceAnalyzer
from mathesis.formalizer import FormalizerMesh
from mathesis.health import health_report
from mathesis.kernel import IntegrityKernel
from mathesis.model_writer import render_model_module
from mathesis.neural_graph import GrowingNeuralRouter
from mathesis.research import ReadOnlyResearcher
from mathesis.safe_math import parse_expr, parse_relation
from mathesis.synthesis import ProgramSynthesizer, _guard_code
from mathesis.verifiers import CompositeVerifier, LeanVerifier


def test_safe_math_parser_accepts_math_and_rejects_code():
    assert str(parse_expr("2+3*4")) == "14"
    assert str(parse_expr("2x+3")) in {"2*x + 3", "3 + 2*x"}
    with pytest.raises(Exception):
        parse_expr("__import__('os').system('id')")
    with pytest.raises(Exception):
        parse_expr("open('/etc/passwd').read()")


def test_formalizer_understands_core_italian_requests():
    mesh = FormalizerMesh()
    assert mesh.formalize("calcola 17+25").kind == "arithmetic"
    assert mesh.formalize("risolvi 2*x+3=7").kind == "equation"
    identity = mesh.formalize("dimostra che (x+1)^2=x^2+2*x+1")
    assert identity.kind == "identity"
    assert not identity.expression.lower().startswith("che ")
    assert mesh.formalize("scrivi una funzione mcd").target == "gcd"
    assert mesh.formalize("scrivimi il tuo prossimo modello").kind == "evolve_model"
    assert mesh.formalize("deriva x^3").kind == "derivative"
    assert mesh.formalize("integra 2*x").kind == "integral"
    assert mesh.formalize("analizza matematicamente x^2-1").kind == "analyze_math"
    assert mesh.formalize("scopri nuova matematica").kind == "discover_math"


def test_true_identity_requires_independent_proof_paths():
    verifier = CompositeVerifier()
    cert = verifier.verify_relation("(x+1)^2 = x^2+2*x+1")
    assert cert.ok is True
    assert cert.status == "verified"
    assert len(cert.methods) >= 2
    assert any("sympy" in method for method in cert.methods)
    assert any("z3" in method for method in cert.methods)


def test_false_but_plausible_identity_is_rejected_with_counterexample():
    verifier = CompositeVerifier()
    cert = verifier.verify_relation("(x+1)^2 = x^2+1")
    assert cert.ok is False
    assert cert.status == "disproved"
    assert cert.counterexample is not None


def test_exact_arithmetic_is_certified():
    result, cert = CompositeVerifier().evaluate("2+3*4")
    assert result == 14
    assert cert.ok is True


def test_linear_equation_solution_set_is_checked_for_completeness():
    solutions, cert = CompositeVerifier().solve_equation("2*x+3=7")
    assert solutions == [2]
    assert cert.ok is True
    assert "z3_solution_set_completeness" in cert.methods


@pytest.mark.parametrize("task", ["gcd", "fibonacci", "factorial", "is_prime", "sort"])
def test_program_synthesis_returns_only_verified_champions(task):
    candidate = ProgramSynthesizer().synthesize(task)
    assert candidate.verified is True
    assert candidate.tests_passed == candidate.tests_total
    assert "import " not in candidate.code


def test_generated_program_guard_blocks_imports_and_attributes():
    with pytest.raises(ValueError):
        _guard_code("import os\ndef solve(): return 1\n")
    with pytest.raises(ValueError):
        _guard_code("def solve(x): return x.__class__\n")


def test_growing_neural_router_can_expand_and_learn():
    router = GrowingNeuralRouter(hidden_size=12)
    router.train(TRAINING_PHRASES, epochs=80, lr=0.07)
    before = router.accuracy(VALIDATION_PHRASES)
    grown = router.grow(6)
    grown.train(TRAINING_PHRASES, epochs=60, lr=0.05)
    after = grown.accuracy(VALIDATION_PHRASES)
    assert grown.hidden_size == 18
    assert before >= 0.40
    assert after >= 0.40


def test_model_writer_emits_data_only_python():
    source = render_model_module(default_genome())
    tree = ast.parse(source)
    assert "MODEL =" in source
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)) for node in ast.walk(tree))


def test_integrity_kernel_detects_no_change_during_readonly_cycle():
    kernel = IntegrityKernel()
    before = kernel.snapshot()
    result = kernel.verify_snapshot(before)
    assert result["ok"] is True
    assert result["changed"] == []


def test_self_evolution_writes_candidate_and_promotes_verified_improvement(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once()
    assert result.promoted is True
    assert result.candidate_score > result.champion_score
    assert (tmp_path / "candidate_model.py").exists()
    assert (tmp_path / "champion_model.py").exists()
    assert (tmp_path / "champion.json").exists()
    champion = json.loads((tmp_path / "champion.json").read_text(encoding="utf-8"))
    baseline = set(default_genome().experts)
    evolved = set(champion["experts"])
    assert champion["generation"] == 1
    assert baseline.issubset(evolved)
    assert len(evolved - baseline) >= 1
    assert result.benchmark["kernel_integrity"]["ok"] is True


def test_engine_end_to_end_understands_proves_and_programs(tmp_path: Path):
    engine = MathesisOmega(tmp_path)

    arithmetic = engine.answer("calcola 12*7")
    assert arithmetic["ok"] is True
    assert arithmetic["result"] == 84

    equation = engine.answer("risolvi 2*x+3=7")
    assert equation["ok"] is True
    assert equation["solutions"] == [2]

    identity = engine.answer("dimostra che (x+1)^2=x^2+2*x+1")
    assert identity["ok"] is True
    assert identity["truth_status"] == "verified"

    program = engine.answer("scrivi una funzione mcd")
    assert program["ok"] is True
    assert program["program"]["verified"] is True


def test_engine_exact_next_model_request_runs_transactional_evolution(tmp_path: Path):
    engine = MathesisOmega(tmp_path)
    result = engine.answer("scrivimi il tuo prossimo modello")
    assert result["ok"] is True
    assert result["intent"]["kind"] == "evolve_model"
    assert result["evolution"]["promoted"] is True
    assert result["current_model"]["genome"]["generation"] == 1


def test_unknown_language_abstains_instead_of_guessing(tmp_path: Path):
    result = MathesisOmega(tmp_path).answer("raccontami una favola sui draghi")
    assert result["ok"] is False
    assert result["status"] == "abstained"


def test_research_is_evidence_only_without_formal_proof(monkeypatch):
    import mathesis.research as research_module

    monkeypatch.setattr(
        research_module,
        "search_web",
        lambda claim, limit=5: [
            {"url": "https://one.example/a", "domain": "one.example", "title": "One"},
            {"url": "https://two.example/b", "domain": "two.example", "title": "Two"},
        ],
    )
    monkeypatch.setattr(
        research_module,
        "fetch_text",
        lambda url, timeout=15, max_bytes=1_000_000: {
            "url": url,
            "content_type": "text/html",
            "text": "<html><body>Evidence text</body></html>",
        },
    )
    result = ReadOnlyResearcher().search("an unformalized mathematical history claim", max_sources=2)
    assert result["ok"] is True
    assert result["status"] == "evidence_only"
    assert result["contract"]["https_only"] is True
    assert result["contract"]["remote_writes"] is False
    assert len(result["sources"]) == 2


def test_engine_web_evidence_does_not_override_mathematical_disproof(monkeypatch, tmp_path: Path):
    import mathesis.research as research_module

    monkeypatch.setattr(
        research_module,
        "search_web",
        lambda claim, limit=5: [
            {"url": "https://one.example/a", "domain": "one.example", "title": "Convincing page"},
            {"url": "https://two.example/b", "domain": "two.example", "title": "Another page"},
        ],
    )
    monkeypatch.setattr(
        research_module,
        "fetch_text",
        lambda url, timeout=15, max_bytes=1_000_000: {
            "url": url,
            "content_type": "text/html",
            "text": "<html><body>2+2=5</body></html>",
        },
    )

    engine = MathesisOmega(tmp_path)
    result = engine.answer("ricerca 2+2=5")
    assert result["truth_status"] == "disproved"
    assert result["ok"] is False


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean is optional outside CI")
def test_real_lean_kernel_adapter_when_installed():
    rel = parse_relation("2+3=5")
    result = LeanVerifier().verify_closed_equality(rel)
    assert result["ok"] is True
    assert result["status"] == "verified"


def test_safe_math_whitelists_real_math_functions_and_assumptions():
    assert str(parse_expr("sin(pi/2)")) == "1"
    positive = parse_expr("sqrt(x^2)", assumptions={"x": {"positive": True}})
    assert str(positive) == "x"
    with pytest.raises(Exception):
        parse_expr("eval(2+2)")


def test_sympy_math_lab_covers_multiple_domains():
    lab = SymPyMathLab()
    assert len(DOMAIN_ATLAS) >= 12
    assert lab.algebra_normal_forms("(x+1)^3").ok is True
    assert lab.derivative("x^4+2*x").ok is True
    assert lab.antiderivative("3*x^2").ok is True
    assert lab.trig_normal_form("sin(x)^2+cos(x)^2").ok is True
    assert lab.matrix_invariants([[2, 1], [1, 1]]).ok is True
    assert lab.number_theory_profile(360).ok is True


def test_discovery_engine_creates_only_proof_gated_internal_novelty(tmp_path: Path):
    engine = ConjectureDiscoveryEngine(tmp_path)
    results = [engine.discover_once() for _ in range(4)]
    assert all(row["ok"] is True for row in results)
    statements = [row["theorem"]["statement"] for row in results]
    assert len(statements) == len(set(statements))
    for row in results:
        theorem = row["theorem"]
        assert theorem["verified"] is True
        assert theorem["internal_novelty"] is True
        assert theorem["human_novelty"] == "unassessed"
    status = engine.status()
    assert status["verified"] >= 4
    assert status["rejected"] == 0


def test_faulhaber_discovery_has_induction_style_certificate(tmp_path: Path):
    engine = ConjectureDiscoveryEngine(tmp_path)
    # cycle 0 is binomial, cycle 1 is Faulhaber.
    engine.discover_once()
    result = engine.discover_once()
    theorem = result["theorem"]
    assert theorem["strategy"] == "faulhaber_interpolation"
    cert = theorem["certificate"]
    assert cert["ok"] is True
    assert cert["base_case"] is True
    assert cert["recurrence_nontrivial"] is True
    assert cert["recurrence"]["ok"] is True
    rel = parse_relation(cert["recurrence"]["statement"])
    import sympy as sp
    assert sp.srepr(rel.lhs) != sp.srepr(rel.rhs)


def test_engine_calculus_analysis_and_discovery_end_to_end(tmp_path: Path):
    engine = MathesisOmega(tmp_path)

    derivative = engine.answer("deriva x^3+2*x")
    assert derivative["ok"] is True
    assert "3*x**2" in derivative["math"]["result"]["derivative"]

    integral = engine.answer("integra 2*x")
    assert integral["ok"] is True

    analysis = engine.answer("analizza matematicamente x^2-1")
    assert analysis["ok"] is True

    discovery = engine.answer("scopri nuova matematica")
    assert discovery["ok"] is True
    assert discovery["discovery"]["status"] == "verified_discovery"


def test_evolution_uses_multiple_challengers_and_keeps_kernel_immutable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MATHESIS_CHALLENGERS", "3")
    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once()
    assert len(result.benchmark["trials"]) == 3
    assert result.benchmark["kernel_integrity"]["ok"] is True
    assert result.promoted is True
    assert result.candidate_score > result.champion_score


def test_persisted_old_router_shape_is_migrated_instead_of_reinterpreted(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    genome = evolution.load_champion()
    old_router = GrowingNeuralRouter(hidden_size=genome.neural_hidden, output_size=7)
    (tmp_path / "router.json").write_text(json.dumps(old_router.to_dict()), encoding="utf-8")
    migrated = evolution.load_router(genome)
    from mathesis.neural_graph import INTENTS
    assert migrated.output_size == len(INTENTS)


def test_experience_feedback_turns_verified_math_into_architecture_signal(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    champion = evolution.load_champion()
    feedback = ExperienceAnalyzer(tmp_path).signals(
        champion,
        latest_discovery={
            "ok": True,
            "status": "verified_discovery",
            "theorem": {
                "verified": True,
                "strategy": "binomial_expansion",
                "complexity": champion.symbolic_depth,
            },
        },
    )
    assert "missing_combinatorics_expert" in feedback["weaknesses"]
    assert "symbolic_depth" in feedback["weaknesses"]


def test_rejected_discovery_requests_stronger_search_not_direct_rewrite(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    champion = evolution.load_champion()
    feedback = ExperienceAnalyzer(tmp_path).signals(
        champion,
        latest_discovery={
            "ok": False,
            "status": "rejected_conjecture",
            "theorem": {
                "verified": False,
                "strategy": "faulhaber_interpolation",
                "complexity": 4,
            },
        },
    )
    assert "counterexample" in feedback["weaknesses"]
    assert "optimization" in feedback["weaknesses"]
    assert "verifier" not in " ".join(feedback["weaknesses"]).lower()


def test_experience_hint_changes_challenger_expert_selection(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once(extra_weaknesses=["missing_combinatorics_expert"])
    trial_experts = [trial["genome"]["experts"] for trial in result.benchmark["trials"]]
    assert any("combinatorics" in experts for experts in trial_experts)
    assert "missing_combinatorics_expert" in result.benchmark["weakness_hints"]


def test_verified_discoveries_become_replayable_regression_theorems(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path)
    first = discovery.discover_once()
    assert first["ok"] is True

    analyzer = ExperienceAnalyzer(tmp_path)
    learned = analyzer.replayable_theorems(limit=16)
    assert first["theorem"]["statement"] in learned


def test_evolution_replays_learned_math_before_promotion(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path)
    theorem = discovery.discover_once()["theorem"]["statement"]

    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once()
    assert theorem in result.benchmark["learned_theorems"]
    candidate_replay = result.benchmark["candidate"]["learned_theorems_replayed"]
    assert any(row["statement"] == theorem and row["ok"] for row in candidate_replay)


def test_false_learned_relation_is_a_critical_regression_gate():
    from mathesis.benchmark import evaluate_genome

    benchmark = evaluate_genome(
        default_genome(),
        learned_theorems=["(x+1)^2=x^2+1"],
    )
    assert benchmark["ok"] is False
    assert "learned_theorem:0" in benchmark["critical_failures"]


def test_continuum_has_cron_watchdog_and_self_handoff_contract():
    workflow = (ROOT / ".github" / "workflows" / "mathesis-continuum.yml").read_text(encoding="utf-8")
    assert "actions: write" in workflow
    assert "MATHESIS_MIN_CHAIN_SECONDS: '300'" in workflow
    assert "Hand off to the next autonomous cycle" in workflow
    assert "actions/workflows/mathesis-continuum.yml/dispatches" in workflow
    assert "Another MATHESIS continuum run is already queued/running" in workflow
    assert '.status == "pending"' in workflow
    assert '.status == "queued"' in workflow
    assert '.status == "in_progress"' in workflow
    assert '.status == "waiting"' in workflow
    assert '.status == "requested"' in workflow
    assert "?per_page=50" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "3,8,13,18,23,28,33,38,43,48,53,58" in workflow


def test_verified_discovery_domain_pressure_persists_until_expert_exists(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path)
    discovery.discover_once()  # binomial -> combinatorics
    discovery.discover_once()  # Faulhaber -> sequences

    evolution = SelfEvolutionEngine(tmp_path)
    champion = evolution.load_champion()
    feedback = ExperienceAnalyzer(tmp_path).signals(champion)

    assert "missing_combinatorics_expert" in feedback["weaknesses"]
    assert "missing_sequences_expert" in feedback["weaknesses"]

    champion.experts.extend(["combinatorics", "sequences"])
    resolved = ExperienceAnalyzer(tmp_path).signals(champion)
    assert "missing_combinatorics_expert" not in resolved["weaknesses"]
    assert "missing_sequences_expert" not in resolved["weaknesses"]


def test_failed_curriculum_retries_on_next_cycle():
    from mathesis.evolution_cycle import _should_study

    assert _should_study(5, 3, "no_sources") is True
    assert _should_study(5, 3, "research_error") is True
    assert _should_study(6, 3, "studied") is True
    assert _should_study(5, 3, "studied") is False


def test_watchdog_treats_all_pending_continuum_states_as_active():
    workflow = (ROOT / ".github" / "workflows" / "mathesis-watchdog.yml").read_text(encoding="utf-8")
    assert '.status == "pending"' in workflow
    assert '.status == "queued"' in workflow
    assert '.status == "in_progress"' in workflow
    assert '.status == "waiting"' in workflow
    assert '.status == "requested"' in workflow
    assert "?per_page=50" in workflow


def test_discovery_relations_are_structurally_nontrivial(tmp_path: Path):
    engine = ConjectureDiscoveryEngine(tmp_path, symbolic_depth=6, discovery_beam=8)
    results = [engine.discover_once() for _ in range(8)]
    verified = [row["theorem"] for row in results if row.get("theorem") and row["theorem"].get("verified")]
    assert verified
    for theorem in verified:
        statement = theorem["statement"]
        if "sum(" in statement and "for integer" in statement:
            continue
        rel = parse_relation(statement)
        import sympy as sp
        assert sp.srepr(rel.lhs) != sp.srepr(rel.rhs)
        assert theorem["nontrivial"] is True
        assert theorem["quality_gate"] == "structurally_nontrivial_and_proof_gated"


def test_difference_and_geometric_discoveries_preserve_nontrivial_forms(tmp_path: Path):
    engine = ConjectureDiscoveryEngine(tmp_path, symbolic_depth=6, discovery_beam=8)
    rows = [engine.discover_once()["theorem"] for _ in range(6)]
    difference = next(row for row in rows if row["strategy"] == "difference_of_powers")
    geometric = next(row for row in rows if row["strategy"] == "finite_geometric_identity")
    assert ")*(" in difference["statement"]
    assert "(x-1)*(" in geometric["statement"]
    assert difference["verified"] is True
    assert geometric["verified"] is True


def test_legacy_structural_tautologies_are_migrated_out_of_active_theorems(tmp_path: Path):
    legacy_statement = "x^4-y^4 = x**4-y**4"
    legacy_id = __import__("hashlib").sha256(legacy_statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 1,
        "cycle": 4,
        "strategy_counts": {"difference_of_powers": 1},
        "theorems": {
            legacy_id: {
                "id": legacy_id,
                "statement": legacy_statement,
                "verified": True,
                "strategy": "difference_of_powers",
                "discovered_at": 1.0,
                "certificate": {"ok": True},
            }
        },
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")

    engine = ConjectureDiscoveryEngine(tmp_path, discovery_beam=4)
    result = engine.discover_once()
    assert result["ok"] is True

    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    assert legacy_id not in migrated["theorems"]
    assert migrated["discarded"][legacy_id]["discarded_reason"] == "structural_tautology"
    assert migrated["last_quality_migration"]["reason"] == "proof_quality_upgrade"
    assert legacy_id in migrated["last_quality_migration"]["moved"]


def test_every_math_lab_domain_is_evolvable_by_architecture():
    assert set(DOMAIN_ATLAS).issubset(set(_ALLOWED_EXPERTS))


def test_polynomial_feedback_can_be_resolved_by_a_challenger(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once(extra_weaknesses=["missing_polynomials_expert"])
    trial_experts = [set(trial["genome"]["experts"]) for trial in result.benchmark["trials"]]
    assert any("polynomials" in experts for experts in trial_experts)


def test_polynomial_expert_is_actually_benchmarked():
    from mathesis.benchmark import evaluate_genome
    genome = default_genome()
    genome.experts.append("polynomials")
    benchmark = evaluate_genome(genome)
    task = next(row for row in benchmark["tasks"] if row["name"] == "domain:polynomials")
    assert task["ok"] is True


def test_symbolic_depth_score_is_independent_from_learned_theorem_replay():
    from mathesis.benchmark import evaluate_genome

    genome = default_genome()
    benchmark = evaluate_genome(
        genome,
        learned_theorems=["(x+1)^2=x^2+1"],
    )
    assert benchmark["ok"] is False
    assert benchmark["verified_symbolic_depth"] == genome.symbolic_depth
    search_task = next(row for row in benchmark["tasks"] if row["name"] == "search:symbolic_depth")
    assert search_task["ok"] is True


def test_legacy_faulhaber_certificate_is_reproved_with_nontrivial_induction(tmp_path: Path):
    statement = "sum(k^2, k=1..n) = n*(n + 1)*(2*n + 1)/6 for integer n>=0"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 2,
        "cycle": 2,
        "strategy_counts": {"faulhaber_interpolation": 1},
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "faulhaber_interpolation",
                "complexity": 2,
                "discovered_at": 1.0,
                "certificate": {
                    "ok": True,
                    "status": "verified",
                    "base_case": True,
                    "sample_count": 6,
                    "polynomial": "n*(n + 1)*(2*n + 1)/6",
                    "recurrence": {
                        "ok": True,
                        "status": "verified",
                        "statement": "n**2+2*n+1=n**2+2*n+1",
                    },
                },
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")

    engine = ConjectureDiscoveryEngine(tmp_path, symbolic_depth=6, discovery_beam=4)
    result = engine.discover_once()
    assert result["ok"] is True

    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    theorem = migrated["theorems"][theorem_id]
    cert = theorem["certificate"]
    assert cert["recurrence_nontrivial"] is True
    assert cert["recurrence"]["ok"] is True
    rel = parse_relation(cert["recurrence"]["statement"])
    import sympy as sp
    assert sp.srepr(rel.lhs) != sp.srepr(rel.rhs)
    assert theorem_id in migrated["last_quality_migration"]["revalidated"]


def test_unreprovable_legacy_faulhaber_is_removed_from_active_corpus(tmp_path: Path):
    statement = "sum(k^2, k=1..n) = n*(n+1)/2 for integer n>=0"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 2,
        "cycle": 2,
        "strategy_counts": {"faulhaber_interpolation": 1},
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "faulhaber_interpolation",
                "complexity": 2,
                "discovered_at": 1.0,
                "certificate": {
                    "ok": True,
                    "status": "verified",
                    "polynomial": "n*(n+1)/2",
                    "sample_count": 6,
                },
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")

    engine = ConjectureDiscoveryEngine(tmp_path, symbolic_depth=6, discovery_beam=4)
    engine.discover_once()

    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    assert theorem_id not in migrated["theorems"]
    assert migrated["discarded"][theorem_id]["discarded_reason"] == "legacy_faulhaber_reproof_failed"


def test_curriculum_retries_same_failed_domain_before_advancing(tmp_path: Path):
    curriculum = MathematicalCurriculum(tmp_path)
    curriculum.researcher.search = lambda query, max_sources=3: {
        "ok": False,
        "sources": [],
        "errors": [],
    }

    first = curriculum.study_once()
    second = curriculum.study_once()
    state = json.loads((tmp_path / "curriculum.json").read_text(encoding="utf-8"))

    assert first["domain"] == "algebra"
    assert second["domain"] == "algebra"
    assert state["cursor"] == 0
    assert state["retry_counts"]["algebra"] == 2

    third = curriculum.study_once()
    state = json.loads((tmp_path / "curriculum.json").read_text(encoding="utf-8"))
    assert third["domain"] == "algebra"
    assert state["cursor"] == 1
    assert "algebra" not in state["retry_counts"]


def test_curriculum_success_clears_retry_and_advances(tmp_path: Path):
    curriculum = MathematicalCurriculum(tmp_path)
    attempts = {"count": 0}

    def fake_search(query, max_sources=3):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"ok": False, "sources": [], "errors": []}
        return {
            "ok": True,
            "sources": [{
                "url": "https://example.org/math",
                "domain": "example.org",
                "authority_hint": 0.75,
                "excerpt": "mathematics",
            }],
            "errors": [],
        }

    curriculum.researcher.search = fake_search
    first = curriculum.study_once()
    second = curriculum.study_once()
    state = json.loads((tmp_path / "curriculum.json").read_text(encoding="utf-8"))

    assert first["domain"] == second["domain"] == "algebra"
    assert second["status"] == "studied"
    assert state["cursor"] == 1
    assert "algebra" not in state["retry_counts"]


def test_autonomous_workflow_api_calls_are_time_bounded_and_retried():
    workflows = [
        ROOT / ".github" / "workflows" / "mathesis-continuum.yml",
        ROOT / ".github" / "workflows" / "mathesis-watchdog.yml",
        ROOT / ".github" / "workflows" / "mathesis-deep-math.yml",
    ]
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        assert "--connect-timeout 10" in text
        assert "--max-time 30" in text
        assert "--retry 3" in text
        assert "--retry-all-errors" in text

    continuum = workflows[0].read_text(encoding="utf-8")
    assert "timeout-minutes: 15" in continuum


def test_handoff_and_watchdog_have_bounded_network_fallbacks():
    continuum = (ROOT / ".github" / "workflows" / "mathesis-continuum.yml").read_text(encoding="utf-8")
    watchdog = (ROOT / ".github" / "workflows" / "mathesis-watchdog.yml").read_text(encoding="utf-8")

    assert "Could not inspect active runs after bounded retries; attempting one continuity dispatch." in continuum
    assert "active_other=0" in continuum

    assert "reason=continuum_run_lookup_failed" in watchdog
    assert 'echo "stale=true"' in watchdog
    assert '} >> "$GITHUB_OUTPUT"' in watchdog


def test_faulhaber_induction_obligation_enters_antiforgetting_replay(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path, symbolic_depth=6, discovery_beam=8)
    discovery.discover_once()  # binomial
    faulhaber = discovery.discover_once()["theorem"]
    assert faulhaber["strategy"] == "faulhaber_interpolation"
    assert faulhaber["certificate"]["recurrence_nontrivial"] is True

    learned = ExperienceAnalyzer(tmp_path).replayable_theorems(limit=16)
    recurrence_statement = faulhaber["certificate"]["recurrence"]["statement"]
    assert recurrence_statement in learned
    assert faulhaber["statement"] not in learned


def test_faulhaber_replay_is_critical_for_future_promotion(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path, symbolic_depth=6, discovery_beam=8)
    discovery.discover_once()
    faulhaber = discovery.discover_once()["theorem"]
    obligation = faulhaber["certificate"]["recurrence"]["statement"]

    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once()
    assert obligation in result.benchmark["learned_theorems"]
    replay = result.benchmark["candidate"]["learned_theorems_replayed"]
    assert any(row["statement"] == obligation and row["ok"] for row in replay)


def test_curriculum_v1_state_is_read_as_v2_schema(tmp_path: Path):
    legacy = {
        "version": 1,
        "cursor": 4,
        "studies": [{"domain": "algebra", "status": "studied"}],
    }
    (tmp_path / "curriculum.json").write_text(json.dumps(legacy), encoding="utf-8")
    status = MathematicalCurriculum(tmp_path).status()
    assert status["version"] == 2
    assert status["cursor"] == 4
    assert status["retry_counts"] == {}


def test_health_gate_accepts_consistent_evolved_state(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path)
    assert discovery.discover_once()["ok"] is True
    evolution = SelfEvolutionEngine(tmp_path)
    result = evolution.evolve_once()
    assert result.benchmark["kernel_integrity"]["ok"] is True
    report = health_report(tmp_path)
    assert report["ok"] is True
    assert report["failed"] == []


def test_health_gate_rejects_topology_escape(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    evolution.evolve_once()
    champion_path = tmp_path / "champion.json"
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    champion["topology"].append(["algebra", "imaginary_unregistered_expert"])
    champion_path.write_text(json.dumps(champion), encoding="utf-8")

    report = health_report(tmp_path)
    assert report["ok"] is False
    assert any(row["name"] == "architecture:topology_closed" for row in report["failed"])


def test_health_gate_rejects_bad_active_faulhaber_certificate(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    evolution.evolve_once()

    statement = "sum(k^2, k=1..n) = n*(n+1)*(2*n+1)/6 for integer n>=0"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    discoveries = {
        "version": 3,
        "cycle": 1,
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "faulhaber_interpolation",
                "certificate": {
                    "ok": True,
                    "recurrence_nontrivial": False,
                    "recurrence": {
                        "ok": True,
                        "statement": "n**2=n**2",
                    },
                },
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(discoveries), encoding="utf-8")
    report = health_report(tmp_path)
    assert report["ok"] is False
    quality = next(row for row in report["failed"] if row["name"] == "discovery:verified_quality")
    assert theorem_id in str(quality["detail"])


def test_continuum_health_gate_runs_before_state_persistence():
    workflow = (ROOT / ".github" / "workflows" / "mathesis-continuum.yml").read_text(encoding="utf-8")
    health_pos = workflow.index("python -m mathesis.health")
    persist_pos = workflow.index("Persist and verify state heartbeat")
    assert health_pos < persist_pos
    assert 'test -f "$MATHESIS_STATE_DIR/health.json"' in workflow


def test_discovery_schema_upgrade_persists_even_without_theorem_changes(tmp_path: Path):
    state = {
        "version": 2,
        "cycle": 0,
        "strategy_counts": {},
        "theorems": {},
        "discarded": {},
    }
    path = tmp_path / "discoveries.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    engine = ConjectureDiscoveryEngine(tmp_path)
    engine._audit_legacy_discoveries(engine._load())
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["version"] == 3
    assert migrated["last_quality_migration"]["schema_upgraded"] is True


def test_legacy_faulhaber_migration_is_idempotent(tmp_path: Path):
    statement = "sum(k^2, k=1..n) = n*(n + 1)*(2*n + 1)/6 for integer n>=0"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 2,
        "cycle": 2,
        "strategy_counts": {"faulhaber_interpolation": 1},
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "faulhaber_interpolation",
                "complexity": 2,
                "discovered_at": 1.0,
                "certificate": {
                    "ok": True,
                    "sample_count": 6,
                    "polynomial": "n*(n + 1)*(2*n + 1)/6",
                },
            }
        },
        "discarded": {},
    }
    path = tmp_path / "discoveries.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    engine = ConjectureDiscoveryEngine(tmp_path)

    engine._audit_legacy_discoveries(engine._load())
    once = path.read_text(encoding="utf-8")
    first = json.loads(once)
    assert first["theorems"][theorem_id]["certificate"]["recurrence_nontrivial"] is True
    assert first["theorems"][theorem_id]["certificate"]["recurrence"]["ok"] is True

    engine._audit_legacy_discoveries(engine._load())
    twice = path.read_text(encoding="utf-8")
    second = json.loads(twice)
    assert twice == once
    assert list(second["theorems"]).count(theorem_id) == 1
    assert theorem_id not in second["discarded"]


def test_legacy_faulhaber_statement_must_match_reproved_polynomial(tmp_path: Path):
    # The polynomial/certificate is the true sum-of-squares formula, but the
    # displayed theorem is deliberately false. Metadata must not rescue it.
    statement = "sum(k^2, k=1..n) = n*(n + 1)/2 for integer n>=0"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 2,
        "cycle": 1,
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "faulhaber_interpolation",
                "complexity": 2,
                "certificate": {
                    "ok": True,
                    "sample_count": 6,
                    "polynomial": "n*(n + 1)*(2*n + 1)/6",
                },
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")
    engine = ConjectureDiscoveryEngine(tmp_path)
    engine._audit_legacy_discoveries(engine._load())
    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    assert theorem_id not in migrated["theorems"]
    assert migrated["discarded"][theorem_id]["discarded_reason"] == "legacy_faulhaber_reproof_failed"


def test_legacy_faulhaber_reproof_bounds_untrusted_sample_count(tmp_path: Path):
    statement = "sum(k^2, k=1..n) = n*(n + 1)*(2*n + 1)/6 for integer n>=0"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 2,
        "cycle": 1,
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "faulhaber_interpolation",
                "complexity": 2,
                "certificate": {
                    "ok": True,
                    "sample_count": 10**12,
                    "polynomial": "n*(n + 1)*(2*n + 1)/6",
                },
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")
    engine = ConjectureDiscoveryEngine(tmp_path)
    engine._audit_legacy_discoveries(engine._load())
    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    assert migrated["theorems"][theorem_id]["certificate"]["sample_count"] <= 64


def test_valid_legacy_non_faulhaber_is_reproved_without_damage(tmp_path: Path):
    statement = "x^4-y^4 = (x - y)*(x**3 + x**2*y + x*y**2 + y**3)"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 2,
        "cycle": 1,
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "difference_of_powers",
                "complexity": 4,
                "certificate": {"ok": True},
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")
    engine = ConjectureDiscoveryEngine(tmp_path)
    engine._audit_legacy_discoveries(engine._load())
    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    theorem = migrated["theorems"][theorem_id]
    assert theorem["verified"] is True
    assert theorem["certificate"]["ok"] is True
    assert theorem["quality_gate"] == "structurally_nontrivial_and_proof_gated"
    assert theorem_id not in migrated["discarded"]


def test_false_verified_metadata_is_archived_by_quality_audit(tmp_path: Path):
    statement = "x+1=x+2"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    state = {
        "version": 3,
        "cycle": 1,
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "difference_of_powers",
                "complexity": 2,
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": True, "status": "verified"},
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(state), encoding="utf-8")
    engine = ConjectureDiscoveryEngine(tmp_path)
    engine._audit_legacy_discoveries(engine._load())
    migrated = json.loads((tmp_path / "discoveries.json").read_text(encoding="utf-8"))
    assert theorem_id not in migrated["theorems"]
    assert migrated["discarded"][theorem_id]["discarded_reason"] == "verified_discovery_revalidation_failed"


def test_antiforgetting_replays_only_currently_valid_active_theorems(tmp_path: Path):
    discovery = ConjectureDiscoveryEngine(tmp_path)
    valid = discovery.discover_once()["theorem"]
    path = tmp_path / "discoveries.json"
    state = json.loads(path.read_text(encoding="utf-8"))

    false_statement = "x+1=x+2"
    false_id = __import__("hashlib").sha256(false_statement.encode("utf-8")).hexdigest()[:20]
    state["theorems"][false_id] = {
        "id": false_id,
        "statement": false_statement,
        "verified": True,
        "strategy": "difference_of_powers",
        "quality_gate": "structurally_nontrivial_and_proof_gated",
        "certificate": {"ok": True, "status": "verified"},
        "discovered_at": valid["discovered_at"] + 1,
    }
    discarded_statement = "(x+2)^2=x^2+4*x+4"
    discarded_id = __import__("hashlib").sha256(discarded_statement.encode("utf-8")).hexdigest()[:20]
    state["discarded"][discarded_id] = {
        "id": discarded_id,
        "statement": discarded_statement,
        "verified": True,
        "quality_gate": "structurally_nontrivial_and_proof_gated",
        "certificate": {"ok": True},
    }
    path.write_text(json.dumps(state), encoding="utf-8")

    learned = ExperienceAnalyzer(tmp_path).replayable_theorems(limit=16)
    assert valid["statement"] in learned
    assert false_statement not in learned
    assert discarded_statement not in learned


def test_all_sympy_domains_complete_curriculum_to_benchmark_chain(tmp_path: Path):
    from mathesis.architecture import generate_challengers
    from mathesis.benchmark import evaluate_genome

    for domain in DOMAIN_ATLAS:
        champion = default_genome()
        champion.experts = [expert for expert in champion.experts if expert != domain]
        feedback = ExperienceAnalyzer(tmp_path / domain).signals(
            champion,
            latest_study={"ok": True, "status": "studied", "domain": domain},
        )
        weakness = f"missing_{domain}_expert"
        assert weakness in feedback["weaknesses"], domain

        challenger = generate_challengers(
            champion,
            weaknesses=feedback["weaknesses"],
            count=1,
        )[0]
        assert domain in challenger.experts, domain
        benchmark = evaluate_genome(challenger)
        task = next(row for row in benchmark["tasks"] if row["name"] == f"domain:{domain}")
        assert task["ok"] is True, (domain, task)


def test_polynomials_real_curriculum_study_reaches_real_benchmark(tmp_path: Path):
    from mathesis.architecture import generate_challengers
    from mathesis.benchmark import evaluate_genome

    curriculum = MathematicalCurriculum(tmp_path)
    (tmp_path / "curriculum.json").write_text(
        json.dumps({"version": 2, "cursor": list(DOMAIN_ATLAS).index("polynomials"), "studies": [], "retry_counts": {}}),
        encoding="utf-8",
    )
    curriculum.researcher.search = lambda query, max_sources=3: {
        "ok": True,
        "sources": [{
            "url": "https://docs.sympy.org/latest/",
            "domain": "docs.sympy.org",
            "authority_hint": 0.95,
            "excerpt": "polynomial documentation evidence only",
        }],
        "errors": [],
    }
    study = curriculum.study_once()
    assert study["ok"] is True and study["domain"] == "polynomials"

    champion = default_genome()
    feedback = ExperienceAnalyzer(tmp_path).signals(champion, latest_study=study)
    assert "missing_polynomials_expert" in feedback["weaknesses"]
    challenger = generate_challengers(champion, weaknesses=feedback["weaknesses"], count=1)[0]
    assert "polynomials" in challenger.experts

    benchmark = evaluate_genome(challenger)
    polynomial_task = next(row for row in benchmark["tasks"] if row["name"] == "domain:polynomials")
    assert polynomial_task["ok"] is True
    assert polynomial_task["detail"] == "verified capability"


def test_benchmark_rejects_duplicate_and_disconnected_expert_gaming():
    from mathesis.benchmark import evaluate_genome

    duplicated = default_genome()
    duplicated.experts.append("algebra")
    duplicate_bench = evaluate_genome(duplicated)
    assert duplicate_bench["ok"] is False
    assert "architecture:unique_experts" in duplicate_bench["critical_failures"]

    disconnected = default_genome()
    disconnected.experts.append("polynomials")
    disconnected_bench = evaluate_genome(disconnected)
    assert disconnected_bench["ok"] is False
    assert "architecture:experts_connected" in disconnected_bench["critical_failures"]


def test_unused_budget_growth_cannot_improve_benchmark_score():
    from dataclasses import replace
    from mathesis.benchmark import evaluate_genome

    baseline = default_genome()
    bloated = replace(
        baseline,
        max_proof_cells=256,
        discovery_beam=8,
        research_budget=4,
        genome_id="bloated-without-capability",
    )
    baseline_bench = evaluate_genome(baseline)
    bloated_bench = evaluate_genome(bloated)
    assert bloated_bench["capability_score"] == baseline_bench["capability_score"]
    assert bloated_bench["neural_accuracy"] == baseline_bench["neural_accuracy"]
    assert bloated_bench["score"] < baseline_bench["score"]


def test_engine_keeps_evolved_discovery_budgets_without_restart(tmp_path: Path):
    engine = MathesisOmega(tmp_path)
    result = engine.answer("scrivimi il tuo prossimo modello")
    assert result["ok"] is True
    assert engine.discovery.symbolic_depth == engine.genome.symbolic_depth
    assert engine.discovery.discovery_beam == engine.genome.discovery_beam


def test_health_gate_rejects_false_theorem_even_with_verified_metadata(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    evolution.evolve_once()

    statement = "x+1=x+2"
    theorem_id = __import__("hashlib").sha256(statement.encode("utf-8")).hexdigest()[:20]
    discoveries = {
        "version": 3,
        "cycle": 1,
        "theorems": {
            theorem_id: {
                "id": theorem_id,
                "statement": statement,
                "verified": True,
                "strategy": "difference_of_powers",
                "quality_gate": "structurally_nontrivial_and_proof_gated",
                "certificate": {"ok": True, "status": "verified"},
            }
        },
        "discarded": {},
    }
    (tmp_path / "discoveries.json").write_text(json.dumps(discoveries), encoding="utf-8")
    report = health_report(tmp_path)
    assert report["ok"] is False
    quality = next(row for row in report["failed"] if row["name"] == "discovery:verified_quality")
    assert theorem_id in str(quality["detail"])


def test_health_gate_rejects_corrupted_discovery_state(tmp_path: Path):
    evolution = SelfEvolutionEngine(tmp_path)
    evolution.evolve_once()
    (tmp_path / "discoveries.json").write_text("{not-json", encoding="utf-8")
    report = health_report(tmp_path)
    assert report["ok"] is False
    assert any(row["name"] == "discovery:schema" for row in report["failed"])


def test_evolution_prefers_gate_eligible_challenger_over_higher_invalid_score(tmp_path: Path, monkeypatch):
    from dataclasses import replace
    import mathesis.evolution as evolution_module

    champion = default_genome()
    bad = replace(champion, generation=1, genome_id="bad-high-score", parent_id=champion.genome_id)
    good = replace(champion, generation=1, genome_id="good-safe-score", parent_id=champion.genome_id)

    monkeypatch.setattr(
        evolution_module,
        "generate_challengers",
        lambda champion, weaknesses=None, count=3: [bad, good],
    )

    def fake_benchmark(genome, router=None, *, learned_theorems=None):
        if genome.genome_id == "bad-high-score":
            return {
                "ok": False,
                "score": 100.0,
                "critical_failures": ["proof:poisoned"],
                "weaknesses": ["proof:poisoned"],
            }
        if genome.genome_id == "good-safe-score":
            return {
                "ok": True,
                "score": 20.0,
                "critical_failures": [],
                "weaknesses": [],
            }
        return {
            "ok": True,
            "score": 10.0,
            "critical_failures": [],
            "weaknesses": [],
        }

    monkeypatch.setattr(evolution_module, "evaluate_genome", fake_benchmark)
    result = evolution_module.SelfEvolutionEngine(tmp_path).evolve_once()
    assert result.promoted is True
    assert result.candidate["genome_id"] == "good-safe-score"
    assert result.benchmark["selected_trial"] == 1


def test_continuum_serializes_state_writers_without_force_push():
    workflow = (ROOT / ".github" / "workflows" / "mathesis-continuum.yml").read_text(encoding="utf-8")
    assert "group: mathesis-omega-continuum" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "git push --quiet origin HEAD:mathesis-state" in workflow
    assert "git push --force" not in workflow
    assert "git push -f" not in workflow


def test_self_rewrite_whitelist_rejects_nested_and_unknown_paths(tmp_path: Path):
    kernel = IntegrityKernel()
    allowed = tmp_path / "candidate_model.py"
    assert kernel.validate_state_path(tmp_path, allowed) == allowed.resolve()

    with pytest.raises(PermissionError):
        kernel.validate_state_path(tmp_path, tmp_path / "not-authorized.py")
    with pytest.raises(PermissionError):
        kernel.validate_state_path(tmp_path, tmp_path / "nested" / "candidate_model.py")
    with pytest.raises(PermissionError):
        kernel.validate_state_path(tmp_path, tmp_path)
    with pytest.raises(PermissionError):
        kernel.validate_state_path(tmp_path, tmp_path.parent / "escape.py")
