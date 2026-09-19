from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "computer"))

from mathesis.architecture import default_genome
from mathesis.benchmark import TRAINING_PHRASES, VALIDATION_PHRASES
from mathesis.engine import MathesisOmega
from mathesis.discovery import ConjectureDiscoveryEngine
from mathesis.sympy_lab import SymPyMathLab, DOMAIN_ATLAS
from mathesis.evolution import SelfEvolutionEngine
from mathesis.experience import ExperienceAnalyzer
from mathesis.formalizer import FormalizerMesh
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
    assert "number_theory" in champion["experts"]
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
    assert cert["recurrence"]["ok"] is True


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
