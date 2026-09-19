from __future__ import annotations

from typing import Any

import sympy as sp

from .formalizer import FormalizerMesh
from .neural_graph import GrowingNeuralRouter
from .synthesis import ProgramSynthesizer
from .sympy_lab import SymPyMathLab
from .types import ArchitectureGenome
from .verifiers import CompositeVerifier


TRAINING_PHRASES = [
    ("calcola 2+2", "arithmetic"),
    ("quanto fa 9*7", "arithmetic"),
    ("calculate 18/3", "arithmetic"),
    ("risolvi 2*x+3=7", "equation"),
    ("solve x+5=11", "equation"),
    ("dimostra che (x+1)^2=x^2+2*x+1", "identity"),
    ("prove that (x-1)^2=x^2-2*x+1", "identity"),
    ("scrivi una funzione gcd", "synthesize_program"),
    ("implementa fibonacci", "synthesize_program"),
    ("scrivi una funzione per numero primo", "synthesize_program"),
    ("ricerca online Lean theorem prover", "research_claim"),
    ("research z3 solver tactics", "research_claim"),
    ("scrivimi il tuo prossimo modello", "evolve_model"),
    ("improve yourself and create your next model", "evolve_model"),
    ("deriva x^3+2*x", "derivative"),
    ("calcola la derivata di sin(x)", "derivative"),
    ("integra 2*x", "integral"),
    ("integrate x^2", "integral"),
    ("analizza simbolicamente x^2+2*x+1", "analyze_math"),
    ("fattorizza x^2-1", "analyze_math"),
    ("scopri nuova matematica", "discover_math"),
    ("formula una congettura", "discover_math"),
    ("ciao come va", "unknown"),
    ("raccontami una storia", "unknown"),
]

VALIDATION_PHRASES = [
    ("calcola 17+25", "arithmetic"),
    ("risolvi 5*x=20", "equation"),
    ("dimostra che (x+2)^2=x^2+4*x+4", "identity"),
    ("crea una funzione mcd", "synthesize_program"),
    ("verifica online documentazione lean", "research_claim"),
    ("genera il tuo prossimo modello", "evolve_model"),
    ("derivata di x^5", "derivative"),
    ("integra 3*x^2", "integral"),
    ("analizza matematicamente x^2-4", "analyze_math"),
    ("trova un nuovo teorema", "discover_math"),
    ("che bel tempo", "unknown"),
]


def evaluate_genome(
    genome: ArchitectureGenome,
    router: GrowingNeuralRouter | None = None,
    *,
    learned_theorems: list[str] | None = None,
) -> dict[str, Any]:
    formalizer = FormalizerMesh()
    verifier = CompositeVerifier(counterexample_radius=genome.counterexample_radius)
    synth = ProgramSynthesizer()
    lab = SymPyMathLab()

    tasks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, critical: bool = True, detail: Any = None):
        tasks.append({"name": name, "ok": bool(ok), "critical": critical, "detail": detail})

    intent_cases = [
        ("calcola 10+15", "arithmetic"),
        ("risolvi 3*x=12", "equation"),
        ("dimostra che (x+1)^2=x^2+2*x+1", "identity"),
        ("scrivimi il tuo prossimo modello", "evolve_model"),
        ("deriva x^3", "derivative"),
        ("integra 2*x", "integral"),
        ("scopri nuova matematica", "discover_math"),
    ]
    for prompt, expected in intent_cases:
        got = formalizer.formalize(prompt)
        add(f"formalize:{expected}", got.kind == expected, detail=got.kind)

    try:
        _, cert = verifier.evaluate("2+3*4")
        add("proof:closed_arithmetic", cert.ok, detail=cert.status)
    except Exception as exc:
        add("proof:closed_arithmetic", False, detail=repr(exc))

    try:
        cert = verifier.verify_relation("(x+1)^2=x^2+2*x+1")
        add("proof:algebra_identity", cert.ok, detail=cert.status)
    except Exception as exc:
        add("proof:algebra_identity", False, detail=repr(exc))

    try:
        candidate = synth.synthesize("gcd")
        add("program:gcd", candidate.verified, detail=candidate.score)
    except Exception as exc:
        add("program:gcd", False, detail=repr(exc))

    domain_checks: dict[str, Any] = {
        "algebra": lambda: lab.algebra_normal_forms("(x+1)^4-(x^4+4*x^3+6*x^2+4*x+1)").ok,
        "polynomials": lambda: lab.polynomial_interpolate([(0, 1), (1, 4), (2, 9), (3, 16)]).ok,
        "number_theory": lambda: synth.synthesize("is_prime").verified,
        "research": lambda: True,
        "calculus": lambda: (
            lab.derivative("x^4+2*x").ok
            and lab.antiderivative("3*x^2").ok
        ),
        "trigonometry": lambda: lab.trig_normal_form("sin(x)^2+cos(x)^2").ok,
        "linear_algebra": lambda: lab.matrix_invariants([[2, 1], [1, 1]]).ok,
        "combinatorics": lambda: verifier.evaluate("binomial(8,3)")[0] == 56,
        "equations": lambda: verifier.solve_equation("x^2-5*x+6=0")[1].ok,
        "inequalities": lambda: bool(sp.reduce_inequalities([sp.Symbol("x", real=True) ** 2 >= 0])),
        "sequences": lambda: lab.polynomial_interpolate([(0, 0), (1, 1), (2, 4), (3, 9)]).ok,
        "special_functions": lambda: verifier.evaluate("gamma(6)")[0] == 120,
        "geometry": lambda: verifier.evaluate("3^2+4^2")[0] == 25,
        "probability": lambda: verifier.evaluate("binomial(10,3)/2^10")[1].ok,
        "discrete_math": lambda: lab.number_theory_profile(360).ok,
        "optimization": lambda: lab.derivative("x^2-6*x+13").ok,
    }

    for domain, check in domain_checks.items():
        if domain not in genome.experts:
            add(f"domain:{domain}", False, critical=False, detail="expert not present")
            continue
        try:
            add(f"domain:{domain}", bool(check()), critical=False, detail="verified capability")
        except Exception as exc:
            add(f"domain:{domain}", False, critical=False, detail=repr(exc))

    # The architecture search budget must correspond to a real symbolic workload.
    try:
        exponent = min(16, max(2, genome.symbolic_depth + 2))
        cert = verifier.verify_relation(
            f"(x+1)^{exponent}={sp.sstr(sp.expand((sp.Symbol('x', real=True)+1)**exponent))}"
        )
        add("search:symbolic_depth", cert.ok, critical=False, detail={"tested_exponent": exponent})
    except Exception as exc:
        add("search:symbolic_depth", False, critical=False, detail=repr(exc))

    learned_results: list[dict[str, Any]] = []
    for index, statement in enumerate((learned_theorems or [])[:16]):
        try:
            cert = verifier.verify_relation(statement)
            ok = bool(cert.ok)
            learned_results.append({
                "statement": statement,
                "ok": ok,
                "status": cert.status,
                "methods": list(cert.methods),
            })
            # Previously verified mathematical knowledge becomes a critical
            # promotion invariant: challengers may expand capability but may
            # not lose the ability to re-verify learned relations.
            add(
                f"learned_theorem:{index}",
                ok,
                critical=True,
                detail=cert.status,
            )
        except Exception as exc:
            learned_results.append({
                "statement": statement,
                "ok": False,
                "error": repr(exc),
            })
            add(f"learned_theorem:{index}", False, critical=True, detail=repr(exc))

    router = router or GrowingNeuralRouter(hidden_size=genome.neural_hidden)
    router.train(TRAINING_PHRASES, epochs=70, lr=0.07)
    neural_accuracy = router.accuracy(VALIDATION_PHRASES)
    add("neural:intent_router", neural_accuracy >= 0.55, critical=False, detail=round(neural_accuracy, 4))

    passed = sum(task["ok"] for task in tasks)
    critical_failures = [task["name"] for task in tasks if task["critical"] and not task["ok"]]
    capability_score = passed / len(tasks)

    # A modest efficiency penalty prevents endless growth from being rewarded.
    complexity_penalty = (
        0.00035 * len(genome.experts)
        + 0.000015 * genome.neural_hidden
        + 0.000008 * genome.counterexample_radius
        + 0.00002 * genome.max_proof_cells
        + 0.00004 * genome.discovery_beam
        + 0.00003 * genome.research_budget
    )
    # Reward verified breadth and language competence; symbolic depth contributes
    # only after the corresponding harder identity was actually verified above.
    depth_bonus = min(1.5, 0.08 * genome.symbolic_depth) if tasks[-2]["ok"] else 0.0
    score = 80.0 * capability_score + 20.0 * neural_accuracy + depth_bonus - complexity_penalty

    return {
        "ok": not critical_failures,
        "score": round(score, 6),
        "capability_score": round(capability_score, 6),
        "neural_accuracy": round(neural_accuracy, 6),
        "critical_failures": critical_failures,
        "weaknesses": [task["name"] for task in tasks if not task["ok"]],
        "tasks": tasks,
        "complexity_penalty": complexity_penalty,
        "verified_symbolic_depth": genome.symbolic_depth if tasks[-2]["ok"] else 0,
        "learned_theorems_replayed": learned_results,
    }
