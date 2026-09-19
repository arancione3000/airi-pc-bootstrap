from __future__ import annotations

from typing import Any

from .formalizer import FormalizerMesh
from .neural_graph import GrowingNeuralRouter
from .synthesis import ProgramSynthesizer
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
    ("che bel tempo", "unknown"),
]


def evaluate_genome(genome: ArchitectureGenome, router: GrowingNeuralRouter | None = None) -> dict[str, Any]:
    formalizer = FormalizerMesh()
    verifier = CompositeVerifier(counterexample_radius=genome.counterexample_radius)
    synth = ProgramSynthesizer()

    tasks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, critical: bool = True, detail: Any = None):
        tasks.append({"name": name, "ok": bool(ok), "critical": critical, "detail": detail})

    intent_cases = [
        ("calcola 10+15", "arithmetic"),
        ("risolvi 3*x=12", "equation"),
        ("dimostra che (x+1)^2=x^2+2*x+1", "identity"),
        ("scrivimi il tuo prossimo modello", "evolve_model"),
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
        cert = verifier.verify_relation("(x+1)^2 = x^2+2*x+1")
        add("proof:algebra_identity", cert.ok, detail=cert.status)
    except Exception as exc:
        add("proof:algebra_identity", False, detail=repr(exc))

    try:
        candidate = synth.synthesize("gcd")
        add("program:gcd", candidate.verified, detail=candidate.score)
    except Exception as exc:
        add("program:gcd", False, detail=repr(exc))

    if "number_theory" in genome.experts:
        try:
            candidate = synth.synthesize("is_prime")
            add("program:is_prime", candidate.verified, detail=candidate.score)
        except Exception as exc:
            add("program:is_prime", False, detail=repr(exc))
    else:
        add("program:is_prime", False, critical=False, detail="missing number_theory expert")

    if "research" in genome.experts:
        add("capability:research", True, critical=False, detail="read-only researcher enabled")
    else:
        add("capability:research", False, critical=False, detail="research expert not yet in genome")

    router = router or GrowingNeuralRouter(hidden_size=genome.neural_hidden)
    router.train(TRAINING_PHRASES, epochs=60, lr=0.07)
    neural_accuracy = router.accuracy(VALIDATION_PHRASES)
    add("neural:intent_router", neural_accuracy >= 0.55, critical=False, detail=round(neural_accuracy, 4))

    passed = sum(task["ok"] for task in tasks)
    critical_failures = [task["name"] for task in tasks if task["critical"] and not task["ok"]]
    capability_score = passed / len(tasks)
    complexity_penalty = (
        0.0005 * len(genome.experts)
        + 0.00002 * genome.neural_hidden
        + 0.00001 * genome.counterexample_radius
    )
    score = 80.0 * capability_score + 20.0 * neural_accuracy - complexity_penalty

    return {
        "ok": not critical_failures,
        "score": round(score, 6),
        "capability_score": round(capability_score, 6),
        "neural_accuracy": round(neural_accuracy, 6),
        "critical_failures": critical_failures,
        "tasks": tasks,
        "complexity_penalty": complexity_penalty,
    }
