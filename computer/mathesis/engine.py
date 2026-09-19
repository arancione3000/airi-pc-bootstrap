from __future__ import annotations

from pathlib import Path
from typing import Any

from .architecture import architecture_report
from .evolution import SelfEvolutionEngine
from .discovery import ConjectureDiscoveryEngine
from .curriculum import MathematicalCurriculum
from .formalizer import FormalizerMesh
from .knowledge import KnowledgeGraph, default_state_dir
from .proof_cells import ProofCellGraph
from .research import ReadOnlyResearcher
from .safe_math import parse_relation
from .synthesis import ProgramSynthesizer
from .sympy_lab import SymPyMathLab
from .verifiers import CompositeVerifier


class MathesisOmega:
    """Proof-gated mathematical agent prototype.

    Natural-language interpretation is probabilistic/heuristic. Mathematical
    acceptance is not: important answers carry a certificate or are explicitly
    marked unverified/unknown.
    """

    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir or default_state_dir()).resolve()
        self.formalizer = FormalizerMesh()
        self.evolution = SelfEvolutionEngine(self.state_dir)
        self.genome = self.evolution.load_champion()
        self.verifier = CompositeVerifier(counterexample_radius=self.genome.counterexample_radius)
        self.synthesizer = ProgramSynthesizer()
        self.researcher = ReadOnlyResearcher()
        self.knowledge = KnowledgeGraph(self.state_dir)
        self.math_lab = SymPyMathLab()
        self.discovery = ConjectureDiscoveryEngine(
            self.state_dir,
            counterexample_radius=self.genome.counterexample_radius,
            symbolic_depth=self.genome.symbolic_depth,
            discovery_beam=self.genome.discovery_beam,
        )
        self.curriculum = MathematicalCurriculum(self.state_dir)

    def understand(self, prompt: str) -> dict[str, Any]:
        intent = self.formalizer.formalize(prompt)
        router = self.evolution.load_router(self.genome)
        neural_kind, neural_confidence = router.predict(prompt)
        result = intent.to_dict()
        result["neural_router"] = {
            "kind": neural_kind,
            "confidence": round(float(neural_confidence), 6),
            "advisory_only": True,
        }
        return result

    def _graph(self, goal: str) -> tuple[ProofCellGraph, str]:
        graph = ProofCellGraph(max_cells=self.genome.max_proof_cells)
        root = graph.add(goal, strategy="orchestrate")
        return graph, root.cell_id

    def answer(self, prompt: str) -> dict[str, Any]:
        intent = self.formalizer.formalize(prompt)
        graph, root_id = self._graph(intent.raw)
        base = {
            "ok": False,
            "input": prompt,
            "intent": intent.to_dict(),
            "architecture": self.genome.genome_id,
        }

        try:
            if intent.kind == "arithmetic":
                expression = intent.expression or intent.raw
                cell = graph.add(expression, strategy="exact_symbolic_evaluation", parent=root_id)
                value, cert = self.verifier.evaluate(expression)
                graph.complete(cell.cell_id, cert.to_dict(), status="proved" if cert.ok else "unverified")
                graph.complete(root_id, value, status="proved" if cert.ok else "unverified")
                if cert.ok:
                    self.knowledge.record(expression, status="verified", certificate=cert.to_dict())
                return {
                    **base,
                    "ok": cert.ok,
                    "result": value,
                    "certificate": cert.to_dict(),
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "equation":
                statement = intent.expression or intent.raw
                cell = graph.add(statement, strategy="exact_solve_plus_completeness", parent=root_id)
                solutions, cert = self.verifier.solve_equation(statement, intent.variable)
                graph.complete(cell.cell_id, cert.to_dict(), status="proved" if cert.ok else "unverified")
                graph.complete(root_id, solutions, status="proved" if cert.ok else "unverified")
                if cert.ok:
                    self.knowledge.record(statement, status="verified", certificate=cert.to_dict())
                return {
                    **base,
                    "ok": cert.ok,
                    "solutions": solutions,
                    "certificate": cert.to_dict(),
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind in {"identity", "proposition"}:
                statement = (intent.expression or intent.raw).strip()
                if statement.lower().startswith("che "):
                    statement = statement[4:].strip()
                cell = graph.add(statement, strategy="multi_verifier_proof", parent=root_id)
                cert = self.verifier.verify_relation(statement)
                graph.complete(cell.cell_id, cert.to_dict(), status="proved" if cert.ok else cert.status)
                graph.complete(root_id, cert.status, status="proved" if cert.ok else cert.status)
                record_status = "verified" if cert.ok else ("disproved" if cert.status == "disproved" else "unknown")
                if record_status != "unknown":
                    self.knowledge.record(statement, status=record_status, certificate=cert.to_dict())
                return {
                    **base,
                    "ok": cert.ok,
                    "truth_status": cert.status,
                    "certificate": cert.to_dict(),
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "derivative":
                expression = intent.expression or intent.raw
                cell = graph.add(expression, strategy="sympy_rule_derivative_plus_numeric_check", parent=root_id)
                result = self.math_lab.derivative(expression, intent.variable or "x")
                graph.complete(cell.cell_id, result.to_dict(), status="proved" if result.ok else "unverified")
                graph.complete(root_id, result.result, status="proved" if result.ok else "unverified")
                return {
                    **base,
                    "ok": result.ok,
                    "math": result.to_dict(),
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "integral":
                expression = intent.expression or intent.raw
                cell = graph.add(expression, strategy="sympy_integrate_plus_differentiate_back", parent=root_id)
                result = self.math_lab.antiderivative(expression, intent.variable or "x")
                graph.complete(cell.cell_id, result.to_dict(), status="proved" if result.ok else "unverified")
                graph.complete(root_id, result.result, status="proved" if result.ok else "unverified")
                return {
                    **base,
                    "ok": result.ok,
                    "math": result.to_dict(),
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "analyze_math":
                expression = intent.expression or intent.raw
                cell = graph.add(expression, strategy="specific_sympy_normal_forms", parent=root_id)
                result = self.math_lab.algebra_normal_forms(expression)
                graph.complete(cell.cell_id, result.to_dict(), status="proved" if result.ok else "unverified")
                graph.complete(root_id, result.result, status="proved" if result.ok else "unverified")
                return {
                    **base,
                    "ok": result.ok,
                    "math": result.to_dict(),
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "discover_math":
                cell = graph.add("generate conjecture and require proof gate", strategy="autonomous_conjecture_discovery", parent=root_id)
                result = self.discovery.discover_once()
                graph.complete(cell.cell_id, result, status="proved" if result.get("ok") else "rejected")
                graph.complete(root_id, result.get("status"), status="proved" if result.get("ok") else "checked")
                return {
                    **base,
                    "ok": bool(result.get("ok")),
                    "discovery": result,
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "synthesize_program":
                task = intent.target or ""
                cell = graph.add(task, strategy="candidate_program_arena", parent=root_id)
                candidate = self.synthesizer.synthesize(task)
                graph.complete(cell.cell_id, candidate.to_dict(), status="proved" if candidate.verified else "failed")
                graph.complete(root_id, candidate.name, status="proved" if candidate.verified else "failed")
                return {
                    **base,
                    "ok": candidate.verified,
                    "program": candidate.to_dict(),
                    "certificate": {
                        "status": "verified_program" if candidate.verified else "rejected",
                        "methods": ["restricted_ast", "reference_properties", "candidate_arena"],
                        "tests": [candidate.tests_passed, candidate.tests_total],
                    },
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "research_claim":
                claim = intent.expression or intent.raw
                research_cell = graph.add(claim, strategy="https_read_only_research", parent=root_id)
                research = self.researcher.search(claim)
                graph.complete(research_cell.cell_id, {"sources": len(research["sources"])}, status="observed")

                certificate = None
                truth_status = "evidence_only"
                try:
                    parse_relation(claim)
                    proof_cell = graph.add(claim, strategy="formal_verification", parent=research_cell.cell_id)
                    cert = self.verifier.verify_relation(claim)
                    certificate = cert.to_dict()
                    truth_status = cert.status
                    graph.complete(proof_cell.cell_id, certificate, status="proved" if cert.ok else cert.status)
                except Exception:
                    pass

                graph.complete(root_id, truth_status, status="proved" if truth_status == "verified" else "observed")
                self.knowledge.record(
                    claim,
                    status=truth_status,
                    certificate=certificate,
                    sources=research["sources"],
                )
                return {
                    **base,
                    "ok": truth_status == "verified",
                    "truth_status": truth_status,
                    "research": research,
                    "certificate": certificate,
                    "proof_graph": graph.to_dict(),
                }

            if intent.kind == "evolve_model":
                cell = graph.add("create and benchmark next architecture", strategy="transactional_self_rewrite", parent=root_id)
                result = self.evolution.evolve_once()
                self.genome = self.evolution.load_champion()
                self.verifier = CompositeVerifier(counterexample_radius=self.genome.counterexample_radius)
                self.discovery = ConjectureDiscoveryEngine(
                    self.state_dir,
                    counterexample_radius=self.genome.counterexample_radius,
                )
                graph.complete(cell.cell_id, result.to_dict(), status="proved" if result.promoted else "checked")
                graph.complete(root_id, self.genome.genome_id, status="proved")
                return {
                    **base,
                    "ok": True,
                    "evolution": result.to_dict(),
                    "current_model": architecture_report(self.genome),
                    "proof_graph": graph.to_dict(),
                }

            neural = self.evolution.load_router(self.genome).predict(prompt)
            graph.complete(root_id, {"abstained": True}, status="unknown")
            return {
                **base,
                "ok": False,
                "status": "abstained",
                "reason": "request could not be formalized safely",
                "neural_hint": {"kind": neural[0], "confidence": neural[1]},
                "proof_graph": graph.to_dict(),
            }

        except Exception as exc:
            graph.fail(root_id, repr(exc))
            return {
                **base,
                "ok": False,
                "status": "error",
                "error": repr(exc),
                "proof_graph": graph.to_dict(),
            }

    def status(self) -> dict[str, Any]:
        self.genome = self.evolution.load_champion()
        return {
            "ok": True,
            "version": "0.2.0",
            "model": architecture_report(self.genome),
            "evolution": self.evolution.status(),
            "knowledge": self.knowledge.status(),
            "discovery": self.discovery.status(),
            "curriculum": self.curriculum.status(),
            "sympy_lab": self.math_lab.manifest(),
            "verifiers": self.verifier.diagnostics(),
            "guarantee": "proof-gated evolving system; not an infallibility or human-novelty guarantee",
        }
