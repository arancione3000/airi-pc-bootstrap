# MATHESIS-Ω v0

MATHESIS-Ω is an experimental **proof-gated mathematical agent** inside Airi-PC.

It is deliberately not described as infallible. Its design goal is stronger:
creative components are allowed to be wrong, but important mathematical claims
are not labelled `VERIFIED` until independent checkers agree.

## Architecture

The v0 implementation combines:

1. **Natural-language Formalizer Mesh** — independent keyword, structural and
   semantic votes convert a request into a typed mathematical intent.
2. **Safe mathematical IR** — Python AST parsing is used only as syntax; calls,
   attributes and arbitrary evaluation are not accepted. The AST is converted
   into SymPy objects by a whitelist.
3. **Proof Cells** — reasoning is represented as a graph of goals,
   assumptions, proof obligations, candidate results and verification states.
4. **Independent verifiers**:
   - SymPy exact/symbolic normalization,
   - Z3 satisfiability/validity checking,
   - Lean 4 kernel checks for applicable closed natural-number equalities,
   - bounded counterexample search as an adversarial falsifier.
5. **Program Synthesis Arena** — multiple restricted algorithm candidates are
   generated for supported tasks, AST-audited, executed with tiny whitelisted
   builtins, property-tested against mathematical references, and scored.
6. **Growing Neural Router** — a tiny MLP implemented directly with Python
   lists, `math.tanh`, softmax and backpropagation. Evolution may add hidden
   units, but the neural output is advisory and never overrides proof gates.
7. **Read-only Researcher** — reuses Airi-PC's HTTPS-only public-web reader.
   Web pages can suggest evidence/hypotheses; web consensus alone cannot produce
   a `VERIFIED` mathematical claim.
8. **Mathematical Knowledge Graph** — stores statements, proof certificates,
   source metadata and dependency IDs.
9. **Transactional Self-Evolution** — candidate architecture genomes can grow
   experts, proof-search order, counterexample budgets and neural capacity.
   Champion replacement requires benchmark improvement and no critical
   regression.
10. **Immutable verifier kernel** — `safe_math.py`, `verifiers.py` and
    `kernel.py` are hashed before/after an evolution cycle. A candidate cannot
    promote if this boundary changes.

## "Write your next model"

The request:

```text
scrivimi il tuo prossimo modello
```

is a first-class intent. MATHESIS:

1. loads the current champion,
2. generates a challenger architecture,
3. writes `candidate_model.py` inside its state sandbox,
4. grows the neural router,
5. runs the benchmark arena,
6. hashes/rechecks the immutable kernel,
7. promotes only if the challenger is measurably better,
8. writes `champion_model.py` and persistent weights on promotion,
9. records every attempt in `history.jsonl`.

It does **not** rewrite the verifier/security kernel. Self-improvement is
transactional model/strategy rewriting, not unrestricted self-modification.

## Supported v0 requests

Examples:

```bash
PYTHONPATH=computer python -m mathesis.cli "calcola 2+3*4"
PYTHONPATH=computer python -m mathesis.cli "risolvi 2*x+3=7"
PYTHONPATH=computer python -m mathesis.cli "dimostra che (x+1)^2=x^2+2*x+1"
PYTHONPATH=computer python -m mathesis.cli "scrivi una funzione mcd"
PYTHONPATH=computer python -m mathesis.cli "scrivimi il tuo prossimo modello"
PYTHONPATH=computer python -m mathesis.cli --status
```

Program synthesis v0 supports `gcd`, `fibonacci`, `factorial`,
`is_prime` and `sort`.

Unknown requests **abstain** rather than pretending to understand them.

## Verification semantics

- **verified**: enough independent proof/check paths agree.
- **disproved**: a counterexample or exact solver refutes the statement.
- **checked_not_complete / not_fully_verified**: useful evidence exists but the
  proof gate was not fully satisfied.
- **evidence_only**: web evidence exists, but no mathematical proof was
  established.
- **abstained**: the request could not be formalized safely.

A finite counterexample search can refute a universal claim, but failure to find
a counterexample is never treated as a proof.

## Continuous autonomous evolution

`.github/workflows/mathesis-continuum.yml` runs a bounded champion/challenger
cycle approximately hourly and stores model state on the dedicated
`mathesis-state` branch.

`.github/workflows/mathesis-watchdog.yml` independently checks the state
heartbeat twice per hour and dispatches a recovery cycle if the state becomes
stale.

GitHub scheduling can be delayed by the platform; the watchdog improves
resilience but cannot promise literal zero downtime during an external GitHub
outage.

## Security boundaries

- web: HTTPS GET only through the existing hardened Airi reader;
- no remote writes, credentials or private-network targets;
- generated program candidates cannot import modules or access attributes;
- self-rewrite targets must remain inside `MATHESIS_STATE_DIR`;
- verifier-kernel files are outside the mutable set;
- no candidate is promoted solely because it is newer, larger or faster.

## Dependencies

Runtime-specific dependencies are isolated in
`computer/mathesis/requirements.txt`. Lean is optional at runtime; CI installs
Lean 4.34.0 and verifies the adapter with the actual kernel.

## Current limits

This is a serious v0 architecture, not a claim that we created the world's most
powerful AI. Natural-language coverage and theorem domains are intentionally
small. The important property is that unsupported tasks abstain and that the
architecture can be extended while retaining independently testable proof and
promotion gates.
