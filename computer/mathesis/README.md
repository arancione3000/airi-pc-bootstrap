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


## v1 autonomous discovery architecture

MATHESIS-Ω v1 extends the v0 proof-gated solver with a bounded mathematical
discovery loop.

Each fast cycle now performs:

1. generate one bounded conjecture candidate;
2. attempt exact symbolic + SMT/counterexample verification;
3. store only proof-gated discoveries as mathematical knowledge;
4. periodically study read-only public mathematical sources;
5. generate multiple architecture challengers;
6. benchmark each challenger across mathematical domains;
7. promote only the best challenger if the verifier kernel stayed unchanged and
   no critical regression appeared;
8. persist model, router, discoveries, curriculum and history on
   `mathesis-state`.

### SymPy mathematical laboratory

The v1 lab exposes domain-specific operations for algebra, polynomials,
calculus, trigonometry, number theory, linear algebra, combinatorics, equations,
inequalities, sequences, special functions, geometry, probability,
discrete mathematics and optimization.

Generic `simplify()` is not treated as a proof oracle. The implementation
prefers explicit transformations such as expansion/factorization and verifies
results independently when possible.

### Autonomous discoveries

The discovery engine currently explores bounded families including binomial
identities, finite geometric identities, difference-of-powers identities and
Faulhaber-style sum-of-powers formulas inferred by exact interpolation and
checked through base cases plus finite-difference induction steps.

A discovery is labelled:

- `internal_novelty: true` when it is new to the persistent MATHESIS state;
- `human_novelty: unassessed` unless an external scholarly process establishes
  otherwise.

MATHESIS must never infer "new to humanity" merely because a web search failed
to find a matching formula.

### Evolving architecture

The mutable model is an architecture DSL containing experts, graph topology,
neural width, proof-cell budget, symbolic depth, conjecture beam and research
budget. These values affect the following discovery cycles; they are not
decorative metadata.

Three bounded challenger styles are tried by default on every cycle:

- broader mathematical coverage;
- deeper proof/discovery search;
- smaller/faster architecture.

The verifier kernel remains immutable across a promotion.

### Continuous operation

The fast continuum is scheduled every five minutes with a five-minute job
timeout. A duplicate-aware watchdog runs every ten minutes and dispatches a
recovery only if state is stale and no continuum job is already active.

GitHub Actions scheduling is best-effort. This design provides continuous
attempted evolution while GitHub Actions is available; it cannot promise
literal zero downtime during provider outages, queue delays or account limits.

A separate deep workflow uses Lean 4.34.0 + Mathlib 4.34.0 on a slower cadence,
with caches for the heavy theorem library.

### What v1 does not claim

MATHESIS does not literally contain all mathematics ever written, cannot prove
every true statement, and is not guaranteed error-free. Its design goal is to
expand verified capability over time while explicitly abstaining or rejecting
results that do not pass the available proof gates.
