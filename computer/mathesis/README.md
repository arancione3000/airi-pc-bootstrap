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
cycle on an approximately five-minute self-handoff cadence, with cron as a
recovery trigger, and stores model state on the dedicated `mathesis-state`
branch.

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

The fast continuum is scheduled every five minutes with a 15-minute job
timeout so cold setup plus persistence/handoff have enough margin. A
duplicate-aware watchdog runs every ten minutes and dispatches a recovery only
if state is stale and no continuum job is already active.

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


### Experience feedback and anti-forgetting

Verified mathematical work now affects later architecture search instead of
remaining passive history.

The experience layer converts bounded evidence into architecture hints:

- verified binomial work can prioritize combinatorics;
- verified Faulhaber/geometric work can prioritize sequences;
- rejected conjectures can prioritize stronger search/falsification;
- curriculum domains not yet represented by an expert can be proposed as gaps.

These hints can only modify the bounded architecture DSL. They cannot rewrite
the immutable verifier kernel or turn web text into mathematical truth.

Previously verified relation-style discoveries are also replayed during every
champion/challenger benchmark. They are critical promotion invariants: a new
champion is rejected if it can no longer re-verify learned mathematical
relations. This provides a bounded anti-forgetting mechanism.

The five-minute GitHub schedule is written as an explicit minute list
(3,8,13,...,58) and is backed by the duplicate-aware watchdog. GitHub Actions
scheduling remains provider best-effort and can be delayed externally.


### Continuous self-handoff

Cron is no longer the only mechanism that keeps MATHESIS alive. After a
continuum run persists its state, it waits until the five-minute cadence and
uses GitHub's `workflow_dispatch` API to hand execution to the next run.
Before dispatching it checks for another queued/in-progress continuum run and
avoids duplicates. The normal five-minute cron and watchdog remain recovery
paths.

This means a delayed GitHub `schedule` event does not by itself stop the
evolution chain. Provider outages, disabled Actions, permission changes or
account-level limits can still interrupt execution; no repository workflow can
guarantee literal zero downtime during an external platform outage.

Verified discovery-domain gaps also remain active architecture priorities until
the corresponding expert is actually present, so useful mathematical feedback
is not forgotten when another challenger wins first.

The read-only curriculum cadence is tuned so a champion with research budget 2
attempts external mathematical study roughly every 15 minutes while the
five-minute chain is healthy.


### Domain closure invariant

Every mathematical domain advertised by the SymPy laboratory must also be an
evolvable architecture expert. CI enforces this set relationship. Polynomial
mathematics is explicitly benchmarked and can now be added by challenger
architectures when curriculum feedback reports it as missing.


The symbolic-depth promotion bonus is tied to the explicit
`search:symbolic_depth` verification result, not to a positional task index.
This prevents anti-forgetting replay tasks from accidentally changing the
meaning of architecture scores.


Legacy Faulhaber discoveries created before the nontrivial-induction gate are
automatically re-proved from their stored polynomial. Their base case, exact
sample points and a structurally nontrivial finite-difference recurrence are
checked again. A legacy result remains active only if that upgrade succeeds;
otherwise it is archived outside the active theorem corpus.


Curriculum failures are retried on the same mathematical domain rather than
silently skipping it. A successful retry clears the retry counter; after three
consecutive failures the cursor advances so a temporarily unreachable source
cannot stall the whole curriculum indefinitely.


### Network-bounded autonomous handoff

GitHub API calls used by the continuum handoff, watchdog recovery and deep
Mathlib kick are explicitly connection-bounded, total-time-bounded and retried.
A slow API response therefore cannot hold an autonomous runner indefinitely.
The continuum job keeps a larger overall timeout margin so a cold environment
can still finish mathematical work, persist state and hand off safely.


If the active-run lookup itself fails after bounded retries, the continuum
attempts one serialized successor dispatch rather than abandoning the chain.
The GitHub concurrency group limits overlap. The watchdog treats an
unreadable continuum-run listing as a stale/uncertain condition and attempts
the normal recovery path.


### Structured anti-forgetting

Anti-forgetting is not limited to theorem statements that fit the simple
relation parser. Verified Faulhaber discoveries replay their structurally
nontrivial induction recurrence as a critical promotion obligation. This keeps
rich sum-of-powers results represented in future champion benchmarks without
pretending the relation parser directly understands the full quantified sum
schema.

Persistent curriculum metadata is migrated on read to schema v2, including
per-domain retry counters. Status reports the effective schema version.

### Persistent-state health gate

Every autonomous continuum cycle runs a state health audit before pushing
anything to the persistent `mathesis-state` branch. The audit checks
architecture/domain closure, topology endpoints, genome bounds, router shape,
verified discovery quality, Faulhaber recurrence certificates, curriculum retry
metadata, benchmark critical failures and verifier/kernel status.

A failed audit exits before persistence, so the previous remote state remains
the last known-good checkpoint.


### 2026-09-20 proof-integrity audit

The persistent discovery corpus is now treated as untrusted input when it is
loaded. A stored `verified=true` flag is not sufficient to enter health or
anti-forgetting:

- relation-style discoveries are rechecked with the current CompositeVerifier;
- structural tautologies are rejected;
- Faulhaber rows are rebuilt from the stored polynomial, and the displayed
  theorem must match both the stored power and the re-proved polynomial;
- Faulhaber base case, bounded exact samples and the nontrivial finite-
  difference recurrence are rechecked;
- legacy sample counts are bounded during migration so corrupted state cannot
  request an unbounded revalidation loop;
- failed legacy reproofs are archived outside the active corpus;
- schema-v3 upgrades are persisted even when no theorem needed moving;
- migrations are idempotent.

Anti-forgetting consumes the complete currently valid active discovery corpus by
default, rather than only the newest fixed-size window. A theorem that is
discarded, unverified, structurally trivial, certificate-inconsistent or
rejected by the current verifier is not turned into a promotion obligation.

Novelty metadata is deliberately separated into three claims: novelty relative
to the MATHESIS memory, novelty relative to consulted sources, and novelty to
human mathematics. Only memory novelty is established automatically. Source
novelty and human novelty remain `unassessed` unless a separate scholarly
comparison establishes them.

The architecture benchmark also verifies the genome itself. Duplicate or
unknown experts/strategies, disconnected expert nodes, invalid topology edges,
unknown proof-order methods, broken research-strategy coupling and out-of-bounds
architecture parameters are critical failures. Every SymPy lab
domain has a real domain probe, and CI exercises the path from curriculum
evidence to weakness, challenger expert, graph membership and benchmarked
capability. Merely growing budgets cannot improve the score unless the
corresponding measured capability improves.

Evolution selects the best challenger only after hard eligibility gates are
applied; an invalid high-scoring candidate cannot hide a lower-scoring safe
candidate. Interactive promotion also rebuilds the discovery engine with the
promoted symbolic-depth and discovery-beam settings immediately, without
requiring a process restart.

Self-rewrite path validation is a strict direct-file whitelist under
`MATHESIS_STATE_DIR`. Nested paths, unknown filenames, the state-directory
root and paths outside the state directory are rejected. Generated model
modules remain data-only and cannot alter the verifier kernel, network policy,
workflow permissions, promotion gate, secrets or host boundary.

The continuum serializes all state writers through the
`mathesis-omega-continuum` concurrency group and never force-pushes the
`mathesis-state` branch. Health validation runs before persistence, so a
failed state cannot replace the last known-good remote checkpoint. The health
gate also fails closed on malformed champion/discovery/curriculum state instead
of silently accepting fallback defaults or crashing on corrupted field types.

### Strict persisted-state health checks

Before a MATHESIS checkpoint is considered healthy, the persisted `champion.json`
must contain the complete required architecture genome schema. Partial JSON is
rejected rather than silently completed with constructor defaults. When runtime
status is present, the health gate also requires SymPy, Z3, and Lean availability.
A failed health audit still stops before persistent state replacement.
