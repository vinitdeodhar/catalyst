# Purl — Implementation Specification

**Purl: Compiler-Managed Cutting of Unbounded Quantum Loops**

Status: proposed · Audience: implementing engineer/agent · Substrate: PennyLane
Catalyst (MLIR `quantum` dialect) + a pure-NumPy reference simulator.

---

## 0. One-line

Purl is an MLIR compiler pass (plus a noise-modelling evaluation harness) that
detects *carry-type* dynamic quantum loops — loops that hold a live qubit across
a measurement-conditioned iteration — and, when profitable under real hardware
calibration, **cuts** the carried wire to bound its coherent depth, recovering
delivered-state fidelity that unbounded holding loses to decoherence.

## 1. Goals and overview

A dynamic quantum loop (`scf.while` conditioned on a `quantum.measure`) can hold
one qubit *live and un-measured* across every iteration — its **coherent depth**
(unbroken run with no measurement/reset) grows with the random trip count `k`,
so tail shots decohere and return garbage. Purl bounds that depth at a
compile-time-chosen period `C` by cutting the carried wire and continuing on a
fresh qubit, with a classical correction that keeps the ensemble statistics
exact.

The system has two halves that **share one real-hardware calibration JSON**:

1. **The Purl MLIR pass** (`--purl`) — classifies the loop, computes body depth,
   selects a cut *strategy* and period from a cost model driven by the hardware
   JSON, rewrites the loop, and emits a *predicted* delivered fidelity.
2. **A noise-modelling simulator** — a pure-NumPy trajectory simulator that reads
   the **same** JSON and *measures* delivered fidelity for each arm, providing an
   independent oracle that cross-validates the pass.

**End-to-end pipeline (the eval script exercises this for every benchmark):**

```
PennyLane/Catalyst benchmark (@qjit, catalyst.while_loop)
  → Catalyst lowers to MLIR (quantum dialect + scf.while)          [real frontend]
  → Purl pass applied to that MLIR (--purl, calib=ibm_eagle_r3.json) [real pass]
      → emits purl.strategy / purl.C / purl.predicted_fidelity
  → the simulator (SAME JSON) measures delivered fidelity per arm    [noise oracle]
  → table: pass-predicted vs simulator-measured, per benchmark & noise scale
```

The pass and the simulator are kept consistent *by the shared JSON and by
cross-validation* (predicted ≈ measured), not by executing the transformed IR on
a compiled backend (compiled execution is explicitly out of scope — see §9).

### Success
- All FileCheck unit tests green (§7).
- All three benchmarks lower to MLIR and the pass applies to the lowered MLIR (§6).
- The eval script runs the full pipeline for every benchmark and prints the
  table with a legend (§8), and pass-*predicted* fidelity agrees with
  simulator-*measured* fidelity to within **|ΔF| ≤ 0.02 at `lam=1`** (§8.4 S3).

### 1.1 Definitions

**Dynamic loop.** A *dynamic loop* is an `scf.while` operation whose continuation
predicate is **measurement-conditioned** — its trip count is a runtime random
variable with no compile-time bound, because the loop is re-entered based on
quantum measurement outcomes.

Precisely, let the loop's `before`-region terminator be `scf.condition(%c) ...`.
The loop is **dynamic** iff the boolean `%c` has a backward data-dependence on the
`mres` result of at least one `quantum.measure` in the loop body, tracing through
(i) classical glue — `tensor.extract` / `tensor.from_elements`, `stablehlo.*`,
`arith.*` — and (ii) the loop carry — a `before`-region block argument at index
*i* resolves to the *i*-th operand of the body's `scf.yield` (one hop across the
iteration boundary). Measurement-dependence of `%c` is **necessary but not
sufficient** for unbounded support: a predicate like `%m AND (counter < N)` is
measurement-conditioned yet statically capped at `N`. Purl therefore additionally
requires that **no compile-time bound dominates** the condition — if a static
counter guard conjoined into `%c` caps the trip count, the loop is treated as
bounded (detected and **warned**, not cut). Absent such a dominating bound, trip
count `k` has unbounded support (`P(k > N) > 0` for all N). (Purl's own bounded
output is protected separately by `purl.applied`.)

Two orthogonal points: *dynamic* is a property of the loop **condition**
(measurement-conditioned ⇒ unbounded k); it says nothing yet about the qubits.
The further **carry-type vs restart-type** split (§3.1) classifies what happens
to the **qubits** inside a dynamic loop. Purl transforms the **carry-type**
subclass of dynamic loops.

A **static loop** — `scf.for` with constant bounds, or an `scf.while` conditioned
only on a compile-time counter (no `quantum.measure` in the dependence) — has a
compile-time-known trip count and is out of scope.

Minimal example of a **dynamic loop** (repeat while the measured coin is 1; the
condition `%fail` is the measurement outcome, so the trip count is runtime-random
and unbounded):

```mlir
func.func @dynamic_loop() -> i1 {
  %true = arith.constant true
  %reg  = quantum.alloc( 1) : !quantum.reg
  %q0   = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit

  %res:2 = scf.while (%fail = %true, %q = %q0)
             : (i1, !quantum.bit) -> (i1, !quantum.bit) {
    // continuation predicate = the loop-carried i1 %fail
    scf.condition(%fail) %fail, %q : i1, !quantum.bit
  } do {
  ^bb0(%f: i1, %qq: !quantum.bit):
    %h      = quantum.custom "Hadamard"() %qq : !quantum.bit
    %m, %qm = quantum.measure %h : i1, !quantum.bit          // runtime measurement
    %qr = scf.if %m -> (!quantum.bit) {                      // reset for next attempt
      %x = quantum.custom "PauliX"() %qm : !quantum.bit
      scf.yield %x  : !quantum.bit
    } else {
      scf.yield %qm : !quantum.bit
    }
    scf.yield %m, %qr : i1, !quantum.bit                     // carry[0] (%fail) := %m
  }
  %rr = quantum.insert %reg[ 0], %res#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %rr : !quantum.reg
  return %res#0 : i1
}
```

The loop satisfies the definition: trace `%c = %fail` in `scf.condition(%fail)` →
`before`-arg 0 → cross the boundary to `scf.yield` operand 0 → `%m` → defined by
`quantum.measure`. (This one measures-then-resets the same wire, so §3.1
classifies it *restart-type*; a wire threading the carry un-measured would be
*carry-type* — the subclass Purl cuts.)

## 2. Non-goals
- Compiled execution of the transformed IR / a noisy Catalyst runtime backend.
- Multi-qubit / logical carries (fault-tolerant code blocks, syndrome extraction).
- Hardware runs; speculative parallelization; the per-job split variant.
- Restart-type loops (detect and decline); programs whose output is not an
  expectation value of the carried qubit (detect and decline).

---

## 3. The Purl MLIR passes (two-phase)

Substrate: C++ passes in Catalyst's `mlir/lib/Quantum/Transforms/`, registered in
`quantum-opt` (and buildable as pass plugins for `@qjit`). Attributes are `purl.*`.

Purl is split into **two passes** so analysis and code-generation never mix:

- **`--purl`** — the analysis + rewrite pass. It runs *all* of 3.1–3.6 (classify,
  depth, window, known-state proof, cost model, fidelity prediction) and **decides
  everything**; at each chosen cut site it emits a single high-level
  **`purl.qcut`** op (3.7) with every decision baked into it. Options: see the
  glossary in 3.0.
- **`--purl-lower-qcut`** — a purely **mechanical** lowering (3.7) that expands
  each `purl.qcut` into its concrete op sequence. It performs **no** analysis
  (no proof, no cost model) and reads only the op's own operands, attributes, and
  `prep` region.

The analyses (3.1–3.3) MUST handle the *actual Catalyst-emitted IR shape*
(tensor-wrapped classical values, `stablehlo`/`tensor.extract` glue,
register-threaded qubits). The rewrite (3.5) fires on carry loops with an
`expval` output.

### 3.0 `--purl` options (glossary)

All options live on `--purl`; `--purl-lower-qcut` takes none (every decision they
influence is baked into the `purl.qcut` op). Four provenance classes:
**hardware** (device properties, from the calibration dataset), **placement** (the
logical→physical wire binding, from mapping/routing — a *selector* into the
hardware data, not a property), **compiler knobs** (genuine tuning), and **profile
inputs** (properties of the *program/observable/run*). Placement and profile inputs
are supplied as options when the pass is driven standalone; in a full pipeline their
source is a wire placement attribute, loop/observable attributes, and the `@qjit`
shot config (see §9).

| option | class | feeds | meaning |
|---|---|---|---|
| `calib` | hardware | 3.2, 3.5, 3.6 | Path to the hardware calibration JSON (§4). Single source for depth-in-seconds, cost model, and fidelity prediction. |
| `carry-qubit` | placement | 3.6 | Physical qubit index the carried wire maps to — a *selector* into `calib`, not a property. Chooses the per-qubit T1/T2/1q-err/readout (and 2q-err from its edges). Passed as an **option because of pass ordering** (see note below), not because it is device data. |
| `f` | compiler | 3.3 | Coherence-budget fraction for the window ceiling `C_max = floor(f·T2/(B+tau))`. |
| `C` | compiler | 3.5 | Override the cut period instead of letting the cost model pick the arg-min over the window. |
| `margin` | compiler | 3.5 | Profitability guard: cut fires iff `predicted·margin < predicted(NONE)`. |
| `depth` | compiler | 3.6 | Report `purl.fidelity_at_depth` at this runtime depth `D`, **in iterations** (`0` disables). |
| `analyze-only` | compiler | all | Emit `purl.*` attributes without rewriting the IR (inspection/audit mode). |
| `p` | profile | 3.3, 3.5 | RUS success probability per iteration → the trip distribution (`E[k]=1/p`, `C_min`, age `sbar(C)`, `E[#cuts]`). Not in the IR (the loop is unbounded-dynamic, §1.1). Sets the *expected payoff* of cutting, hence the NONE-vs-cut decision. |
| `shots` | profile | 3.5 | `S`, the shot budget → the statistical term `sigma0/sqrt(S)` and the KNIT variance `sigma0·sqrt(V(C)/S)`. |
| `sigma0` | profile | 3.5 | Per-shot standard deviation of the observable (≤1 for a Pauli); numerator of the statistical term. |

Note: the cost model minimizes **ensemble-expected** error, which is why the
profile inputs (esp. `p`) are load-bearing — they set the *expected* benefit/cost
averaged over the trip distribution, not the cut period itself. A worst-case
per-trajectory objective would need only `C_max` and could drop `p`.

Note (`carry-qubit` and pass ordering): `--purl` runs **before** qubit placement/
routing, so the logical→physical binding is not present in the IR — hence it
is supplied as an *option*, a representative/target-qubit hint for the §3.6
prediction. This is deliberate: Purl *changes the circuit* (inserts measure+reset+
re-prep on the carried wire — no new ancillas; both refresh and KNIT reuse the
carried wire), so it must run before placement and let a single
downstream mapping pass route the final circuit. Were Purl placed *after* routing
it could read the real physical qubit from a wire attribute (exact prediction), but
its inserted ops would then need re-routing. The chosen order trades an exact qubit
for a clean, single-pass placement; `carry-qubit` is the modeling stand-in for the
not-yet-assigned qubit.

Note (dataset-sourced hardware, not options): not every model input is an option.
The per-iteration classical **feedback latency `tau`** (the readout→controller→
condition→dispatch round-trip the carried wire idles through each dynamic
iteration), the state-prep error `p_prep`, and the per-2q-gate **leakage**
(`leak_2q_default` / per-edge `leak_2q`, §4.1) all arrive via the `calib` file —
`tau` and `p_prep` are **top-level control-system properties** and leakage is
**per-edge calibration data**, so they ride in with `calib`/`carry-qubit` rather
than as their own knobs. (Leakage was formerly a `p-leak` knob; it is now
first-class calibration, single-sourced from the JSON.) `tau` enters `B+tau`
everywhere it matters: body cost (3.2), the window ceiling `C_max` (3.3), and the
idle time `t_idle=D·(B+tau)` (3.6).

### 3.1 Classification (carry / restart / unknown)
Walk each quantum slot of the `scf.while` carry (per-slot for a `!quantum.reg`
via extract/insert indices; or a bare `!quantum.bit`). A slot is:
- **RESET** — measured then outcome-conditioned-`PauliX` corrected (or re-prepared
  from a constant);
- **CARRY** — reaches `scf.yield` through gates with **no** measure on its line;
- **UNKNOWN** — measured but not provably reset.

Loop class: CARRY if ≥1 CARRY slot; RESTART if all RESET; else UNKNOWN. Exactly
one CARRY slot supported; ≥2 → diagnostic "multi-wire cut unsupported". Emit
`purl.class`, `purl.carry_slot`.

### 3.2 Body depth B
Single topological walk assigning a depth (calibrated duration, or unit layers)
to each qubit value; gate = `w(op) + max(operand depths)`; measure adds readout;
`scf.if` takes the per-branch max plus one feedback `tau` if its condition
derives from a measure. `B = max` over yielded qubit depths. Also count 1q/2q
gates and readouts (per body) for the cost model. Nested `scf.while` → error.
Emit `purl.body_seconds` (or `purl.body_layers`).

### 3.3 Cut-period window
`C_min` = the smallest `C` with **`V(C) ≤ V_max`** (variance cap, default
`V_max = 4`), where `V(C) = (1-q)/(1-γ²q)`, `q = (1-p)^C`, `γ² = 16` (§3.5). This
bounds the KNIT sampling overhead, not merely its finiteness: bare finiteness
`γ²q < 1` is the `V_max → ∞` limit and recovers the closed form
`ceil(ln(γ²)/ln(1/(1-p)))`.
`C_max = floor(f·T2/(B+tau))` (`f` default **0.05**) — equivalently a **coherent-depth
cap**: the number of held iterations at which the idle retention `exp(-D·(B+tau)/T2)`
falls to `exp(-f)`, i.e. the depth budget for a fraction `f` of idle coherence
(§3.6). Emit `purl.window=[C_min,C_max]`. The deterministic (refresh) strategy has
**no** variance floor, so its window is `[1, C_max]`.

### 3.4 Known-state proof (enables the cheap cut)
Stabilizer / Pauli-frame tracking of the carried wire over the loop body: carry
its logical operators X_d, Z_d and the fresh ancillas' stabilizers as multi-qubit
Paulis; conjugate through Clifford **and entangling** (CNOT/CZ) gates; at each
ancilla Z-measurement do the Aaronson–Gottesman reduction (bail if a logical
operator is *disturbed* — that is a heralding measurement, not a known state);
apply conditional-Pauli corrections. If the net carried-wire action is a Pauli
with outcome-independent sign → the carried state is a **known** pure state
(`identity` or `pauli`); else `unknown`. Emit `purl.known_state`. Non-Clifford on
the carried failure path or Toffoli/multi-control → `unknown` (safe fallback).

### 3.5 Strategy selection (profitability cost model) + rewrite
Per-iteration error is derived from the **same** fidelity model as §3.6 — one model,
so the §3.5 arg-min and the §3.6 fidelity ranking cannot disagree. From §3.6's
single-iteration retention factors `F1_t` (transportable) and `F1_nt` (leakage):

- `eps_t  = 1 − F1_t`  — **transportable** (depolarizing + T2 idle over `B+tau`);
- `eps_nt = 1 − F1_nt` — **non-transportable** (leakage per 2q gate; `p_leak` from the §4.1 calibration);
- `eps_all = 1 − F1_t·F1_nt ≈ eps_t + eps_nt`;
- `eps_cut = p_ro + p_prep`.

With `q=(1-p)^C`, `E[k]=1/p`, `E[#cuts]=q/(1-q)`, `V(C)=(1-q)/(1-γ²q)` (`γ²=16`),
and the expected delivered **coherent depth**
`sbar(C) = Σ_{j≥1} p(1-p)^{j-1}·(((j-1) mod C) + 1)` (summed to convergence; capped
at `C`), the
predicted expval error combines its bias and statistical terms **in quadrature**,
`predicted = sqrt(bias² + statistical²)`, per strategy:

| strategy | bias | statistical |
|---|---|---|
| NONE | eps_all·E[k] | sigma0/sqrt(S) |
| REFRESH@C (tier-1, gamma=1) | eps_all·sbar(C) + E[#cuts]·eps_cut | sigma0/sqrt(S) |
| KNIT@C (tier-3, gamma=4) | eps_t·E[k] + eps_nt·sbar(C) + E[#cuts]·eps_cut | sigma0·sqrt(V(C)/S) |

Minimize each *applicable* strategy over its C-range (REFRESH `[1,C_max]`; KNIT
`[C_min,C_max]`, `γ²q<1`); pick the arg-min; fire it iff
`predicted·margin < predicted(NONE)`, else leave the loop unchanged. Emit
`purl.strategy` ∈ {none, refresh, knit}, `purl.C`, and the auditable
`purl.predicted = {none, refresh, knit}`.

**No discard arm** — neither in the pass nor as an eval baseline. Under the §11
Markovian analysis truncate+discard is a strong *fidelity* baseline, but it changes
the delivered **computation** (conditioning the estimator on early success biases
the observable), so it is deliberately excluded rather than compared. Refresh
delivers the *same* computation at higher fidelity — that equivalence is the claim
under test, and a discard arm would not be answering the same question.

Both rewrites *insert a `purl.qcut` op* (3.7) rather than expanding a cut
inline — `--purl` builds the carry surgery + guard and bakes the strategy into the
op; `--purl-lower-qcut` produces the actual op sequence.

**REFRESH (gamma=1)** — fires when `known_state ∈ {identity, pauli}`: extend the
carry with an `i32` counter and, every C failing iterations, emit
`purl.qcut {strategy="refresh"}` carrying the known-`|psi0>` `prep` region (and,
for a KnownPauli, the counter-parity `pauli_correction`). No weight, no RNG hook,
zero variance; the **`quantum.expval` output survives** (a refresh delivers a
genuine quantum state — no legalization). This is the only strategy that escapes
the Markovian no-go (it replaces the noisy state with the ideal one).

**KNIT (gamma=4)** — the general quasi-probability cut: extend the carry with an
`i32` counter + `f64` weight and, every C failing iterations, emit
`purl.qcut {strategy="knit", axis=…}` threading the weight; legalize the
`expval` output to a weighted Z sample. Idempotent (`purl.applied`). Its lowering
(3.7) performs the `func.call @purl_sample_term` RNG hook → basis change → measure
→ reset → eigenstate prep → `4·sigma·s` weight fold.

Both REFRESH and KNIT use the **single counter idiom**: the `i32` counter increments
each failing iteration and **resets to 0 at every fired cut** (period `C`). KNIT's
`f64` weight, by contrast, **accumulates across cuts** (it is never reset).

### 3.6 Depth-based fidelity model

Delivered fidelity is a function of the carried wire's **coherent depth `D`** — the
number of iterations it is held un-cut (§1) — and of nothing else. Per held
iteration the wire keeps a fixed fraction `rho` of its fidelity (the **per-iteration
retention**), so a depth-`D` coherent chain delivers

```
rho  = F1_t · F1_nt          // per-iteration retention (one held iteration)
F(D) = rho^D                 // fidelity depends only on coherent depth D
```

The retention is calibrated from the hardware and factored into a **transportable**
and a **leakage** part:

```
F1_t  = exp(-(B+tau)/T2) · (1-e1q)^n1q · (1-e2q)^n2q     // transportable, 1 iteration
F1_nt = (1-p_leak)^n2q                                   // leakage, 1 iteration
```

`rho` bundles *everything the wire loses per held iteration* — idle T1/T2
decoherence over one body (`B+tau`), per-gate depolarizing, and per-2q leakage —
into a single **per-depth constant**. Wall-clock time (via `T2`) is only the
*source* of the idle part of `rho`; the model — and the pass — reason in **coherent
depth**, not in time. (Symbols: `n1q,n2q` = per-iteration 1q/2q gate counts (§3.2);
`e1q = gate_1q_err`, `e2q` = median incident `gate_2q_err`; `p_leak` = global median
of `leak_2q` over the coupling map (`leak_2q_default` fills gaps), sourced from the
§4.1 calibration; readout leakage `p_leak_ro = p_leak·0.5` is derived. Idle decay
uses `exp(-t/T2)` only — `1/T2` already contains
`1/(2·T1)` (why §4.2 enforces `T2 ≤ 2·T1`), so a separate `exp(-t/T1)` would
double-count.)

**Why the pass optimizes for depth.** Since `F(D) = rho^D` is strictly decreasing in
`D`, *maximizing delivered fidelity is exactly minimizing the carried wire's coherent
depth.* The cut bounds that depth at the period `C`: it caps the **delivered** state's
coherent depth (its age since the last cut) at `≤ C`, hence its error at `≤ 1−rho^C`,
**regardless of the runtime trip count `k`**. The §3.3 window and the §3.5 strategy
selection are all expressed in this depth / `rho^D` model — one model, so the pass's
decision and its predicted fidelity cannot disagree (§3.5's `eps_t = 1−F1_t`,
`eps_nt = 1−F1_nt` are the per-depth error).

The retention is evaluated at the **nominal `lam=1`** operating point (§5); the pass
has no `lam` knob, so `purl.predicted_fidelity` is comparable to the simulator only
at `lam=1` — which is where S3 (§8.4) compares.

Emit `purl.predicted_fidelity = {unbounded, bounded}` — the **mean** `rho^D`
delivered fidelity over the geometric trip distribution, where the delivered
coherent depth `D` is the full trip count `k` for `unbounded` and the capped
`((k-1) mod C)+1 ≤ C` for `bounded`/refresh — and, if `depth>0`,
`purl.fidelity_at_depth`.

**Gate depth ↔ coherence time (why one `rho` absorbs both).** The two error sources
have different natural units: **gate error** (depolarizing, leakage) is *count*-based
— `(1-e)^{#gates}`, geometric in the number of gates — while **coherence loss**
(T1/T2) is *time*-based — `exp(-t/T2)`, exponential in elapsed time. They meet on a
single axis because every held iteration is identical: it has a fixed **gate count**
(`n1q, n2q`) **and** a fixed **duration** (`B+tau`), so time is proportional to depth,
`t = D·(B+tau)`. Evaluating each source over one iteration and multiplying gives the
per-depth retention

```
rho =  exp(-(B+tau)/T2)             // coherence loss over one body's DURATION
     · (1-e1q)^n1q · (1-e2q)^n2q     // gate error over one body's gate COUNT
     · (1-p_leak)^n2q                // leakage over one body's 2q gates
```

so `F(D) = rho^D` uses the coherence model (T2, via the idle factor) **and** the gate
model (`e1q, e2q, p_leak`, via the count factors) at once. Bounding the coherent depth
`D` therefore bounds **both** the accumulated idle time `D·(B+tau)` (T2 loss) and the
accumulated gate count `D·n` (gate loss) — one knob, both error sources. Which term
dominates is set by `n1q,n2q` vs `(B+tau)/T2`: a *held-idle* wire (`n ≈ 0`) is
**coherence-dominated** (`rho ≈ exp(-(B+tau)/T2)`); a *gate-heavy* wire is
**gate-count-dominated**. Consistently, the §3.3 window takes its ceiling `C_max` from
the time source (the coherence budget) and its floor `C_min` from the count source
(the sampling variance).

### 3.7 The `purl.qcut` op and its mechanical lowering

`--purl` never expands a cut inline. At each guarded cut site it emits one op from
a small **Purl-owned dialect** (`purl.qcut`) that names the cut abstractly. (The op
lives in its own dialect rather than the core `quantum` dialect so the transform is
self-contained; it reuses quantum-dialect value types — `!quantum.bit` — and the
`quantum` observable attribute for its `axis`.)

```mlir
// KNIT: threads the accumulated quasi-probability weight
%q', %w' = purl.qcut %q, %w_in
             { strategy = "knit", axis = #quantum<named_observable PauliZ> }
             prep { ^bb0(%fresh: !quantum.bit):     // eigenstate prep is axis-driven
                    quantum.yield %fresh : !quantum.bit }
           : (!quantum.bit, f64) -> (!quantum.bit, f64)

// REFRESH: no weight; prep region reproduces the loop's input |psi0>
%q' = purl.qcut %q
        { strategy = "refresh", pauli_correction = #purl<pauli none> }
        prep { ^bb0(%fresh: !quantum.bit): /* input prep */ 
               quantum.yield %prepared : !quantum.bit }
      : (!quantum.bit) -> !quantum.bit
```

The op is deliberately **self-contained** — because the lowering does no analysis,
`--purl` bakes every decision into it: the chosen `strategy`, the observable
`axis`, the KnownPauli `pauli_correction`, and the known-state preparation as a
captured `prep` region (the loop's input prep of `|psi0>` from a fresh `|0>`). All
the `strategy`/`pauli_correction` attributes are **purl-dialect** attributes while
`axis` reuses the `quantum` observable attribute (`#quantum<named_observable …>`);
the op carries `!quantum.bit` operands/results. The
counter/weight carry surgery and the `if counter == C { qcut }` guard are also
built by `--purl` (which alone knows `strategy` and `C`); the op sits inside that
guard. Verifier: `refresh` takes/returns one qubit and no weight; `knit` takes/
returns a qubit and an `f64` weight.

**`--purl-lower-qcut`** is a single mechanical rewrite that switches on `strategy`
and needs nothing beyond the op itself:
- `refresh` → `quantum.measure` + reset + inline the `prep` region (+ apply
  `pauli_correction` before the observable). gamma=1, no RNG, no weight.
- `knit` → `func.call @purl_sample_term` (RNG hook) → basis change to `axis` →
  `measure` → reset → eigenstate prep → fold `4·sigma·s` into `%w'`. gamma=4.

Running `--purl-lower-qcut` immediately after `--purl` reproduces the same lowered
program the monolithic inline rewrite would have emitted: the two-phase split is an
internal refactor with no change to the final IR, while giving a stable,
inspectable `purl.qcut` level in between (useful for `--analyze-only` dumps and
for testing insertion and expansion independently, §7).

**Worked example (REFRESH).** Simplified IR — real Catalyst IR wraps classical
values in tensors + `stablehlo`/`tensor.extract`; elided here.

*(a) Input — before `--purl`.* A carry loop that holds `|psi0>` and repeats until
an ancilla heralds success; the carried wire takes gates but is never measured.

```mlir
func.func @rus() -> f64 {
  %q0 = ...                                   // prepare |psi0> on the carried wire
  %qf = scf.while (%q = %q0) : (!quantum.bit) -> !quantum.bit {
  ^before(%q: !quantum.bit):
    %q1, %m = ...                             // RUS body: entangle %q w/ fresh
                                              // ancilla, measure ANCILLA -> %m
    %fail = arith.cmpi eq, %m, %false : i1    // continue while not heralded
    scf.condition(%fail) %q1 : !quantum.bit
  } do {
  ^after(%q: !quantum.bit):
    scf.yield %q : !quantum.bit
  }
  %obs = quantum.namedobs %qf[PauliZ] : !quantum.obs
  %e   = quantum.expval %obs : f64
  return %e : f64
}
```

*(b) After `--purl` (= input to `--purl-lower-qcut`).* An `i32` counter is threaded
through the carry; the failing branch gets an `if counter==C` guard holding one
self-contained `purl.qcut`. Analyses land as `purl.*` attributes.

```mlir
func.func @rus() -> f64
    attributes { purl.class = "carry", purl.known_state = "identity",
                 purl.strategy = "refresh", purl.C = 8 : i32,
                 purl.window = [1, 12], purl.applied } {
  %q0 = ...                                   // prepare |psi0>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cC = arith.constant 8 : i32
  %qf, %cf = scf.while (%q = %q0, %ctr = %c0)
      : (!quantum.bit, i32) -> (!quantum.bit, i32) {
  ^before(%q: !quantum.bit, %ctr: i32):
    %q1, %m = ...                             // same RUS body
    %fail = arith.cmpi eq, %m, %false : i1
    scf.condition(%fail) %q1, %ctr : !quantum.bit, i32
  } do {
  ^after(%q: !quantum.bit, %ctr: i32):        // runs only on a FAILING iteration
    %ctr1 = arith.addi %ctr, %c1 : i32
    %hit  = arith.cmpi eq, %ctr1, %cC : i32
    %qn, %ctrn = scf.if %hit -> (!quantum.bit, i32) {
      %qr = purl.qcut %q { strategy = "refresh" }
              prep { ^bb0(%fresh: !quantum.bit):    // reproduces |psi0>
                     %p = ... ; quantum.yield %p : !quantum.bit }
            : (!quantum.bit) -> !quantum.bit
      scf.yield %qr, %c0 : !quantum.bit, i32        // refreshed + counter reset
    } else {
      scf.yield %q, %ctr1 : !quantum.bit, i32
    }
    scf.yield %qn, %ctrn : !quantum.bit, i32
  }
  %obs = quantum.namedobs %qf[PauliZ] : !quantum.obs
  %e   = quantum.expval %obs : f64              // expval SURVIVES (refresh, no legalize)
  return %e : f64
}
```

*(c) After `--purl-lower-qcut`.* The `purl.qcut` is mechanically expanded to
measure + reset + inline the `prep` region (γ=1; the measured value is discarded —
the noisy state is thrown away and the ideal `|psi0>` re-prepared). Only the guarded
`scf.if` changes; everything else is byte-identical to (b).

```mlir
    %qn, %ctrn = scf.if %hit -> (!quantum.bit, i32) {
      // purl.qcut{refresh} expands to: measure (discard) -> reset -> inline prep
      %m2, %qm = quantum.measure %q : i1, !quantum.bit
      %qz = scf.if %m2 -> (!quantum.bit) {          // conditional-X reset -> |0>
        %x = quantum.custom "PauliX"() %qm : !quantum.bit
        scf.yield %x : !quantum.bit
      } else {
        scf.yield %qm : !quantum.bit
      }
      %qr = ...                                     // inlined prep region -> |psi0>
      scf.yield %qr, %c0 : !quantum.bit, i32
    } else {
      scf.yield %q, %ctr1 : !quantum.bit, i32
    }
```

(The KNIT lowering instead threads an `f64` weight and expands to `@purl_sample_term`
→ basis change → measure → reset → eigenstate prep → `4·sigma·s` weight fold, and
the `expval` in (b) is legalized to a weighted-Z sample.)

---

## 4. The shared IBM hardware dataset (one JSON, two consumers)

`ibm_eagle_r3.json` — a 127-qubit **IBM Eagle r3** dataset with representative
published medians and a deterministic per-qubit spread. The **same file** feeds
(a) the pass's depth-in-seconds, cost model, and fidelity prediction, and (b) the
simulator's noise model (§5). A generator writes the JSON deterministically; a
loader validates it and extracts a flat per-carried-qubit calibration.

### 4.1 Schema

All times are **SI seconds**; all error/probability fields are dimensionless in
`[0,1]`. Unknown extra keys are ignored (forward-compatible).

| field | where | type | unit | meaning |
|---|---|---|---|---|
| `backend` | top | string | — | device label, e.g. `"ibm_eagle_r3"` |
| `n_qubits` | top | int ≥ 1 | — | must equal `len(qubits)` |
| `units` | top | string | — | must be `"SI"` (guards against µs/ns mixups) |
| `gate_1q_time` | top | float > 0 | s | 1q (sx) gate duration |
| `gate_2q_time` | top | float > 0 | s | 2q (ECR) gate duration |
| `readout_time` | top | float > 0 | s | mid-circuit measurement duration |
| `tau` | top | float ≥ 0 | s | classical feedback latency (§3.0 note) |
| `p_prep` | top | float ∈ [0,1] | — | reset/state-prep error |
| `qubits[i].T1` | per-qubit | float > 0 | s | amplitude-damping time |
| `qubits[i].T2` | per-qubit | float > 0 | s | dephasing time (**≤ 2·T1**) |
| `qubits[i].gate_1q_err` | per-qubit | float ∈ [0,1] | — | 1q depolarizing error |
| `qubits[i].readout_err` | per-qubit | float ∈ [0,1] | — | readout bit-flip prob |
| `leak_2q_default` | top | float ∈ [0,1] | — | **required** per-2q-gate leakage default (fills edges without an explicit value) |
| `leak_source` | top | string | — | *optional* provenance note for the leakage numbers (written, not validated) |
| `edges[k].q` | per-edge | `[int,int]` | — | undirected pair, `0 ≤ i<j < n_qubits` |
| `edges[k].gate_2q_err` | per-edge | float ∈ [0,1] | — | 2q depolarizing error |
| `edges[k].leak_2q` | per-edge | float ∈ [0,1] | — | *optional* per-edge leakage override (else `leak_2q_default`) |

**Leakage is calibration data** — it lives in the JSON like any other rate, not in
a CLI knob. `leak_2q_default` is **required** (a device without a measured leakage
number still states its assumed value explicitly); optional per-edge `leak_2q`
overrides refine it. IBM does not publish leakage, so `leak_source` records where
the number came from (an estimate, not vendor calibration). This single-sources
leakage: the pass and the simulator both read it from the same file, so they cannot
disagree. (The former `p-leak` / `--leak` knob is removed.)

**Leakage is charged per 2q gate** — one mechanism, stated once here and referenced
everywhere: the pass (§3.6, `F1_nt = (1-p_leak)^n2q`), the simulator (§5), and the
loader all apply `p_leak` **per 2q gate**, not per idle-second. A per-second charge
on one side and per-gate on the other would break S3 (predicted↔measured) by
construction, so the per-2q-gate convention is normative. The flattened per-qubit
`p_leak` = **global median** of `leak_2q` over the coupling map (mirroring
`gate_2q_err`), and readout leakage `p_leak_ro = p_leak·0.5` is derived.

### 4.2 Validation (loader-enforced)

The loader rejects a file (hard error) unless:

1. `n_qubits == len(qubits)` and `n_qubits ≥ 1`; `units == "SI"`.
2. every top-level time is `> 0`; `tau ≥ 0`; `p_prep ∈ [0,1]`.
3. every per-qubit `T1,T2 > 0` and **`T2 ≤ 2·T1`** (physical; the pure-dephasing
   rate `1/T2 = 1/(2 T1) + 1/T_phi` must be non-negative); `gate_1q_err`,
   `readout_err ∈ [0,1]`.
4. every edge `q=[i,j]` has `0 ≤ i < j < n_qubits`; no duplicate undirected pair;
   `gate_2q_err ∈ [0,1]`.
5. `leak_2q_default` is present and in `[0,1]`; every per-edge `leak_2q` (where
   given) is in `[0,1]` (see §4.1).

Non-fatal **warnings**: a disconnected coupling map; a qubit with no incident edge
(its 2q error then falls back to the global median `gate_2q_err`); any per-qubit
value deviating > 10× from the stated medians (typo guard); any per-edge `leak_2q`
deviating > 10× from `leak_2q_default` (typo guard).

The `--purl` `carry-qubit` index is validated against this file: it must satisfy
`0 ≤ carry-qubit < n_qubits`, else the pass errors. **Loader → flat calib:** for
the chosen qubit `k` it emits `{T1,T2,gate_1q_err,readout_err}` from `qubits[k]`,
`gate_2q_err` = **global median** of `gate_2q_err` over the coupling map, `p_leak` =
**global median** of `leak_2q` (`leak_2q_default` filling gaps), `p_leak_ro =
p_leak·0.5`, plus the top-level times/`tau`/`p_prep`. Both medians are global
(device-representative), not restricted to edges incident to `k`. That flat dict is
exactly what `Calib::load` and the simulator consume.

### 4.3 Complete valid example

Representative medians: T1 ~ 250 µs, T2 ~ 150 µs, sx 1q-err ~ 2.5e-4 (32 ns),
ECR 2q-err ~ 8e-3 (560 ns), readout-err ~ 1.3e-2 (1.2 µs), tau ~ 1 µs. A complete,
schema-valid instance (5 qubits on a path; the shipped file scales this to
`n_qubits = 127` with a heavy-hex `edges` map, generated deterministically):

```json
{
  "backend": "ibm_eagle_r3",
  "n_qubits": 5,
  "units": "SI",
  "gate_1q_time": 3.2e-8,
  "gate_2q_time": 5.6e-7,
  "readout_time": 1.2e-6,
  "tau": 1.0e-6,
  "p_prep": 1.0e-3,
  "leak_2q_default": 1.0e-3,
  "leak_source": "estimate, not vendor calibration (IBM does not publish leakage)",
  "qubits": [
    { "T1": 2.51e-4, "T2": 1.48e-4, "gate_1q_err": 2.4e-4, "readout_err": 1.30e-2 },
    { "T1": 2.33e-4, "T2": 1.61e-4, "gate_1q_err": 2.6e-4, "readout_err": 1.42e-2 },
    { "T1": 2.68e-4, "T2": 1.39e-4, "gate_1q_err": 2.5e-4, "readout_err": 1.11e-2 },
    { "T1": 2.19e-4, "T2": 1.52e-4, "gate_1q_err": 2.7e-4, "readout_err": 1.55e-2 },
    { "T1": 2.60e-4, "T2": 1.44e-4, "gate_1q_err": 2.3e-4, "readout_err": 1.20e-2 }
  ],
  "edges": [
    { "q": [0, 1], "gate_2q_err": 7.8e-3 },
    { "q": [1, 2], "gate_2q_err": 8.4e-3, "leak_2q": 1.2e-3 },
    { "q": [2, 3], "gate_2q_err": 9.1e-3 },
    { "q": [3, 4], "gate_2q_err": 7.2e-3 }
  ]
}
```

(Every qubit above satisfies `T2 ≤ 2·T1`; every edge indexes a valid pair;
`leak_2q_default` is present and one edge carries a per-edge `leak_2q` override —
the file passes §4.2.)

---

## 5. The noise-modelling simulator (reads the same JSON)

A pure-NumPy Monte-Carlo **trajectory simulator** (statevector per shot; noise via
stochastic channel sampling), independent of Catalyst at runtime.

- **Op set:** alloc, h, x, z, s, sdg, t, cnot, toffoli, measure (Born + collapse),
  reset, force_zero (fresh qubit).
- **Noise (parameterised by the §4 JSON + a global scale `lam`∈[0,4]):**
  per-1q/2q depolarizing; readout flip + pre-measure depolarizing; **idle
  amplitude-damping + pure-dephasing** over the per-iteration idle time (from
  T1/T2); and **leakage** (population leaving the computational subspace, charged
  **per 2q gate** at rate `lam·p_leak` — the same mechanism as §3.6/§4.1, *not* a
  per-idle-second charge; a reset / cut clears it — the reset-clearable error the
  transform reduces). `lam=0` is exactly noiseless.
- **Executors** mirroring the pass's arms: `unbounded`, `refresh` (measure +
  force_zero + re-prep known |psi0>), `knit` (inline quasi-probability cut with
  weights). Delivered-state fidelity is measured by **3-basis tomography**
  (estimate ⟨X⟩,⟨Y⟩,⟨Z⟩ over S/3 shots each; F = ½(1 + a·n_ideal)). For the **knit**
  arm `a` is a *reconstructed*, weight-summed estimate that can be non-physical
  (`|a| > 1`) at finite shots, so `|a|` is **clamped to ≤ 1** before computing F,
  and this arm's column is labelled **"reconstructed"** in the §8.2 legend.
- **Validation gates:** at `lam=0` the primary benchmark reproduces its noiseless
  ⟨Z⟩; an idle |+⟩ over T2 shows ⟨X⟩ ≈ e^-1.

### 5.1 Calibration wiring and noise model

The simulator is structured as a three-stage pipeline from the §4 dataset to the
trajectory core:

- **Generator** (`ibm_dataset.build_json`) — writes `ibm_eagle_r3.json` in the §4
  schema (per-qubit `T1/T2/gate_1q_err/readout_err`, per-edge `gate_2q_err` and
  optional `leak_2q`, top-level times/`tau`/`p_prep`/`leak_2q_default`). A leakage
  sweep is done by generating **variant JSONs** (`eval/variants.py`), not a knob.
- **Loader** (`ibm_dataset.carried_calib(qubit)`) — flattens the per-qubit +
  coupling JSON for the carried qubit onto the trajectory core's calib keys:
  `gate_1q_err→p1`, `median gate_2q_err→p2`, `readout_err→p_ro`,
  `gate_1q_err→p_meas`, `T1/T2`, durations, `tau`, `p_prep`, and `p_leak` = global
  median of `leak_2q` (from the JSON, §4.1) — no `p_leak` argument.
- **Trajectory core** (`qsim.QSim(n, calib, lam)`) — draws those rates into the
  noise model: T1/T2 idle (amplitude damping `1/T1` + pure dephasing `Tφ` from
  `1/T2 = 1/(2T1)+1/Tφ`), per-gate depolarizing (`p1/p2`), readout flip (`p_ro`),
  pre-measure depolarizing (`p_meas`), spectator-qubit idle during every
  gate/readout.

The IBM path is selected by passing `calib=carried_calib(<carry-qubit>)` (the §8.3
`--ibm` default); a flat `_DEFAULT_CALIB` (`calib="unit"`) path serves unit checks.

Two model rules are **normative**:

1. **Leakage is charged per 2q gate** (§4.1). On each 2q gate (`cnot`, `toffoli`) the
   target is marked `leaked` with probability `lam·p_leak` — the §3.6
   `F1_nt=(1-p_leak)^n2q` mechanism. There is **no** per-idle-second leakage term
   (no `T_leak` time-constant). A `reset`/cut-reprep clears `leaked`. `lam=0` is
   exactly noiseless, and the `p_leak=0` case must reproduce the noiseless numbers (a
   `validate.py` gate).
2. **KNIT delivered fidelity** clamps the reconstructed Bloch vector to `|a| ≤ 1`
   before `F = ½(1 + a·n_ideal)` — its `a` is a weight-summed estimate that can be
   non-physical at finite shots — and the arm is labelled **"reconstructed"** in the
   §8.2 legend.

The flat `_DEFAULT_CALIB` / `calib="unit"` path is reserved for unit-style checks;
the `--ibm` dataset path is the default for the study.

---

## 6. Benchmarks (PennyLane/Catalyst → MLIR)

Each benchmark exists as (a) a **PennyLane/Catalyst `@qjit` program** using
`catalyst.while_loop` + mid-circuit `measure`, which **lowers to MLIR** (captured
via `keep_intermediate` / `catalyst-cli`) and is the input the Purl pass is
applied to; and (b) a **Python mirror** driving the §5 simulator, kept in lockstep
for the fidelity study. All are carry-type, hold the non-Clifford magic
state `|psi0> = H T H T H |0>`, and return `qml.expval(PauliZ(target))`.

| Name | p | coin gate set | Role |
|---|---|---|---|
| `rus_rx_ibm` | 5/8 | Toffoli-sandwich (multi-control, **non-Clifford**) | primary; IBM-tutorial RUS shape (2 controls + 1 held target) |
| `rus_chain(N)` | 5/8 / stage | Toffoli per stage | N sequential RUS gates on one held data qubit (N ∈ {1,2,4,8}) |
| `rus_lowp` | 0.1 | **CNOT-heralded** (Clifford + CNOT ancilla; identity on the target on failure) | low-p heavy-tail regime (quantum-repeater / heralded memory); mean trip count 10 — where cutting is meant to help |
| `pump` | 0.1 | **CNOT-sandwich ×3** (Clifford; identity on the held data each iteration; **6 two-qubit gates/iter**) | entanglement-pumping proxy (§6.1); leakage-heavy carry — the primary consumer of the per-2q leakage model (§4.1) |
| `ipe_project` | ~0.12 / ~0.45 | **controlled-Rz(θ)** (non-Clifford; each round partially projects the carried superposition — the carried state is **outcome-dependent**) | phase-estimation-as-projection (§6.3); the **knit-only** benchmark (refresh is *unsound in principle*), with a refresh falsification arm |

Most hold the same magic-state input and identical carried-qubit *physics*; they
differ in the trip distribution (`p`), stage count (`N`), the 2q-gate load per
iteration, and — deliberately — the
**coin gate set**, which decides provability (§3.4). `rus_lowp` uses a CNOT-based
heralding coin whose net action on the held target is provably **identity** on
failure, so REFRESH fires; the Toffoli-based coins are non-Clifford multi-control,
which §3.4 cannot prove, so they fall back to KNIT/NONE. Expected pass outcomes:

| benchmark | purl.class | purl.known_state | expected purl.strategy |
|---|---|---|---|
| `rus_rx_ibm` | carry | unknown | **none** (thin p=5/8 tail, §11; knit only if its window is non-empty and profitable) |
| `rus_chain(N)` | carry | unknown | none / knit per stage |
| `rus_lowp` | carry | identity | **refresh** — the headline positive result (§11, S2) |
| `pump` | carry | identity | **refresh** (window `[1,2]`; leakage-clearing gain, §6.1) |
| `ipe_project` (faithful) | carry | unknown | **none** (knit window empty, `C_min≈24 > C_max`; §6.3) |
| `ipe_project_fast` | carry | unknown | **knit** — the knit-arm positive result (non-empty window at p≈0.45, B≈4, f=0.15; §6.3) |

### 6.1 `pump` — entanglement-pumping proxy

**What it models.** Entanglement pumping repeatedly attempts a purification round
against one held quantum resource until a herald succeeds, so the held resource
ages through a *geometric* number of two-qubit-gate-heavy rounds — exactly Purl's
carry shape. Two properties the other benchmarks lack: (a) each iteration charges
**many** 2q gates to the held wire, so **leakage** (charged per 2q gate, §4.1)
becomes a first-order term rather than a sliver — `pump` is the primary consumer of
the per-gate leakage model; and (b) the protocol family is *canonical* (a named,
citable purification/pumping lineage, §6.2) rather than constructed. The held wire
is a **single-qubit proxy** for the true two-qubit resource — the same
simplification `rus` already uses, and it must be described as such in the paper.

**Definition.** Two lockstep artifacts (a `@qjit` program lowered to MLIR + a Python
mirror). *Held wire `d`*: prepared once, before the loop, in
`|psi0> = H T H T H |0>` (identical to `rus`, ideal `<Z>=0.5`); never measured in
the loop. *Ancilla `a`*: reset and reused every iteration (fixed register, no
per-iteration allocation). Body, per iteration:
1. **Coin.** Reset `a`, then `Ry(theta)` on `a`, with `theta` chosen so the success
   probability is exactly `p = 0.1` (mean 10 trips, matching `rus_lowp` for
   comparability).
2. **Entangling block ×3** (6 two-qubit gates/iteration): `CNOT(a→d)`, `S(a)`,
   `CNOT(a→d)`. The sandwich decouples exactly — on the `a=|0>` branch the data wire
   is untouched, and on the `a=|1>` branch the second CNOT undoes the first kick — so
   the net action on `d` is the **identity** and the measurement outcome is provably
   independent of the data state. All gates are Clifford, so §3.4 certifies identity.
3. **Measure `a`.** Success exits the loop, failure repeats.

Output `qml.expval(PauliZ(d))`. Target body depth `B ≈ 12` gate layers (the 6 2q
gates dominate); **report the exact computed `B` in the results, do not force it**.

**Expected pass behavior (assert, do not assume).** `purl.class = carry` (single
carry slot); `purl.known_state = identity` — the Clifford sandwich is the point, so
an `unknown` result means the analysis has a bug or the benchmark deviated from the
definition above; **refresh** with window `[1, C_max]`, and with `B≈12` expect
`C_max = C = 2` as in `rus_lowp`; KNIT stays inadmissible at `p=0.1` (variance floor
near 29).

**Evaluation.** Standard configuration (§8.3: 6000 shots, 8 seeds,
`lam∈{0,0.25,0.5,1,2,4}`), arms **unbounded** and **refresh**, delivered fidelity by
3-basis tomography against `|psi0>`, plus the RMSE/depth columns as for the other
benchmarks. Two runs beyond the standard sweep:
1. **Zero-leakage ablation** — an identical run with leakage set to zero (variant
   JSON, §4.1). The gap between the refresh gains of the two runs is the **measured
   leakage-clearing component**, reported explicitly.
2. **Leakage sweep** — leakage ∈ {1e-4, 1e-3, 1e-2} via variant calibration files
   (`eval/variants.py`, §5.1), reporting refresh gain and predicted↔measured
   agreement at each point.

**Acceptance.** (1) The expected-behavior assertions hold on the **real lowered IR**.
(2) Refresh gain at nominal noise **exceeds the `rus` gain** (the leakage-heavy coin
should decohere the unbounded arm faster) — report the number, whatever it is. (3)
The ablation cleanly separates age-capping from leakage-clearing, and
predicted↔measured stays within the **S3 0.02 criterion** at nominal noise across the
leakage sweep.

**Out of scope.** Two-qubit held states (true pairs), seepage, per-edge leakage
placement studies, and any change to the coin's success-probability model.

### 6.2 `pump` citations (verify before use)

For the paper's benchmark description. Best-effort; **verify every entry against the
published record before adding to the bibliography** — author lists and page numbers
must be checked, and DOIs omitted rather than guessed.

- Bennett, Brassard, Popescu, Schumacher, Smolin, Wootters. *Purification of Noisy
  Entanglement and Faithful Teleportation via Noisy Channels.* Phys. Rev. Lett.
  **76**, 722 (1996). — origin of entanglement purification.
- Deutsch, Ekert, Jozsa, Macchiavello, Popescu, Sanpera. *Quantum Privacy
  Amplification and the Security of Quantum Cryptography over Noisy Channels.* Phys.
  Rev. Lett. **77**, 2818 (1996). — the standard two-way purification protocol.
- Dür, Briegel, Cirac, Zoller. *Quantum repeaters based on entanglement
  purification.* Phys. Rev. A **59**, 169 (1999). — entanglement *pumping*
  specifically: the repeated-attempt loop against one held pair this benchmark
  proxies.
- Briegel, Dür, Cirac, Zoller. *Quantum Repeaters: The Role of Imperfect Local
  Operations in Quantum Communication.* Phys. Rev. Lett. **81**, 5932 (1998). —
  already in the paper's bibliography as `briegel1998repeaters`; **reuse the key**.
- Kalb, Reiserer, Humphreys, et al. *Entanglement distillation between solid-state
  quantum network nodes.* Science **356**, 928 (2017). — experimental demonstration
  that held-resource purification loops are real practice, not theory alone.

### 6.3 `ipe_project` — phase estimation as eigenstate projection

**What it models & why (spec role).** `ipe_project` is the benchmark where
**refresh is unsound in principle and knit is the only valid cut** — the complement
of the identity/refresh benchmarks. The existing `ipe` holds a *known* eigenstate
(provable → refresh fires). Here the carried wire starts in a **superposition of
eigenstates**, and each round's ancilla measurement partially **projects** it toward
one eigenstate, so the carried state at any iteration boundary depends on the
**outcome history**. There is no fixed state to re-prepare, and re-preparing anything
would erase the projection progress that *is* the computation. This gives the paper
three things: a **knit** result on a real, citable protocol (not a synthetic loop);
a demonstration that the known-state analysis returning `unknown` is sometimes the
*physically correct* answer, not a limitation; and a **falsification arm** showing
refresh actively destroying a computation it is not entitled to touch. It replaces
the dropped Toffoli negative control with a physically motivated one. It reuses the
`ipe` program shape, mirror, and adaptive stop (§6, ipe row).

**Definition.** Carried wire `d` prepared in `cos(α)|+> + sin(α)|->` with `α = π/8`
(well away from an eigenstate), where `|+>,|->` are the eigenstates of
`U = Rz(θ)` for a fixed non-Clifford `θ = 2π/7` (no special structure); `d` is never
measured in the loop. Body per round (the `ipe` structure): reset ancilla, Hadamard,
**controlled-`U^(2^j)`** on the exponent schedule, Hadamard, measure the ancilla,
update the classical posterior, and continue until the posterior is confident. The
controlled rotation is **non-Clifford**, so §3.4 must return `unknown` (assert it).
Output: the expectation of the eigenstate observable on `d` (per-shot reference,
below).

**Two configurations, both reported.**
- **`ipe_project` (faithful)** — stop probability near the existing `ipe` value
  (`p≈0.12`). Expected: `class=carry`, `known_state=unknown`, knit window **empty**
  (`C_min≈24 > C_max`), `strategy=none`. The paper's negative control.
- **`ipe_project_fast` (knit-live)** — a coarser confidence threshold giving
  `p≈0.45`, body trimmed toward `B≈4` layers, coherence-budget fraction `f=0.15`
  (a legitimate compiler knob, reported as such). **Build-time check:** verify
  `C_min ≤ C_max` on the deployed calibration; raise `f` before touching `p`.
  Expected: `strategy=knit` at the cost model's arg-min period (§3.5). Also run with
  the **elevated-leakage variant** (`leak_2q_default = 1e-2`, §4.1) so the
  knit-cleared leakage is measurable.

**Expected pass behavior (assert, do not assume).** `class=carry`,
`known_state=unknown` in **both** configs. Faithful: knit inadmissible, loop left
unchanged, decision attributes still emitted for audit. Fast: `strategy=knit`, the
expval legalized to the weighted sample, the quasi-probability weight threaded
through the carry, counter reset + weight accumulation per the existing knit rewrite.

**Delivered-fidelity metric (the subtle point).** The ideal delivered state is
**trajectory-dependent**, so the tomography reference is chosen **per shot**: the
eigenstate identified by that shot's own classical record (the posterior's winner at
exit). At `lam=0` this reference is exact and delivered fidelity must be **1.0**
within statistics (validation gate). Conditioning on the record is legitimate (the
record is part of each shot's data); for the **knit** arm the quasi-probability
weight multiplies the *conditioned* estimate — keep the weight attached to the shot
through the conditioning, and clamp the reconstructed Bloch vector as the existing
knit tomography does (§5.1). This estimator must be documented in code comments; it
is the part a reviewer will check.

**Evaluation.** Standard sweep (§8.3: 6000 shots, 8 seeds, the usual `lam` grid) on
both configs, arms **unbounded** and **knit** (refresh is not applicable). Report the
usual fidelity/RMSE/depth columns, plus predicted↔measured for the fast config at
nominal noise. **Falsification arm** (run once at `lam=1` on the faithful config):
FORCE a refresh-style cut that re-prepares the initial superposition every `C`
iterations. Expected: delivered fidelity against the per-shot reference **collapses**
toward the input state's overlap with the winning eigenstate — concretely showing
refresh on an `unknown` carried state changes the computation. Label it clearly as a
**deliberately invalid** transformation run for illustration.

**Acceptance.** (1) The expected-behavior assertions hold on the **real lowered IR**
for both configs. (2) Fast config: knit fires, and with the elevated-leakage variant
its measured gain over unbounded is **positive with non-overlapping seed error
bars**, while near-zero leakage makes the gain statistically indistinguishable from
zero (the no-go, observed). (3) predicted↔measured within the **S3 0.02 criterion**
at nominal noise for the fast config's selected strategy. (4) The falsification arm
reproduces the expected fidelity collapse.

**Out of scope.** Multi-qubit `U`, mid-loop posterior resets, seepage, any change to
the knit lowering or the `γ=4` decomposition.

### 6.4 `ipe_project` citations (verify before use)

Best-effort; **verify every entry against the published record before adding to the
bibliography**, and omit DOIs rather than guess.

- Abrams, Lloyd. *Quantum Algorithm Providing Exponential Speed Increase for Finding
  Eigenvalues and Eigenvectors.* Phys. Rev. Lett. **83**, 5162 (1999). — phase
  estimation as projection of an approximate input onto exact eigenstates (the exact
  structure this benchmark runs).
- Kitaev. *Quantum measurements and the Abelian Stabilizer Problem.*
  arXiv:quant-ph/9511026 (1995). — origin of phase estimation.
- Dobšíček, Johansson, Shumeiko, Wendin. *Arbitrary accuracy iterative quantum phase
  estimation algorithm using a single ancilla qubit.* Phys. Rev. A **76**, 030306
  (2007). — the single-ancilla iterative mechanics the loop body uses.
- Wiebe, Granade. *Efficient Bayesian Phase Estimation.* Phys. Rev. Lett. **117**,
  010503 (2016). — adaptive, posterior-conditioned stopping (the loop's termination
  rule).
- O'Brien, Tarasinski, Terhal. *Quantum phase estimation of multiple eigenvalues for
  small-scale (noisy) experiments.* New J. Phys. **21**, 023022 (2019). — IPE under
  realistic noise (practice anchor).

---

## 7. Unit tests (FileCheck / lit)

Under `mlir/test/Quantum/Purl/`, run by lit / `check-dialects`:
- `classify_{carry,restart,unknown}`, `depth_unit`, `window_empty`,
  `nested_loop`, `two_carry` — analyses + diagnostics.
- `rewrite_knit`, `idempotence` — the quasi (gamma=4) rewrite.
- `cheap_cut` (held-memory identity → refresh, expval intact), `entangling_cut`
  (entangling RUS coin proven identity), `knownpauli_cut`,
  `known_state_unprovable` (→ knit fallback) — the proof + refresh rewrite.
- `tier1_detect` / `profit_none` / `tier3_unknown` — the 3.5 strategy selection.
- `ibm_fidelity` — the per-qubit dataset drives `purl.predicted_fidelity`.
- `pump_refresh` (§6.1) — the 3-block CNOT-sandwich body classifies **carry**,
  proves **identity**, and receives the **refresh** rewrite.
- `ipe_project_unknown` (§6.3) — the controlled-`Rz` body classifies **carry** and
  returns **`unknown`**; **no refresh rewrite** occurs.
- `ipe_project_knit` (§6.3) — the fast-config shape receives the **knit** rewrite
  (guard, weight carry, `purl.qcut` with axis, expval legalization).

Each asserts the relevant `purl.*` attributes and, for rewrites, the transformed
structure (carry extension, guard, cut expansion / reset+re-prep, output).

Simulator/model gates for `ipe_project` (§5 validation harness, not FileCheck): at
`lam=0` the projection fidelity against the **per-shot reference** (the posterior
winner's eigenstate) is **1.0** within statistics and the posterior winner matches
the collapsed eigenstate on every noiseless shot; a window-math unit test pins
`C_min`/`C_max` for both configs on the deployed calibration (empty for faithful,
non-empty for fast).

Simulator/model gates for `pump` (in the §5 validation harness, not FileCheck): at
`lam=0` the mirror reproduces `<Z>=0.5` exactly; the pass's per-iteration `n2q = 6`
matches the mirror's leakage charging (a one-iteration parity check); fixed seeds
reproduce the run byte-for-byte.

---

## 8. The end-to-end eval script

`eval/e2e.py` runs **every benchmark through the full pipeline** and prints one
table.

### 8.1 Per benchmark
1. Build the `@qjit` program; lower to MLIR (`keep_intermediate` / `catalyst-cli`).
2. Apply the Purl pass to that MLIR (`quantum-opt --purl calib=ibm_eagle_r3.json
   p=<p> shots=<S> carry-qubit=<q>`; leakage rides in via the JSON, §4.1); parse
   `purl.strategy`,
   `purl.C`, `purl.window`, `purl.predicted_fidelity`.
3. For each noise scale `lam`, measure delivered fidelity with the §5 simulator
   (same JSON) for the arms **unbounded**, **refresh (g1)**, **knit (g4)**. If an
   arm's C-window is empty (e.g. KNIT on `rus_lowp`, where `C_min ≈ 27 > C_max`),
   **skip it and mark `n/a`** with a "divergent-variance" flag — never force a `C`
   outside the window.
4. **C-sweep (refresh arm, one benchmark — `rus_lowp`):** at `lam=1`, sweep `C`
   across `purl.window` and record the simulator's best-fidelity `C*`. This is the
   empirical best-C that S4 checks the pass window brackets.

### 8.2 Output table (with legend)
Columns: `benchmark[config]`, `lam`, runtime coherent depth of the unbounded arm
(`depth/iter`, `min/mean/max_iters`, `runtime_depth`), delivered fidelity
(`unbounded`, `knit(g4)`, `refresh(g1)` — mean ± seed-std), the pass-selected
`strategy`, and **pass-predicted vs simulator-measured** fidelity for the selected
strategy. A legend defines every column, the arms, and the units. Also write a
CSV.

### 8.2.1 Reported metrics (beyond fidelity)

Purl is a **bias–variance–resource** tradeoff, so the eval reports four families of
metrics, each with a legend, and writes them to the CSV. All are per
`(benchmark, lam)` unless noted; `F` is the delivered-state Bloch fidelity and
`sigma_F` its seed-std.

**1. Estimator accuracy** — the user runs the loop to estimate `⟨O⟩`, so the error in
that estimate is the bottom line.
- `infidelity = 1 − F` — the delivered-state systematic error (bias).
- `RMSE = sqrt((1−F)² + sigma_F²)` — the delivered-state root-mean-square error,
  combining the systematic infidelity (bias) and the statistical spread `sigma_F`.
  This is the *measured analog* of the §3.5 predicted `RMSE = sqrt(bias² +
  statistical²)`: refresh has low bias **and** zero cut-variance → low RMSE; knit has
  ~zero bias but inflated variance → high RMSE; unbounded has high bias.

**2. Sampling cost** (why refresh ≫ knit; invisible to fidelity).
- `V(C) = (1−q)/(1−gamma²·q)`, `q=(1−p)^C`, `gamma²=16` — the KNIT quasi-probability
  variance-inflation factor. `V(C_min)` large/divergent is *why* KNIT is inadmissible.
- `ESS = (Σw)²/Σw²` — effective sample size of the weighted KNIT estimator.
- `E[|w|] = gamma^{E[#cuts]}` — mean |weight|, a direct sampling-cost proxy.
- `shots-to-ε = sigma0²·V(C)/ε²` — shots KNIT needs to reach estimator error `ε`.

**3. Resource / structural** (the pass's job; noise-independent).
- `runtime_depth` — mean realized coherent depth of the unbounded arm
  (`mean_iters × B` gate-layers); `max_iters` is the decohering **tail**.
- `bounded_cap = C·B` — the compile-time coherent-depth **cap** the cut guarantees
  (the core structural result, independent of any noise model).
- `E[#cuts] = q/(1−q)` — expected cuts per shot (extra measure + reset + feedback).
- `added_ops` — ops the rewrite inserts (counter/guard/qcut); a compile-time (IR)
  quantity, measured by diffing op counts before/after `--purl`.

**4. Decision quality** (grading the §3.5 cost model, not one arm).
- `best_arm` — the arm with the minimum *measured* RMSE (the oracle).
- `regret = RMSE(strategy) − RMSE(best_arm)` — how far the pass-selected strategy is
  from optimal (`0` = the pass chose the best arm). The headline decision metric.
- `predicted ≈ measured` (S3) and the window brackets `C*` (S4), as in §8.4.

### 8.3 Config
Default `--ibm` (the §4 dataset), `--carry-qubit`, `--ibm-json` (calibration file;
leakage sweeps via `eval/variants.py` variant files, §4.1), `-S` (shots), `--seeds`
(default **8**, for the seed-std error bars), `--f` (coherence-budget fraction,
default **0.05**), and `lam ∈ {0, 0.25, 0.5, 1, 2, 4}`.

### 8.4 Success criteria (paper claims)
- **S1 — pipeline:** every benchmark lowers to MLIR and the pass applies to it,
  emitting `purl.strategy`/`purl.C`/`purl.predicted_fidelity`.
- **S2 — refresh helps under real noise:** on `rus_lowp` with the IBM dataset,
  simulator-measured `refresh` fidelity > `unbounded` for `lam ≥ 0.5`, by a
  margin that grows with `lam` (non-overlapping error bars).
- **S3 — predicted ≈ measured:** pass-`purl.predicted_fidelity` agrees with the
  simulator-measured fidelity of the selected strategy to within **|ΔF| ≤ 0.02
  absolute, evaluated at `lam=1`** (the only operating point where the two are
  comparable — the pass predicts at nominal noise, §3.6).
- **S4 — window/strategy honesty:** the pass's window `[C_min,C_max]` **brackets
  the empirical best `C*`** from the refresh C-sweep (§8.1 step 4); the cost model
  selects `refresh` where the state is provably known and `none`/`knit` otherwise.
  The regime where cutting does NOT help is itself a reported result (e.g. p=5/8
  thin tail).
- **S5 — tests:** all §7 FileCheck tests green.

---

## 9. Explicit boundary (what "end-to-end" does and does not mean here)

"End-to-end" = **Python → real Catalyst lowering → real Purl pass → fidelity via
the shared-JSON simulator.** It does **not** mean the transformed MLIR is lowered
to a binary and executed on a noisy compiled backend — that requires a noisy
Catalyst runtime + a runtime for the cut ops and is out of scope. The pass and
the simulator are tied together by the **shared calibration JSON** and validated
by the predicted↔measured agreement (§8.4 S3). (A future extension could compile
and execute the refresh arm, whose output lowers without new runtime.)

### 9.1 Frontend & lowering requirements

**The benchmarks are expressible in Catalyst** — the `@qjit` forms use `@while_loop`
(dynamic, measurement-conditioned loop), `measure` (mid-circuit), `qml.cond`
(feedforward), and a never-measured target wire that crosses the `scf.while` carry
un-measured (the CARRY slot, §3.1). Ancillas are reset-and-reused via
`qml.cond(m, PauliX)`, not per-iteration allocation (fixed device register).
Toffoli-heavy gadgets can be slow to `qjit` — a performance, not an expressibility,
concern.

Requirements split by which side of the pass they touch:

**Input side — feeding `--purl` (no lowering change required).** Catalyst lowers
`@qjit`+`while_loop`+`measure`+`cond` to quantum-dialect `scf.while` with
register-threaded qubits, tensor-wrapped classical values, and
`stablehlo`/`tensor.extract` glue — the shape §3.1–3.3 must handle. Capture it
with `keep_intermediate`/`catalyst-cli`, then `quantum-opt --purl`. The one
constraint is **pass ordering**: `--purl` must run while MCM is still
`quantum.measure` + `scf.while` — *before* `dynamic-one-shot` rewrites measurements
for sampling and *before* gate decomposition flattens the body — and *before*
placement (Purl changes the circuit, §3.0 note).

**Output side — compiling the transformed IR (the real work, largely out of scope
per §9):**
1. **Purl dialect (`purl.qcut`) + `--purl-lower-qcut`** — a small self-contained
   dialect and its lowering pass (§3.7). Additive; no change to the core `quantum`
   dialect (it reuses `!quantum.bit` and the quantum observable attr).
2. **Reset** — refresh needs measure→|0>; no native `quantum.reset`, so expand as
   measure + conditional-X (§3.7c). A native reset op would be cleaner (optional).
3. **Catalyst-faithful carry surgery** — the counter (`i32`) and KNIT weight (`f64`)
   added to the `scf.while` carry must follow the **tensor-wrapped classical
   convention** (`tensor<...>` + `from_elements`/`extract`) for the transformed IR to
   re-lower (bufferization → LLVM). This is the main re-lowering risk. For the
   analysis-and-study scope (which does not re-lower, §9), the rewrite may emit bare
   `i32`/`f64` carries — valid MLIR the pass and `--analyze-only` consume; targeting
   execution requires the tensor-wrapped form here.
4. **Profile & placement metadata in the IR** — to retire the `p`/`shots`/`sigma0`/
   `carry-qubit` options (§3.0): a `while_loop` success-rate attribute (`p`),
   observable variance (`sigma0`), shots from the qjit config, and a wire→physical
   placement attribute (`carry-qubit`).
5. **Noisy execution backend** — to *measure* (not predict) fidelity of the lowered
   output; `lightning` is noiseless. This is the gap §9 scopes out; the study
   substitutes the trajectory simulator via the shared JSON.
6. **KNIT runtime surface** — the KNIT arm also needs the `@purl_sample_term` RNG
   hook and `expval`→weighted-sample legalization (`sample`/`counts` + runtime RNG).
   The **refresh** arm needs none of this (`expval` survives), which is why refresh
   is the clean end-to-end path.

---

## 10. Repository layout (deliverables)

```
purl/
  pass/            # Purl MLIR passes (in the Catalyst tree)
    Purl.cpp          # --purl: analysis + rewrite; emits purl.qcut
    LowerQCut.cpp     # --purl-lower-qcut: mechanical expansion of purl.qcut
    PurlDialect.td    # the Purl-owned dialect
    PurlOps.td        # the purl.qcut op definition (+ purl attrs)
    tests/*.mlir      # FileCheck tests (§7): insertion and lowering, run by lit
  sim/             # NumPy simulator (reads the shared JSON)
    qsim.py  knit_runtime.py  fast_target.py  ibm_dataset.py  validate.py
  benchmarks/      # rus_rx_ibm / rus_chain / rus_lowp: @qjit program + mirror
    *.py  ibm_eagle_r3.json
  eval/
    e2e.py         # the end-to-end pipeline + table (§8)
    plots.py
  results/         # CSV + table dump + figures
  NOTES.md         # findings (§11)
```

## 11. Findings the design encodes (honest physics)

- Under **Markovian** noise the *quasi* cut gives **zero** error reduction (the
  quasi-probability cut is exactly the identity superoperator; channels compose)
  — proven by density matrix. It helps only against **leakage / non-Markovian,
  reset-clearable** error.
- The **refresh** cut (proven known state) *does* escape the no-go: it replaces
  the noisy state with the ideal one, removing all accumulated error at gamma=1,
  zero variance, and no coherence-budget-limited variance floor.
- For high p (RUS, 5/8) the corrupted tail is ~5% of shots and the KNIT floor
  `C_min ≈ 3/p` (variance cap `V_max=4`, §3.3; refresh has no floor) tracks the mean
  trip count — so cutting only ever touches the ~6% tail regardless of p; the
  benefit is real but small unless the tail is heavy (`rus_lowp`) and the cut is
  cheap (refresh).
- On **real IBM Eagle r3** coherence, `rus_lowp` (mean hold ~10 iters) decoheres
  substantially and refresh recovers several percent of delivered fidelity —
  the headline positive result.
