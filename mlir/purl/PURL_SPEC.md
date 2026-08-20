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
  simulator-*measured* fidelity to within a few percent (§8.4).

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
iteration boundary). Equivalently: **trip count k is determined by runtime
measurement results and has unbounded support** (`P(k > N) > 0` for all N).

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

## 3. The Purl MLIR pass

Substrate: a C++ pass in Catalyst's `mlir/lib/Quantum/Transforms/`, registered in
`quantum-opt` (and buildable as a pass plugin for `@qjit`). CLI: `--purl`.
Attributes are `purl.*`. Options: `calib`, `p`, `C`, `f`, `shots`, `margin`,
`sigma0`, `carry-qubit`, `p-leak`, `depth`, `analyze-only`.

The analyses (3.1–3.3) MUST handle the *actual Catalyst-emitted IR shape*
(tensor-wrapped classical values, `stablehlo`/`tensor.extract` glue,
register-threaded qubits). The rewrite (3.5) fires on carry loops with an
`expval` output.

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
`C_min = ceil(ln(gamma^2)/ln(1/(1-p)))` (variance floor, gamma=4);
`C_max = floor(f*T2/(B+tau))` (coherence budget). Emit `purl.window=[C_min,C_max]`.
The deterministic (refresh) strategy has **no** variance floor, so its window is
`[1, C_max]`.

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
Per-iteration error is split into **transportable** `eps_t` (depolarizing +
T1/T2 idle over B+tau) and **non-transportable** `eps_nt` (leakage per 2q gate /
readout; `p-leak` is a separate knob). `eps_cut = p_ro + p_prep`. With
`q=(1-p)^C`, `E[k]=1/p`, `E[#cuts]=q/(1-q)`, `V(C)=(1-q)/(1-16q)`, and `sbar(C)`
= expected delivered-state age, the predicted expval error (RMSE of a bias + a
statistical term) for each strategy:

| strategy | bias | statistical |
|---|---|---|
| NONE | eps_all·E[k] | sigma0/sqrt(S) |
| REFRESH@C (tier-1, gamma=1) | eps_all·sbar(C) + E[#cuts]·eps_cut | sigma0/sqrt(S) |
| KNIT@C (tier-3, gamma=4) | eps_t·E[k] + eps_nt·sbar(C) + E[#cuts]·eps_cut | sigma0·sqrt(V(C)/S) |

Minimize each *applicable* strategy over its C-range (REFRESH `[1,C_max]`; KNIT
`[C_min,C_max]`, `16q<1`); pick the arg-min; fire it iff
`predicted·margin < predicted(NONE)`, else leave the loop unchanged. Emit
`purl.strategy` ∈ {none, refresh, knit}, `purl.C`, and the auditable
`purl.predicted = {none, refresh, knit}`. (No discard arm.)

**REFRESH rewrite (gamma=1)** — fires when `known_state ∈ {identity, pauli}`:
extend the carry with an `i32` counter; every C failing iterations
`measure`+reset the carried wire and **re-prepare the known |psi0>** (a KnownPauli
adds a counter-parity correction before the observable). No weight, no RNG hook,
zero variance; the **`quantum.expval` output survives** (a refresh delivers a
genuine quantum state — no legalization). This is the only strategy that escapes
the Markovian no-go (it replaces the noisy state with the ideal one).

**KNIT rewrite (gamma=4)** — the general quasi-probability cut: extend the carry
with an `i32` counter + `f64` weight; the periodic guarded cut expands inline
(`func.call @purl_sample_term` RNG hook → basis change → measure → reset →
eigenstate prep → `4·sigma·s` weight); legalize the `expval` output to a weighted
Z sample. Idempotent (`purl.applied`).

### 3.6 Real-hardware fidelity prediction
`calib` may be a **per-qubit + coupling-map** hardware dataset (§4). The carried
wire's physical qubit (`carry-qubit`) supplies T1/T2/1q-err/readout; the median
2q error comes from the coupling map. Predict the carried qubit's delivered
fidelity with a **time-based model**:

```
F(D) = exp(-t/T1)·exp(-t/T2)·(1-e1q)^(D·n1q)·(1-e2q)^(D·n2q)·(1-leak)^(D·n2q),
t_idle = D·(B+tau)
```

Emit `purl.predicted_fidelity = {unbounded, bounded}` — the **mean** delivered
fidelity over the geometric trip distribution (unbounded age = k; bounded/refresh
age = ((k-1) mod C)+1) — and, if `depth>0`, `purl.fidelity_at_depth`.

---

## 4. The shared IBM hardware dataset (one JSON, two consumers)

`ibm_eagle_r3.json` — a 127-qubit **IBM Eagle r3** dataset with representative
published medians and a deterministic per-qubit spread:

- top level: `gate_1q_time`, `gate_2q_time`, `readout_time`, `tau`, `p_prep`;
- `qubits[]`: per-qubit `T1`, `T2`, `gate_1q_err`, `readout_err`;
- `edges[]`: heavy-hex coupling map with per-edge `gate_2q_err`.

Representative medians: T1 ~ 250 µs, T2 ~ 150 µs, sx 1q-err ~ 2.5e-4 (32 ns),
ECR 2q-err ~ 8e-3 (560 ns), readout-err ~ 1.3e-2 (1.2 µs), tau ~ 1 µs.
**Leakage is NOT in the dataset** (IBM does not publish it) — it is a separate
knob `p-leak` (published estimate ~1e-3 per 2q gate) supplied to both the pass
and the simulator.

The **same file** feeds (a) the pass's depth-in-seconds, cost model, and fidelity
prediction, and (b) the simulator's noise model (§5). A generator writes the JSON
deterministically; a loader extracts a flat per-carried-qubit calibration.

---

## 5. The noise-modelling simulator (reads the same JSON)

A pure-NumPy Monte-Carlo **trajectory simulator** (statevector per shot; noise via
stochastic channel sampling), independent of Catalyst at runtime.

- **Op set:** alloc, h, x, z, s, sdg, t, cnot, toffoli, measure (Born + collapse),
  reset, force_zero (fresh qubit).
- **Noise (parameterised by the §4 JSON + `p-leak` + a global scale `lam`∈[0,4]):**
  per-1q/2q depolarizing; readout flip + pre-measure depolarizing; **idle
  amplitude-damping + pure-dephasing** over the per-iteration idle time (from
  T1/T2); and **leakage** (population leaving the computational subspace during
  idle; a fresh qubit / cut clears it — the reset-clearable error the transform
  reduces). `lam=0` is exactly noiseless.
- **Executors** mirroring the pass's arms: `unbounded`, `refresh` (measure +
  force_zero + re-prep known |psi0>), `knit` (inline quasi-probability cut with
  weights). Delivered-state fidelity is measured by **3-basis tomography**
  (estimate ⟨X⟩,⟨Y⟩,⟨Z⟩ over S/3 shots each; F = ½(1 + a·n_ideal)).
- **Validation gates:** at `lam=0` the primary benchmark reproduces its noiseless
  ⟨Z⟩; an idle |+⟩ over T2 shows ⟨X⟩ ≈ e^-1.

---

## 6. Benchmarks (PennyLane/Catalyst → MLIR)

Each benchmark exists as (a) a **PennyLane/Catalyst `@qjit` program** using
`catalyst.while_loop` + mid-circuit `measure`, which **lowers to MLIR** (captured
via `keep_intermediate` / `catalyst-cli`) and is the input the Purl pass is
applied to; and (b) a **Python mirror** driving the §5 simulator, kept in lockstep
for the fidelity study. All three are carry-type, hold the non-Clifford magic
state `|psi0> = H T H T H |0>`, and return `qml.expval(PauliZ(target))`.

| Name | p | Role |
|---|---|---|
| `rus_rx_ibm` | 5/8 | primary; IBM-tutorial RUS shape (2 controls + 1 held target) |
| `rus_chain(N)` | 5/8 / stage | N sequential RUS gates on one held data qubit (N ∈ {1,2,4,8}) |
| `rus_lowp` | 0.1 | low-p heavy-tail regime (quantum-repeater / heralded memory); mean trip count 10 — where cutting is meant to help |

NOTE: the carried qubit's physics is identical across the three; they differ in
the trip-count distribution (p) and stage count (N).

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

Each asserts the relevant `purl.*` attributes and, for rewrites, the transformed
structure (carry extension, guard, cut expansion / reset+re-prep, output).

---

## 8. The end-to-end eval script

`eval/e2e.py` runs **every benchmark through the full pipeline** and prints one
table.

### 8.1 Per benchmark
1. Build the `@qjit` program; lower to MLIR (`keep_intermediate` / `catalyst-cli`).
2. Apply the Purl pass to that MLIR (`quantum-opt --purl calib=ibm_eagle_r3.json
   p=<p> shots=<S> p-leak=<leak> carry-qubit=<q>`); parse `purl.strategy`,
   `purl.C`, `purl.predicted_fidelity`.
3. For each noise scale `lam`, measure delivered fidelity with the §5 simulator
   (same JSON) for the arms **unbounded**, **refresh (g1)**, **knit (g4)**.

### 8.2 Output table (with legend)
Columns: `benchmark[config]`, `lam`, runtime coherent depth of the unbounded arm
(`depth/iter`, `min/mean/max_iters`, `runtime_depth`), delivered fidelity
(`unbounded`, `knit(g4)`, `refresh(g1)` — mean ± seed-std), the pass-selected
`strategy`, and **pass-predicted vs simulator-measured** fidelity for the selected
strategy. A legend defines every column, the arms, and the units. Also write a
CSV.

### 8.3 Config
Default `--ibm` (the §4 dataset), `--carry-qubit`, `--leak` (published estimate),
`-S`, `--seeds`, and `lam ∈ {0, 0.25, 0.5, 1, 2, 4}`.

### 8.4 Success criteria (paper claims)
- **S1 — pipeline:** every benchmark lowers to MLIR and the pass applies to it,
  emitting `purl.strategy`/`purl.C`/`purl.predicted_fidelity`.
- **S2 — refresh helps under real noise:** on `rus_lowp` with the IBM dataset,
  simulator-measured `refresh` fidelity > `unbounded` for `lam ≥ 0.5`, by a
  margin that grows with `lam` (non-overlapping error bars).
- **S3 — predicted ≈ measured:** pass-`purl.predicted_fidelity` agrees with the
  simulator-measured fidelity of the selected strategy to within a few percent.
- **S4 — window/strategy honesty:** the pass's window brackets the simulator's
  best C; the cost model selects `refresh` where the state is provably known and
  `none`/`knit` otherwise. The regime where cutting does NOT help is itself a
  reported result (e.g. p=5/8 thin tail).
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

---

## 10. Repository layout (deliverables)

```
purl/
  pass/            # Purl MLIR pass (in the Catalyst tree; --purl)
    Purl.cpp
    tests/*.mlir   # FileCheck tests (§7), run by lit
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
- For high p (RUS, 5/8) the corrupted tail is ~5% of shots and `C_min ≈ 2.77/p`
  tracks the mean trip count — so cutting only ever touches the ~6% tail
  regardless of p; the benefit is real but small unless the tail is heavy
  (`rus_lowp`) and the cut is cheap (refresh).
- On **real IBM Eagle r3** coherence, `rus_lowp` (mean hold ~10 iters) decoheres
  substantially and refresh recovers several percent of delivered fidelity —
  the headline positive result.
