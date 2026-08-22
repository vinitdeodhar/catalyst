# Purl — Compiler-Managed Cutting of Unbounded Quantum Loops

Purl detects **carry-type dynamic quantum loops** — loops that hold a live qubit
across a measurement-conditioned `while` — and, when a hardware cost model says it
is profitable, **cuts** the carried wire to bound its coherent depth, recovering
delivered-state fidelity that unbounded holding loses to decoherence.

Purl has two halves that share **one real-hardware calibration JSON**:

1. **An MLIR compiler pass** (`--purl` + `--purl-lower-qcut`) in the Catalyst tree
   that classifies the loop, proves the carried state, selects a cut strategy, and
   rewrites the loop via an abstract `purl.qcut` op.
2. **A pure-NumPy noise simulator + eval harness** (this directory) that measures
   delivered fidelity on the *same* JSON, cross-validating the pass's prediction.

The full design is in [`doc/specs/PURL_SPEC.md`](../../doc/specs/PURL_SPEC.md).

---

## 1. Repository layout

```
doc/specs/PURL_SPEC.md              # the specification
mlir/include/Purl/ , mlir/lib/Purl/ # the Purl dialect + the two passes (C++)
mlir/test/Quantum/Purl/             # FileCheck / lit tests
frontend/catalyst/passes/           # the @qjit decorators (purl, purl_lower_qcut)
mlir/purl/                          # THIS package — simulator + benchmarks + eval
  sim/        qsim.py knit_runtime.py fast_target.py ibm_dataset.py validate.py
  benchmarks/ rus_rx_ibm.py rus_lowp.py rus_chain.py ibm_eagle_r3.json
  eval/       experiment.py run_eval.py plots.py ...
  results/    experiment.csv (+ figures)
```

---

## 2. Prerequisites and build

### 2a. The Python eval (no compiler build needed)

The simulator and eval are pure Python and only need **NumPy**:

```bash
python3 -m pip install numpy
```

You can run every experiment in §4 with just this — the eval does not invoke the
compiler.

### 2b. The MLIR pass and tools (needed for §3 and the lit tests)

Purl builds as part of Catalyst's MLIR dialects. From the repo root:

```bash
# one-time: build LLVM/MLIR, StableHLO, Enzyme (slow, hours the first time)
make llvm stablehlo enzyme

# build the Catalyst dialects, including the Purl dialect + passes.
# Produces mlir/build/bin/{quantum-opt, catalyst}.
make dialects

# fast incremental rebuild after editing the pass:
cmake --build mlir/build --target quantum-opt catalyst-cli
```

### 2c. The `@qjit` frontend (needed to run Purl inside a compiled program)

```bash
make frontend        # or: pip install -e frontend
```

`@qjit` invokes the `catalyst` CLI built in 2b (`mlir/build/bin/catalyst`), so
rebuild `catalyst-cli` after any pass change.

---

## 3. Using the pass

### 3a. On MLIR directly (`quantum-opt`)

`--purl` runs the analysis + rewrite (emits `purl.qcut`); `--purl-lower-qcut`
expands it into concrete ops. Run them in sequence:

```bash
mlir/build/bin/quantum-opt \
  --purl="calib=ibm_eagle_r3.json p=0.1 shots=20000 carry-qubit=0" \
  --purl-lower-qcut  program.mlir
```

Use `--purl="... analyze-only=true"` to emit the `purl.*` analysis attributes
(`purl.class`, `purl.known_state`, `purl.strategy`, `purl.window`,
`purl.predicted_fidelity`, …) **without** rewriting. Full option glossary: spec §3.0.

### 3b. Inside a `@qjit` program

Apply the passes as QNode decorators (order matters — analysis then lowering):

```python
import pennylane as qml
from catalyst import qjit, while_loop, measure
from catalyst.passes import purl, purl_lower_qcut

@qjit
@purl_lower_qcut
@purl(calib="ibm_eagle_r3.json", p=0.1, shots=20000)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def rus():
    qml.Hadamard(0); qml.T(0); qml.Hadamard(0); qml.T(0); qml.Hadamard(0)  # held |psi0>
    @while_loop(lambda cont: cont)
    def loop(cont):
        qml.Hadamard(1); m = measure(1); return m   # coin on wire 1; wire 0 held
    loop(True)
    return qml.expval(qml.PauliZ(0))

print(rus())   # compiles + runs; Purl cuts the held wire when profitable
```

`purl(...)` accepts `calib, p, shots, margin, sigma0, C, f, depth` (the
placement/hardware knobs `carry-qubit` / `p-leak` go via
`catalyst.passes.apply_pass("purl", **{"carry-qubit": 3})`).

---

## 4. Running experiments

All eval commands run from **this directory** (`mlir/purl/`) with `PYTHONPATH=.`.

### 4a. The main comparison table — `eval/experiment.py`

Sweeps a noise scale `lam` and reports delivered Bloch fidelity for the
**unbounded**, **refresh (γ=1)**, and **knit (γ=4)** arms.

```bash
# rus_lowp on real IBM Eagle r3 data (the headline heavy-tail case)
PYTHONPATH=. python3 eval/experiment.py --bench rus_lowp --ibm -S 6000 --seeds 8

# the primary thin-tail benchmark
PYTHONPATH=. python3 eval/experiment.py --bench rus_rx_ibm --ibm
```

Options:

| flag | default | meaning |
|---|---|---|
| `--bench {rus_rx_ibm,rus_lowp}` | `rus_rx_ibm` | which benchmark |
| `--ibm` | off | use the real IBM Eagle r3 per-qubit dataset (else a synthetic calib) |
| `--carry-qubit N` | `0` | which physical qubit the carried wire maps to (`--ibm`) |
| `--leak X` | published est. | per-2q-gate leakage probability on the carried qubit (`--ibm`) |
| `-S S` | `6000` | total shots per fidelity point |
| `--seeds K` | `8` | independent seeds (the ± is the seed-std) |

Output goes to the console (table + legend) and `results/experiment.csv`.

### 4b. The full sweep set — `eval/run_eval.py`

Long-format CSV (`results/eval.csv`) with crossover, C-window, chain, and tail
sweeps, consumed by `eval/plots.py`.

```bash
PYTHONPATH=. python3 eval/run_eval.py            # default (S=4000, 6 seeds)
PYTHONPATH=. python3 eval/run_eval.py --fast     # smoke config (S=1200, 3 seeds)
PYTHONPATH=. python3 eval/run_eval.py --full     # spec config (S=20000, 20 seeds)
PYTHONPATH=. python3 eval/run_eval.py -S 8000 --seeds 12   # custom
```

### 4c. Regenerate the IBM dataset

```bash
PYTHONPATH=. python3 sim/ibm_dataset.py          # (re)writes benchmarks/ibm_eagle_r3.json
```

### 4d. Validation gates — `sim/validate.py`

Checks the noiseless (`lam=0`) benchmark reproduces its ideal ⟨Z⟩ and the idle
|+⟩-over-T2 coherence gate. Run this first if you change the simulator.

```bash
PYTHONPATH=. python3 sim/validate.py
```

---

## 5. Reading the table

```
                          |  lam | ...runtime coherent depth... |   unbounded     knit(g4)*   refresh(g1)
rus_lowp[p=0.1,Cr=2,B=12] | 1.00 | 12  1  9.93  94  119         |     0.9017        n/a      0.9733±0.007
```

- **lam** — global noise scale (`0` = noiseless, `1` = calibrated device, `2/4` = noisier).
- **runtime coherent depth** — realized trip count `k` on the unbounded arm
  (`depth/iter` × `mean_iters`); the tail is where holding decoheres.
- **fidelity** — delivered-state Bloch fidelity vs the ideal state (higher is
  better), mean ± seed-std.
- **arms** — `unbounded` (no cutting), `refresh(g1)` (deterministic γ=1 cut of a
  proven state, zero variance), `knit(g4)` (general γ²=16 quasi cut; shown `n/a`
  where its variance window is empty).

Refresh beating unbounded (non-overlapping bars), with the gap growing in `lam`, is
the target result (spec S2). The header line also reports the refresh C-sweep and
whether the pass window brackets the empirical best `C*` (spec S4).

---

## 6. Running the lit tests

```bash
# after building quantum-opt (§2b)
mlir/llvm-project/build/bin/llvm-lit -sv mlir/build/test/Quantum/Purl
# or the whole dialect suite:
make test-mlir      # (= cmake --build mlir/build --target check-dialects)
```

---

## 7. Adding a new benchmark

A benchmark is a **carry-type** loop: a held carried wire (the delivered state)
plus a measurement-conditioned coin. The simplest new benchmark reuses the held
magic state `|psi0> = H T H T H |0>` and only changes the trip distribution and/or
whether the coin entangles the target.

1. **Create `benchmarks/<name>.py`** exposing the carried-qubit model:

   ```python
   from sim.qsim import QSim
   from benchmarks.rus_rx_ibm import N_WIRES, TARGET, ANCILLAS, Z_IDEAL, prepare_input

   P_ANALYTIC = 0.3                 # per-iteration success probability

   def attempt(sim):
       # OPTIONAL: entangle the held target so per-2q-gate leakage accrues (and
       # refresh has leakage to clear). Omit for an idle-target benchmark.
       sim.touch_2q(TARGET)         # net-identity CZ touch (spec 5.1)
       a, b, c = ANCILLAS           # target-independent coin; idles the target
       sim.h(a); sim.h(b); sim.h(c)
       sim.measure(a); sim.measure(b); sim.measure(c)
       sim.feedback(active=[])
       sim.force_zero(a); sim.force_zero(b); sim.force_zero(c)
       return bool(sim.rng.random() < (1.0 - P_ANALYTIC))
   ```

   Keep the loop body **provably identity (or a known Pauli) on the held wire** so
   the pass proves a known state and can refresh — a `touch_2q` (CZ with a `|0>`
   partner) is net-identity; a measurement of the target is not.

2. **Register it in `eval/experiment.py`:**
   - add `"<name>"` to the `--bench` `choices`;
   - add an `import benchmarks.<name> as bench` branch in `main`;
   - add `"<name>": <layers>` to `B_LAYERS`;
   - set the `touch` flag if your coin entangles the target
     (`touch = args.bench in ("rus_lowp", "<name>")`).

3. **If the carried state ≠ `H T H T H |0>`**, also update the ideal target used by
   the fast executors: `IDEAL_BLOCH` in `eval/experiment.py` and `_prep_psi0` in
   `sim/fast_target.py` (and `Z_IDEAL` in your benchmark).

4. **(Optional) MLIR side** — add a lit test under `mlir/test/Quantum/Purl/` with
   the Catalyst-emitted IR shape for your loop (see `register_refresh.mlir` for the
   real `!quantum.reg`-threaded shape) and appropriate `CHECK` lines.

5. **Run it:**
   ```bash
   PYTHONPATH=. python3 sim/validate.py                      # sanity (lam=0)
   PYTHONPATH=. python3 eval/experiment.py --bench <name> --ibm
   ```

---

## 8. Further reading

- [`doc/specs/PURL_SPEC.md`](../../doc/specs/PURL_SPEC.md) — full specification
  (classification, known-state proof, cost model, the `purl.qcut` op + lowering,
  the shared JSON schema, success criteria, and the honest physics findings).
- `sim/qsim.py` — the trajectory simulator and its noise model.
- `mlir/lib/Purl/Transforms/Purl.cpp` / `LowerQCut.cpp` — the two passes.
