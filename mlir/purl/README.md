# Purl — Compiler-Managed Cutting of Unbounded Quantum Loops

Purl detects **carry-type dynamic quantum loops** — loops that hold a live qubit
across a measurement-conditioned `while` — and, when a hardware cost model says it
is profitable, **cuts** the carried wire to bound its coherent depth, recovering
delivered-state fidelity that unbounded holding loses to decoherence.

Purl has two halves that share **one real-hardware calibration JSON**:

1. **An MLIR compiler pass** (`--purl` + `--purl-lower-qcut`) in the Catalyst tree
   that classifies the loop, proves the carried state, selects a **cut strategy**, and
   rewrites the loop (via an abstract `purl.renew` op, or a direct SWAP for migrate).
2. **A pure-NumPy noise simulator + eval harness** (this directory) that measures
   delivered fidelity on the *same* JSON, cross-validating the pass's prediction.

**Cut strategies** (see §5.1): `refresh` (γ=1, proven-known state → re-prepare it),
`knit` (γ²=16 quasi-probability, unknown state, kept as a comparison arm), `migrate`
(γ=1, unknown state → SWAP onto a fresh partner; the cost model's default for unknown
states), or `none`.

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
  benchmarks/ rus_rx_ibm.py rus_lowp.py rus_chain.py            # RUS family
              pump.py ipe_project.py rus_data.py qwalk.py       # newer benchmarks
              ibm_eagle_r3.json                                 # the shared calibration
  eval/       experiment.py run_eval.py variants.py plots.py    # headline + sweeps
              ipe_project.py rus_data.py migrate.py qwalk.py     # per-benchmark evals
  results/    experiment.csv, <benchmark>.txt (+ figures)
```

`sim/validate.py` is the single gate suite for the simulator + every benchmark; run
it after any simulator change.

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

`--purl` runs the analysis + rewrite (refresh/knit emit `purl.renew`; migrate emits a
SWAP directly); `--purl-lower-qcut` expands `purl.renew` into concrete ops. Run them
in sequence:

```bash
mlir/build/bin/quantum-opt \
  --purl="calib=ibm_eagle_r3.json p=0.1 shots=20000 carry-qubit=0" \
  --purl-lower-qcut  program.mlir
```

Use `--purl="... analyze-only=true"` to emit the `purl.*` analysis attributes
(`purl.class`, `purl.known_state`, `purl.strategy`, `purl.window`, `purl.pair`,
`purl.predicted_fidelity`, …) **without** rewriting. Full option glossary: spec §3.0.

Notable options: `force-knit=true` selects knit for an unknown state (the paper's
comparison arm) instead of the default migrate; `age-trigger=true` searches a first-cut
threshold for knit (spec §12). **Leakage is calibration data now** — it lives in the
JSON (`leak_2q_default` / per-edge `leak_2q`), not a `--leak`/`p-leak` knob.

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

`purl(...)` accepts `calib, p, shots, margin, sigma0, C, f, depth` (the placement knob
`carry-qubit` and flags like `force-knit` go via
`catalyst.passes.apply_pass("purl", **{"carry-qubit": 3})`). Leakage comes from the
calibration JSON, not a knob.

---

## 4. Running experiments

All eval commands run from **this directory** (`mlir/purl/`) with `PYTHONPATH=.`.

### 4a. The main comparison table — `eval/experiment.py`

Sweeps a noise scale `lam` and reports delivered Bloch fidelity for the
**unbounded**, **refresh (γ=1)**, and **knit (γ=4)** arms, plus the coherent-depth,
RMSE, and decision columns. It also prints a **`pass strategy:`** line — the strategy
the cost model would select for that benchmark (refresh / migrate / none).

```bash
# rus_lowp on real IBM Eagle r3 data (the headline heavy-tail case)
PYTHONPATH=. python3 eval/experiment.py --bench rus_lowp --ibm -S 6000 --seeds 8

# the primary thin-tail benchmark
PYTHONPATH=. python3 eval/experiment.py --bench rus_rx_ibm --ibm
```

Options:

| flag | default | meaning |
|---|---|---|
| `--bench {rus_rx_ibm,rus_lowp,ipe,pump}` | `rus_rx_ibm` | which benchmark |
| `--ibm` | off | use the real IBM Eagle r3 per-qubit dataset (else a synthetic calib) |
| `--ibm-json PATH` | bundled | calibration JSON (use `eval/variants.py` for leakage sweeps) |
| `--carry-qubit N` | `0` | which physical qubit the carried wire maps to (`--ibm`) |
| `-S S` | `6000` | total shots per fidelity point |
| `--seeds K` | `8` | independent seeds (the ± is the seed-std) |

Leakage is no longer a CLI flag — it is the `leak_2q_default` key in the JSON; sweep it
with `eval/variants.py` (writes variant calibration files). Output goes to the console
(table + legend) and `results/experiment.csv`.

### 4a′. Per-benchmark evals (the newer benchmarks + strategies)

Each writes a same-named `results/<name>.txt`. Run from `mlir/purl/` with `PYTHONPATH=.`:

| script | benchmark | what it shows |
|---|---|---|
| `eval/ipe_project.py` | `ipe_project` | knit-only projection; leakage ablation + a refresh **falsification** arm |
| `eval/rus_data.py` | `rus_data` | RUS `V3` on data; noise sweep + leakage sweep |
| `eval/migrate.py` | `rus_data` | unbounded vs knit vs **migrate** (γ=1 vs γ²=16) |
| `eval/qwalk.py` | `qwalk` | fat-tailed migrate headline; `--leak-sweep` for the gain-vs-leakage table |

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

One suite of gates for the simulator **and every benchmark**: the noiseless (`lam=0`)
⟨Z⟩ + idle-|+⟩-over-T2 coherence gates, the leakage-schema loader, and a per-benchmark
gate for `pump`, `ipe_project`, `rus_data`, `qwalk`, plus the migrate-strategy
semantics (ping-pong, leaked-transfer, cost threshold). Run it first if you change the
simulator.

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
  where its variance window is empty). The per-benchmark evals (§4a′) add a
  **`migrate(g=1)`** arm for unknown states.

Refresh beating unbounded (non-overlapping bars), with the gap growing in `lam`, is
the target result (spec S2). The header line also reports the refresh C-sweep and
whether the pass window brackets the empirical best `C*` (spec S4).

### 5.1 Strategies and benchmarks

| strategy | γ | carried state | cut action |
|---|---|---|---|
| `refresh` | 1 | proven known | measure + reset + re-prepare the known state (`purl.renew`) |
| `knit` | 4 (γ²=16) | unknown (comparison arm / `force-knit`) | quasi-probability wire cut, threads a signed weight (`purl.renew`) |
| `migrate` | 1 | unknown (cost-model default, spec §13) | SWAP the state onto a fresh partner (3 CNOTs) + reset the abandoned wire; `purl.pair` |
| `none` | — | any | not profitable → loop unchanged |

| benchmark | shape | expected strategy |
|---|---|---|
| `rus_rx_ibm` / `rus_chain` | Toffoli-coin RUS, unknown | none / migrate |
| `rus_lowp` | CNOT-heralded, identity | **refresh** (heavy-tail headline) |
| `pump` | 3× CNOT-sandwich, identity, 6 2q gates/iter | **refresh** (leakage-heavy) |
| `ipe_project` | controlled-Rz projection, unknown | none / knit / migrate |
| `rus_data` | Paetznick–Svore `V3` RUS on data, unknown | knit / migrate |
| `qwalk` | fat-tailed random-walk herald, 2q-heavy, unknown | **migrate** (the headline migrate win) |

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
   - add `"<name>": <n2q>` to `N2Q_PER_ITER` (per-iteration 2q-gate count on the held
     wire — 0 = idle target, 1 = a single touch, 6 = pump/qwalk-style; drives leakage).

   For an unknown-state or bespoke benchmark, a **dedicated eval** (like
   `eval/ipe_project.py` / `eval/rus_data.py` / `eval/qwalk.py`) is usually cleaner
   than the shared `experiment.py` fast path — copy the closest one.

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
  (classification, known-state proof, cost model, the strategies, the `purl.renew` op
  + lowering, the migrate strategy §13, the shared JSON leakage schema, success
  criteria, and the honest physics findings).
- `sim/qsim.py` — the trajectory simulator and its noise model.
- `mlir/lib/Purl/Transforms/Purl.cpp` / `LowerQCut.cpp` — the two passes.
