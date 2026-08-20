# loop-knitting — Track-B evaluation harness

Pure-NumPy reference executor and evaluation for the `--loop-knit` pass
(`mlir/lib/Quantum/Transforms/loop_knitting.cpp`). This is a research/evaluation
harness — it is **not** built or run by CMake/lit; it validates the pass's
physics and cost model out-of-band.

```
sim/
  qsim.py          trajectory simulator + noise model (depol / T1-T2 idle / leakage)
  knit_runtime.py  inline cut protocol, weights, arm executors
  fast_target.py   exact target-only fast path
  ibm_dataset.py   IBM Eagle r3 (127q) per-qubit + coupling dataset generator
  validate.py      simulator/estimator validation gates
benchmarks/        rus_rx_ibm (p=5/8), rus_chain, heralded (low p) + ibm_eagle_r3.json
eval/
  experiment.py    UNBOUNDED vs KNIT: depths + delivered fidelities (supports --ibm)
  run_eval.py      sweeps -> results/eval.csv
  plots.py         figures
```

## Run

```bash
cd mlir/loop-knitting
PYTHONPATH=. python3 sim/ibm_dataset.py                 # write benchmarks/ibm_eagle_r3.json
PYTHONPATH=. python3 sim/validate.py                    # validation gates
PYTHONPATH=. python3 eval/experiment.py --bench heralded --ibm --carry-qubit 0
```

`eval/experiment.py --ibm` builds the simulator calibration from the carried
qubit's real IBM Eagle r3 data (`--carry-qubit N`) plus the separate published
leakage estimate (`--leak`, default ~1e-3), and reports unbounded vs the two
knit cuts. On the low-p heralded loop this reproduces the pass's
`knit.predicted_fidelity`: the deterministic gamma=1 refresh recovers several
percent of delivered fidelity that idle decoherence costs the unbounded hold.

Requires only `numpy` (and `matplotlib` for `plots.py`).
