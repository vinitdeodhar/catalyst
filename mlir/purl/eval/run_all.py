"""
eval/run_all.py -- the eval suite. Compiles each benchmark @qjit program (both Purl
passes in the pipeline) and runs it on the noisy QSimDevice, uniformly: ONE path for
every benchmark -- compile, then average the returned expval over trajectories. The
passes select and lower the cut; the device applies the calibrated noise. No
per-benchmark execution logic.

  PYTHONPATH=. python3 eval/run_all.py            # default
  PYTHONPATH=. python3 eval/run_all.py --fast     # quick smoke

Requires the qsim runtime device built (make runtime) and the Catalyst frontend.
"""

import os
import sys

import numpy as np

from eval.programs import PROGRAMS

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
LAMS = (0.0, 1.0, 4.0)


def run(name, builder, lams, seeds, shots):
    """One benchmark across noise scales. Each (lam, seed) is a separate compile
    that averages `shots` one-shot noise trajectories; the seeds give error bars.
    Returns rows of (lam, mean, seed-std, ideal)."""
    rows = []
    for lam in lams:
        vals, ideal = [], None
        for sd in range(seeds):
            f, ideal = builder(lam, seed=sd, shots=shots)
            vals.append(float(f()))
        v = np.array(vals)
        rows.append((lam, v.mean(), v.std(ddof=1) if seeds > 1 else 0.0, ideal))
    return rows


def main(argv=()):
    fast = "--fast" in argv
    seeds, shots = (3, 400) if fast else (6, 1500)
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit("=== Purl eval suite (compiled @qjit + both passes, run on qsim device) ===")
    emit(f"one path per benchmark; one-shot trajectory averaging, shots={shots}, "
         f"seeds={seeds}; noise lam in {LAMS}")
    for name, builder in PROGRAMS.items():
        emit(f"\n# {name}")
        emit(f"  {'lam':>5} | {'<O> (mean±std)':>18} | {'ideal':>6} | {'|Δ|':>7}")
        for lam, mean, std, ideal in run(name, builder, LAMS, seeds, shots):
            emit(f"  {lam:5.2f} | {mean:9.4f} ± {std:.4f} | {ideal:6.3f} | "
                 f"{abs(mean - ideal):7.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "suite.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
