"""
eval/run_all.py -- the eval suite. Compiles each benchmark @qjit program (both Purl
passes in the pipeline) and runs it on the noisy QSimDevice, uniformly: ONE path for
every benchmark -- compile, then average the returned expval over trajectories. The
passes select and lower the cut; the device applies the calibrated noise. No
per-benchmark execution logic.

  PYTHONPATH=. python3 eval/run_all.py                     # prompts for hardware
  PYTHONPATH=. python3 eval/run_all.py --hardware heron    # or eagle
  PYTHONPATH=. python3 eval/run_all.py --hardware eagle --fast

Requires the qsim runtime device built (make runtime) and the Catalyst frontend, and
the chosen hardware JSON generated (sim/ibm_dataset.py writes eagle + heron).
"""

import os
import sys

import numpy as np

from eval.programs import PROGRAMS, probe_strategy
from sim.ibm_dataset import HARDWARE, build_hardware

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
LAMS = (0.0, 1.0, 4.0)


def pick_hardware(argv):
    """Resolve the target hardware: --hardware {eagle,heron}; else prompt (or default
    eagle when non-interactive). Returns (name, calib_json_path)."""
    name = None
    for i, a in enumerate(argv):
        if a == "--hardware" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--hardware="):
            name = a.split("=", 1)[1]
    if name is None:
        if sys.stdin.isatty():
            name = input(f"Which hardware? {list(HARDWARE)} [eagle]: ").strip() or "eagle"
        else:
            name = "eagle"
    if name not in HARDWARE:
        raise SystemExit(f"unknown hardware {name!r}; choose from {list(HARDWARE)}")
    _, _, path = HARDWARE[name]
    if not os.path.exists(path):
        build_hardware(name)          # generate the JSON on first use
    return name, path


def run(name, builder, lams, seeds, shots, calib):
    """One benchmark across noise scales. Each (lam, seed) is a separate compile
    that averages `shots` one-shot noise trajectories; the seeds give error bars.
    Returns rows of (lam, mean, seed-std, ideal)."""
    rows = []
    for lam in lams:
        vals, ideal = [], None
        for sd in range(seeds):
            f, ideal = builder(lam, seed=sd, shots=shots, calib=calib)
            vals.append(float(f()))
        v = np.array(vals)
        rows.append((lam, v.mean(), v.std(ddof=1) if seeds > 1 else 0.0, ideal))
    return rows


def main(argv=()):
    fast = "--fast" in argv
    hw, calib = pick_hardware(argv)
    seeds, shots = (3, 400) if fast else (6, 1500)
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit(f"=== Purl eval suite ({hw}): compiled @qjit + both passes on qsim device ===")
    emit(f"calib={os.path.basename(calib)}; one-shot trajectory averaging, "
         f"shots={shots}, seeds={seeds}; noise lam in {LAMS}")
    for name, builder in PROGRAMS.items():
        # spec §8.2: report the PASS-selected strategy (its own in-pipeline decision)
        dec = probe_strategy(builder, calib)
        cut = f", C={dec['C']}" if dec["C"] is not None else ""
        predf = (f", pass-predicted F(bounded)={dec['predicted_bounded']:.4f}"
                 if dec["predicted_bounded"] is not None else "")
        emit(f"\n# {name}  ->  pass strategy: {dec['strategy']}"
             f"{' (applied)' if dec['applied'] else ' (not applied)'}{cut}{predf}")
        emit(f"  {'lam':>5} | {'<O> (mean±std)':>18} | {'ideal':>6} | {'|Δ|':>7}")
        for lam, mean, std, ideal in run(name, builder, LAMS, seeds, shots, calib):
            emit(f"  {lam:5.2f} | {mean:9.4f} ± {std:.4f} | {ideal:6.3f} | "
                 f"{abs(mean - ideal):7.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"suite_{hw}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
