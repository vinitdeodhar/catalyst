"""
close_loop.py -- Milestone 6: MLIR-in -> transformed-MLIR -> executed evaluation.

The loop-knit PASS is the compile-time half; the knit_runtime executor is its
faithful runtime (the pass output's inline cut protocol is exactly what
sim/knit_runtime.cut_and_reprepare implements). This script closes the loop:

  1. run quantum-opt --loop-knit on the canonical RUS program, parse the
     pass-emitted knit.C and knit.window out of the transformed IR;
  2. run the knit executor at THAT C on the benchmark and confirm it reproduces
     the ideal <Z> at lam=0 (unbiased) -- i.e. the compile-time-selected cut
     period drives a correct runtime estimator;
  3. confirm the pass's window matches the eval's own window() computation.

Run:  PYTHONPATH=. python3 eval/close_loop.py
"""

import os
import re
import subprocess
import sys

import numpy as np

import benchmarks.rus_rx_ibm as rus
from sim.knit_runtime import run_knit
from eval.run_eval import window
from sim.qsim import load_calib

QOPT = "/home/vadeo/catalyst/mlir/build/bin/quantum-opt"
REWRITE_IN = os.path.join(os.path.dirname(__file__), os.pardir,
                          "pass", "tests", "rewrite_rus.mlir")


def pass_params(p=0.625):
    """Run the pass (analyze-only) and parse knit.C / knit.window from the IR."""
    out = subprocess.run(
        [QOPT, f"--loop-knit=analyze-only=true p={p}", REWRITE_IN],
        capture_output=True, text=True, check=True).stdout
    C = int(re.search(r"knit\.C = (\d+)", out).group(1))
    win = re.search(r"knit\.window = array<i64: (\d+), (\d+)>", out)
    return C, (int(win.group(1)), int(win.group(2)))


def main():
    if not os.path.exists(QOPT):
        print("quantum-opt not built; skipping M6")
        return 0

    C_pass, win_pass = pass_params(0.625)
    print(f"pass-selected  C = {C_pass}   window = {win_pass}")

    # eval's own window (unit / layer mode, no coherence ceiling)
    C_min, C_max = window(0.625, calib=load_calib("unit"))
    print(f"eval window()  [C_min, C_max] = [{C_min}, {C_max}]  (C_min={C_min})")
    assert C_pass == C_min, "pass C must equal the variance-floor C_min"
    assert win_pass[0] == C_min, "window lower bound mismatch"

    # run the knit executor at the pass-selected C; must be unbiased at lam=0
    rng = np.random.default_rng(2026)
    S = 12000
    ests = [run_knit(rus, C_pass, rng, lam=0.0)[0] for _ in range(S)]
    mu, se = float(np.mean(ests)), float(np.std(ests, ddof=1) / np.sqrt(S))
    print(f"knit@C={C_pass} (lam=0): <Z> = {mu:+.4f} +- {se:.4f}  "
          f"(ideal {rus.Z_IDEAL})")
    ok = abs(mu - rus.Z_IDEAL) <= 4 * se
    print("MILESTONE 6:", "PASS -- loop closed (pass C -> runtime knit -> ideal)"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
