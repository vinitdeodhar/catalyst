"""
run_eval.py -- fidelity evaluation sweeps (spec Section 6). Writes long-format
results/eval.csv consumed by eval/plots.py.

Arms (all on the SAME simulator, noise model, and TOTAL shot budget S):
  UNBOUNDED       -- input program as-is (tail decoherence)
  KNIT            -- inline cut every C failing iterations (charged variance)
(the truncate+discard baseline has been removed from the measurements.)

Sweeps:
  1 crossover   lam in {0,.25,.5,1,2,4}   rus_rx_ibm    E and Bloch fidelity
  2 c_window    C in {2,3,4,6,8} at lam=1 rus_rx_ibm    E (+ C=2 instability)
  3 chain       N in {1,2,4,8} at lam=1   rus_chain     E, arms unbounded vs knit
  4 tail        unbounded, per-k error    rus_rx_ibm    motivates the transform

Usage:
  PYTHONPATH=. python3 eval/run_eval.py           # fast preset
  PYTHONPATH=. python3 eval/run_eval.py --full     # spec S=20000, 20 seeds
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

from sim.fast_target import fast_unbounded, fast_truncated, fast_knit
from sim.qsim import load_calib
import benchmarks.rus_rx_ibm as rus
import benchmarks.rus_chain as chain

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
IDEAL_BLOCH = np.array([1 / math.sqrt(2), 0.5, 0.5])  # H T H T H |0>

# Evaluation calibration: LEAKAGE-DOMINATED so the transform's error-reduction is
# real (see NOTES.md -- Markovian noise gives knit zero benefit). Weak T1/T2,
# low gate/readout error, moderate reset-clearable leakage T_leak.
EVAL_CALIB = {
    "gate_1q": 30e-9, "gate_2q": 60e-9, "readout": 1e-6, "tau": 1e-6,
    "T1": 400e-6, "T2": 400e-6,
    "p1": 2e-4, "p2": 1e-3, "p_ro": 2e-3, "p_meas": 1e-3,
    "T_leak": 150e-6,
}


def window(p, gamma2=16.0, f=0.05, calib=None):
    """Pass cut-period window (spec 3.3). Returns (C_min, C_max)."""
    C_min = math.ceil(math.log(gamma2) / math.log(1.0 / (1.0 - p)))
    if calib is None or math.isinf(calib.get("T2", math.inf)):
        C_max = C_min + 2
    else:
        B_plus_tau = calib["readout"] * 3 + calib["tau"]  # per-iter idle proxy
        C_max = int(f * calib["T2"] / B_plus_tau)
    return C_min, max(C_min, C_max)


# ---------------------------------------------------------------------------
# per-arm estimators over a shot budget S
# ---------------------------------------------------------------------------
def est_unbounded(rng, S, lam, N=1, basis="Z"):
    zs, trips = [], []
    for _ in range(S):
        z, k = fast_unbounded(rng, EVAL_CALIB, lam, N=N, basis=basis)
        zs.append(z)
        trips.append(k)
    return float(np.mean(zs)), {"trips": trips}


def est_discard(rng, S, lam, C, N=1, basis="Z"):
    kept, ntr, trips = [], 0, []
    for _ in range(S):
        z, k, trunc = fast_truncated(rng, EVAL_CALIB, lam, C, N=N, basis=basis)
        trips.append(k)
        if trunc:
            ntr += 1
        else:
            kept.append(z)
    est = float(np.mean(kept)) if kept else float("nan")
    return est, {"discard_rate": ntr / S, "n_kept": len(kept), "trips": trips}


def est_knit(rng, S, lam, C, N=1, basis="Z"):
    ests, ws, ncut, trips = [], [], 0, []
    for _ in range(S):
        e, k, nc, w = fast_knit(rng, EVAL_CALIB, lam, C, N=N, basis=basis)
        ests.append(e)
        ws.append(abs(w))
        ncut += (nc > 0)
        trips.append(k)
    ests = np.asarray(ests)
    ess = (ests.sum() ** 2 / np.sum(ests ** 2)) if np.any(ests) else 0.0
    return float(np.mean(ests)), {
        "cut_rate": ncut / S, "mean_absw": float(np.mean(ws)),
        "ess": float(ess), "trips": trips,
    }


def _est(arm, bench, rng, S, lam, C, N, basis="Z"):
    if arm == "unbounded":
        return est_unbounded(rng, S, lam, N=N, basis=basis)
    if arm == "discard":
        return est_discard(rng, S, lam, C, N=N, basis=basis)
    return est_knit(rng, S, lam, C, N=N, basis=basis)


def bloch_fidelity(arm, bench, rng, S, lam, C, N=1):
    """Estimate delivered Bloch vector (S/3 per axis) and fidelity vs ideal."""
    comps = {}
    for basis in ("X", "Y", "Z"):
        e, _ = _est(arm, bench, rng, S // 3, lam, C, N, basis=basis)
        comps[basis] = e
    a = np.array([comps["X"], comps["Y"], comps["Z"]])
    F = 0.5 * (1.0 + float(a @ IDEAL_BLOCH))  # <psi_ideal| rho |psi_ideal>
    return F, a


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def sweep_crossover(writer, S, seeds, C, base_seed=1000):
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for lam in lams:
        for s in range(seeds):
            rng = np.random.default_rng(base_seed + s)
            for arm in ("unbounded", "knit"):
                est, diag = _est(arm, rus, rng, S, lam, C, 1)
                E = abs(est - rus.Z_IDEAL)
                row(writer, "rus_rx_ibm", arm, lam, C, 1, s, "E", E)
                row(writer, "rus_rx_ibm", arm, lam, C, 1, s, "est", est)
                for key in ("discard_rate", "cut_rate", "mean_absw", "ess"):
                    if key in diag:
                        row(writer, "rus_rx_ibm", arm, lam, C, 1, s, key, diag[key])
                F, _ = bloch_fidelity(arm, rus, rng, S, lam, C)
                row(writer, "rus_rx_ibm", arm, lam, C, 1, s, "fidelity", F)


def sweep_c_window(writer, S, seeds, calib, base_seed=2000):
    Cs = [2, 3, 4, 6, 8]
    lam = 1.0
    for C in Cs:
        for s in range(seeds):
            rng = np.random.default_rng(base_seed + s)
            for arm in ("unbounded", "knit"):
                est, diag = _est(arm, rus, rng, S, lam, C, 1)
                row(writer, "rus_rx_ibm", arm, lam, C, 1, s, "E",
                    abs(est - rus.Z_IDEAL))
                if arm == "knit":
                    row(writer, "rus_rx_ibm", arm, lam, C, 1, s, "mean_absw",
                        diag["mean_absw"])


def sweep_chain(writer, S, seeds, C, base_seed=3000):
    lam = 1.0
    for N in chain.CHAIN_LENGTHS:
        for s in range(seeds):
            rng = np.random.default_rng(base_seed + s)
            for arm in ("unbounded", "knit"):
                est, diag = _est(arm, chain, rng, S, lam, C, N)
                row(writer, "rus_chain", arm, lam, C, N, s, "E",
                    abs(est - chain.Z_IDEAL))
                if arm == "knit":
                    row(writer, "rus_chain", arm, lam, C, N, s, "cut_rate",
                        diag["cut_rate"])


def sweep_tail(writer, S, seeds, base_seed=4000):
    """Unbounded arm: per-shot delivered error vs realized trip count k."""
    for lam in (1.0, 2.0):
        byk = defaultdict(list)
        for s in range(seeds):
            rng = np.random.default_rng(base_seed + s)
            for _ in range(S):
                z, k = fast_unbounded(rng, EVAL_CALIB, lam, N=1, basis="Z")
                byk[k].append(z)
        for k in sorted(byk):
            if len(byk[k]) >= 30:
                err = abs(float(np.mean(byk[k])) - rus.Z_IDEAL)
                # store k in the N column, seed=-1 (aggregated)
                row(writer, "rus_rx_ibm", "unbounded", lam, 0, k, -1,
                    "err_at_k", err)
                row(writer, "rus_rx_ibm", "unbounded", lam, 0, k, -1,
                    "n_at_k", len(byk[k]))


def row(writer, benchmark, arm, lam, C, N, seed, metric, value):
    writer.writerow([benchmark, arm, lam, C, N, seed, metric, value])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="spec S=20000, 20 seeds")
    ap.add_argument("--fast", action="store_true", help="tiny smoke config")
    ap.add_argument("-S", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    if args.full:
        S, seeds = 20000, 20
    elif args.fast:
        S, seeds = 1200, 3
    else:
        S, seeds = 4000, 6
    if args.S:
        S = args.S
    if args.seeds:
        seeds = args.seeds

    calib = load_calib(EVAL_CALIB)
    C_min, C_max = window(rus.P_ANALYTIC, calib=calib)
    C = C_min
    print(f"config: S={S} seeds={seeds}  p={rus.P_ANALYTIC}  "
          f"window=[{C_min},{C_max}]  C={C}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "eval.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "arm", "lam", "C", "N", "seed", "metric", "value"])
        # stash window on a meta row for plots
        w.writerow(["_meta", "window", "", C_min, C_max, "", "C", C])
        print("sweep 1: crossover ...");  sweep_crossover(w, S, seeds, C)
        print("sweep 2: c_window ...");   sweep_c_window(w, S, seeds, calib)
        print("sweep 3: chain ...");      sweep_chain(w, S, seeds, C)
        print("sweep 4: tail ...");       sweep_tail(w, S, seeds)
    print("wrote", out)


if __name__ == "__main__":
    main()
