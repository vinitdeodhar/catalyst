"""
experiment.py -- the loop-knitting experiment: UNBOUNDED vs KNIT (no discard).

Reports, per benchmark and noise level:
  * bounded depth        -- knit's capped coherent depth  C            (= C*B layers)
  * actual depth         -- the unbounded coherent depth  k            (= k*B layers)
                            (mean and max realized trip count)
  * fidelity_unbounded   -- delivered-state Bloch fidelity, unbounded arm
  * fidelity_knit        -- delivered-state Bloch fidelity, knit arm

"Coherent depth" is the length of the carried qubit's longest unbroken run with
no measurement/reset. Unbounded: it equals the realized trip count k (unbounded
in the tail). Knit: it is capped at the cut period C, by construction.

Usage:
  PYTHONPATH=. python3 eval/experiment.py                # rus_rx_ibm (p=5/8)
  PYTHONPATH=. python3 eval/experiment.py --bench heralded
Outputs results/experiment.csv.
"""

import argparse
import csv
import math
import os

import numpy as np

from sim.fast_target import fast_unbounded, fast_knit, fast_knit_det
from eval.run_eval import window, EVAL_CALIB

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
IDEAL_BLOCH = np.array([1 / math.sqrt(2), 0.5, 0.5])  # H T H T H |0>

# body depth B in gate-layers per loop body (pass `knit.body_layers` on the
# Catalyst-emitted IR). rus_rx_ibm = the IBM 2-control Toffoli body.
B_LAYERS = {"rus_rx_ibm": 12, "heralded": 12}

# a good-memory device so the window opens at low p (heralded); rus uses EVAL_CALIB
GOOD_MEM = dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                T_leak=4e-3)


def bloch_fidelity(runner, calib, lam, C, p, S, seeds):
    """Delivered-state Bloch fidelity vs the ideal magic state (S/3 per axis).
    Returns (mean fidelity, seed-std of fidelity) -- the std exposes knit's
    sampling variance vs the deterministic cut's lack of it."""
    per_seed = []
    for sd in range(seeds):
        comp = {}
        for basis in ("X", "Y", "Z"):
            rng = np.random.default_rng(1300 + sd)
            vals = []
            for _ in range(S // 3):
                if runner is fast_unbounded:
                    v = runner(rng, calib, lam, basis=basis, p=p)[0]
                else:
                    v = runner(rng, calib, lam, C, basis=basis, p=p)[0]
                vals.append(v)
            comp[basis] = float(np.mean(vals))
        a = np.array([comp["X"], comp["Y"], comp["Z"]])
        per_seed.append(0.5 * (1.0 + float(a @ IDEAL_BLOCH)))
    return float(np.mean(per_seed)), float(np.std(per_seed))


def depths(calib, lam, p, S, seeds):
    """Realized trip counts on the unbounded arm -> mean/max coherent depth."""
    ks = []
    for sd in range(seeds):
        rng = np.random.default_rng(1400 + sd)
        for _ in range(S):
            ks.append(fast_unbounded(rng, calib, lam, p=p)[1])
    return float(np.mean(ks)), int(np.min(ks)), int(np.max(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="rus_rx_ibm",
                    choices=["rus_rx_ibm", "heralded"])
    ap.add_argument("-S", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ibm", action="store_true",
                    help="run on real IBM Eagle r3 per-qubit hardware data")
    ap.add_argument("--carry-qubit", type=int, default=0,
                    help="physical qubit index for the carried wire (--ibm)")
    ap.add_argument("--leak", type=float, default=None,
                    help="per-iteration idle leakage prob on the carried qubit "
                         "(--ibm; default = published IBM estimate)")
    args = ap.parse_args()

    if args.bench == "heralded":
        import benchmarks.heralded as bench
        p = bench.P_ANALYTIC
    else:
        import benchmarks.rus_rx_ibm as bench
        p = bench.P_ANALYTIC

    if args.ibm:
        # real IBM Eagle r3 data for the carried qubit; leakage is the separate
        # published-estimate knob applied as an idle process over the body.
        from sim.ibm_dataset import carried_calib, IBM_LEAK_PER_2Q
        from sim.qsim import load_calib
        calib = load_calib(carried_calib(args.carry_qubit))
        dt = 3 * calib["readout"] + 3 * calib["gate_1q"] + calib["tau"]
        leak = args.leak if args.leak is not None else IBM_LEAK_PER_2Q
        calib["T_leak"] = dt / leak if leak > 0 else float("inf")
        src = f"IBM Eagle r3 (qubit {args.carry_qubit}, leak/iter={leak:g})"
    else:
        calib = GOOD_MEM if args.bench == "heralded" else EVAL_CALIB
        src = "held-memory device" if args.bench == "heralded" else "leakage-dominated"
    B = B_LAYERS[args.bench]
    C_min, C_max = window(p, calib=calib)
    C = C_min          # quasi (gamma=4) cut: bounded by the variance floor C_min
    # deterministic (gamma=1) cut: NO variance floor (C_min=1), so bound tightly
    C_det = min(3, C_max)
    S, seeds = args.S, args.seeds

    print(f"=== loop-knitting experiment: UNBOUNDED vs KNIT ===")
    print(f"benchmark {args.bench}   p={p}   calib: {src}")
    print(f"  T1={calib['T1']*1e6:.0f}us  T2={calib['T2']*1e6:.0f}us  "
          f"readout={calib['readout']*1e9:.0f}ns  2q_err={calib['p2']:.1e}  "
          f"B={B} layers/body   window=[{C_min},{C_max}]")
    print(f"  quasi cut (g4): C = C_min = {C}   ({C} iters = {C*B} layers)")
    print(f"  det   cut (g1): C = {C_det}   ({C_det} iters = {C_det*B} layers)  "
          f"[no variance floor -> tight bound]")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "experiment.csv")
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["benchmark", "lam", "p", "C", "B_layers", "depth_per_iter",
                "bounded_depth_iters", "bounded_depth_layers",
                "min_number_of_iterations", "mean_number_of_iterations",
                "max_number_of_iterations", "runtime_depth",
                "fidelity_unbounded", "fidelity_knit_quasi",
                "fidelity_knit_det", "std_knit_quasi", "std_knit_det"])

    # short benchmark+config tag carried in every row
    tag = f"{args.bench}[p={p:g},C={C},B={B}]"
    tw = max(len(tag), 22)
    print(f"\n{'bench[config]':<{tw}} | {'lam':>5} | "
          f"{'runtime coherent depth (unbounded)':>50} | "
          f"{'fidelity (mean, seed-std)':>40}")
    print(f"{'':<{tw}} | {'':>5} | {'depth/iter':>10} "
          f"{'min_iters':>9} {'mean_iters':>10} {'max_iters':>9} "
          f"{'runtime_depth':>14} | "
          f"{'unbounded':>11} {'knit-quasi(g4)':>16} {'knit-det(g1)':>15}")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        mk, nk, xk = depths(calib, lam, p, S, seeds)
        fu, _ = bloch_fidelity(fast_unbounded, calib, lam, C, p, S, seeds)
        fq, sq = bloch_fidelity(fast_knit, calib, lam, C, p, S, seeds)
        fd, sd = bloch_fidelity(fast_knit_det, calib, lam, C_det, p, S, seeds)
        print(f"{tag:<{tw}} | {lam:5.2f} | {B:10d} "
              f"{nk:9d} {mk:10.2f} {xk:9d} {mk*B:14.0f} | "
              f"{fu:11.4f} {fq:8.4f}±{sq:.4f} {fd:8.4f}±{sd:.4f}")
        w.writerow([args.bench, lam, p, C, B, B, C_det, C_det * B,
                    nk, round(mk, 3), xk, round(mk * B, 1),
                    round(fu, 4), round(fq, 4), round(fd, 4),
                    round(sq, 4), round(sd, 4)])
    fh.close()

    print("\nlegend:")
    print("  lam            global noise scale (0 = noiseless, 1 = calibrated "
          "device, 2/4 = noisier)")
    print(f"  depth/iter     B = {B} gate-layers per loop body (the pass's "
          f"knit.body_layers)")
    print("  number_of_iterations  realized trip count per shot = the carried "
          "qubit's RUNTIME coherent")
    print("                 depth in loop iterations (unbounded arm); "
          "min_iters / mean_iters / max_iters over shots")
    print(f"  runtime_depth  mean_iters x depth/iter, the mean runtime coherent "
          f"depth in gate-layers")
    print(f"  bounded_depth  C x B = {C_det*B} layers -- the compile-time depth "
          f"cap the knit cut guarantees")
    print("  fidelity       delivered-state Bloch fidelity vs the ideal state "
          "(higher = better),")
    print("                 shown mean +/- seed-std over seeds")
    print("  arms           unbounded = no cutting (control);  "
          "knit-quasi(g4) = general gamma^2=16 cut;")
    print("                 knit-det(g1) = deterministic gamma=1 cut "
          "(proven known state, zero-variance)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
