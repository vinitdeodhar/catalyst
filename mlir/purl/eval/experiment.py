"""
experiment.py -- the Purl experiment (spec 8): UNBOUNDED vs KNIT(g4) vs REFRESH(g1),
no discard arm (spec 3.5).

Reports, per benchmark and noise level:
  * runtime coherent depth  -- the unbounded coherent depth k (mean/min/max trip)
  * fidelity_unbounded   -- delivered-state Bloch fidelity, unbounded arm
  * fidelity_refresh     -- delivered-state Bloch fidelity, gamma=1 refresh arm
  * fidelity_knit        -- gamma=4 knit arm (reconstructed, |a|<=1; n/a if the
                            variance window is empty)

"Coherent depth" is the length of the carried qubit's longest unbroken run with
no measurement/reset. Unbounded: it equals the realized trip count k (unbounded
in the tail). Knit: it is capped at the cut period C, by construction.

Usage:
  PYTHONPATH=. python3 eval/experiment.py                # rus_rx_ibm (p=5/8)
  PYTHONPATH=. python3 eval/experiment.py --bench rus_lowp
Outputs results/experiment.csv.
"""

import argparse
import csv
import math
import os

import numpy as np

from sim.fast_target import fast_unbounded, fast_knit, fast_refresh
from eval.run_eval import window, variance_V, EVAL_CALIB

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
IDEAL_BLOCH = np.array([1 / math.sqrt(2), 0.5, 0.5])  # H T H T H |0>

# body depth B in gate-layers per loop body (pass `purl.body_layers` on the
# Catalyst-emitted IR). rus_rx_ibm = the IBM 2-control Toffoli body.
B_LAYERS = {"rus_rx_ibm": 12, "rus_lowp": 12}

# a good-memory device so the window opens at low p (rus_lowp); rus uses EVAL_CALIB.
# p_leak is the per-2q-gate leakage the carried wire accrues (spec 4.1) -- it only
# bites on a target-entangling benchmark (rus_lowp's touch), zero when idle.
GOOD_MEM = dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                p_leak=1e-3)


def bloch_fidelity(runner, calib, lam, C, p, S, seeds, touch=False, clamp=False):
    """Delivered-state Bloch fidelity vs the ideal magic state (S/3 per axis).
    Returns (mean fidelity, seed-std of fidelity) -- the std exposes knit's
    sampling variance vs refresh's lack of it. `clamp` (knit arm) clips the
    reconstructed Bloch vector to |a|<=1 (spec 5.1): the weight-summed estimate
    can be non-physical at finite shots."""
    per_seed = []
    for sd in range(seeds):
        comp = {}
        for basis in ("X", "Y", "Z"):
            rng = np.random.default_rng(1300 + sd)
            vals = []
            for _ in range(S // 3):
                if runner is fast_unbounded:
                    v = runner(rng, calib, lam, basis=basis, p=p, touch=touch)[0]
                else:
                    v = runner(rng, calib, lam, C, basis=basis, p=p, touch=touch)[0]
                vals.append(v)
            comp[basis] = float(np.mean(vals))
        a = np.array([comp["X"], comp["Y"], comp["Z"]])
        if clamp:                               # spec 5.1: clip |a| <= 1
            norm = float(np.linalg.norm(a))
            if norm > 1.0:
                a = a / norm
        per_seed.append(0.5 * (1.0 + float(a @ IDEAL_BLOCH)))
    return float(np.mean(per_seed)), float(np.std(per_seed))


def depths(calib, lam, p, S, seeds, touch=False):
    """Realized trip counts on the unbounded arm -> mean/max coherent depth."""
    ks = []
    for sd in range(seeds):
        rng = np.random.default_rng(1400 + sd)
        for _ in range(S):
            ks.append(fast_unbounded(rng, calib, lam, p=p, touch=touch)[1])
    return float(np.mean(ks)), int(np.min(ks)), int(np.max(ks))


def refresh_c_sweep(calib, p, S, seeds, C_lo, C_hi, touch):
    """Spec 8.1 step 4: at lam=1 sweep C over the window on the refresh arm and
    return the empirical best-fidelity C* (what S4 checks the window brackets)."""
    best_C, best_F = None, -1.0
    for C in range(max(1, C_lo), C_hi + 1):
        F, _ = bloch_fidelity(fast_refresh, calib, 1.0, C, p, S, seeds, touch=touch)
        if F > best_F:
            best_F, best_C = F, C
    return best_C, best_F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="rus_rx_ibm",
                    choices=["rus_rx_ibm", "rus_lowp"])
    ap.add_argument("-S", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ibm", action="store_true",
                    help="run on real IBM Eagle r3 per-qubit hardware data")
    ap.add_argument("--carry-qubit", type=int, default=0,
                    help="physical qubit index for the carried wire (--ibm)")
    ap.add_argument("--leak", type=float, default=None,
                    help="per-2q-gate leakage prob on the carried qubit "
                         "(--ibm; default = published IBM estimate)")
    args = ap.parse_args()

    if args.bench == "rus_lowp":
        import benchmarks.rus_lowp as bench
        p = bench.P_ANALYTIC
    else:
        import benchmarks.rus_rx_ibm as bench
        p = bench.P_ANALYTIC

    # rus_lowp entangles the carried target each iteration (spec 5.1) -> per-2q-gate
    # leakage accrues on it; rus_rx_ibm holds it idle (no 2q gate) -> zero leakage.
    touch = (args.bench == "rus_lowp")

    if args.ibm:
        # real IBM Eagle r3 data for the carried qubit; leakage is the separate
        # published per-2q-gate estimate (spec 4.1), charged by qsim on each 2q gate.
        from sim.ibm_dataset import carried_calib, IBM_LEAK_PER_2Q
        from sim.qsim import load_calib
        leak = args.leak if args.leak is not None else IBM_LEAK_PER_2Q
        calib = load_calib(carried_calib(args.carry_qubit, p_leak=leak))
        src = f"IBM Eagle r3 (qubit {args.carry_qubit}, p_leak/2q={leak:g})"
    else:
        calib = dict(GOOD_MEM) if args.bench == "rus_lowp" else dict(EVAL_CALIB)
        src = "held-memory device" if args.bench == "rus_lowp" else "leakage-dominated"
    B = B_LAYERS[args.bench]
    C_min, C_max = window(p, calib=calib)
    knit_ok = C_min <= C_max          # KNIT admissible only if its window is non-empty
    C = C_min                         # quasi (gamma=4) cut: at the variance-cap floor
    # refresh (gamma=1) cut: NO variance floor (window [1,C_max]), so bound tightly
    C_refresh = min(3, C_max) if C_max >= 1 else 1
    S, seeds = args.S, args.seeds

    print(f"=== Purl experiment: UNBOUNDED vs KNIT(g4) vs REFRESH(g1) ===")
    print(f"benchmark {args.bench}   p={p}   calib: {src}")
    print(f"  T1={calib['T1']*1e6:.0f}us  T2={calib['T2']*1e6:.0f}us  "
          f"readout={calib['readout']*1e9:.0f}ns  2q_err={calib['p2']:.1e}  "
          f"p_leak/2q={calib.get('p_leak', 0):.1e}  B={B} layers/body   "
          f"window=[{C_min},{C_max}]")
    if knit_ok:
        print(f"  knit  cut (g4): C = C_min = {C}   ({C} iters = {C*B} layers)")
    else:
        print(f"  knit  cut (g4): window EMPTY (C_min={C_min} > C_max={C_max}) "
              f"-> KNIT inadmissible (divergent variance); reported n/a")
    print(f"  refresh   (g1): C = {C_refresh}   ({C_refresh} iters = "
          f"{C_refresh*B} layers)  [no variance floor -> tight bound]")

    # spec 8.1 step 4 / S4: refresh C-sweep at lam=1 -> empirical best C*
    Cstar, Fstar = refresh_c_sweep(calib, p, S, seeds, 1, max(1, C_max), touch)
    brackets = 1 <= Cstar <= C_max
    print(f"  refresh C-sweep (lam=1): best C* = {Cstar} (F={Fstar:.4f}); "
          f"window [1,{C_max}] {'brackets' if brackets else 'does NOT bracket'} "
          f"C*  [S4]")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "experiment.csv")
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["benchmark", "lam", "p", "C_knit", "C_refresh", "B_layers",
                "depth_per_iter", "min_number_of_iterations",
                "mean_number_of_iterations", "max_number_of_iterations",
                "runtime_depth", "fidelity_unbounded", "fidelity_knit",
                "fidelity_refresh", "std_knit", "std_refresh", "knit_admissible"])

    # short benchmark+config tag carried in every row
    tag = f"{args.bench}[p={p:g},Cr={C_refresh},B={B}]"
    tw = max(len(tag), 22)
    print(f"\n{'bench[config]':<{tw}} | {'lam':>5} | "
          f"{'runtime coherent depth (unbounded)':>50} | "
          f"{'fidelity (mean, seed-std)':>42}")
    print(f"{'':<{tw}} | {'':>5} | {'depth/iter':>10} "
          f"{'min_iters':>9} {'mean_iters':>10} {'max_iters':>9} "
          f"{'runtime_depth':>14} | "
          f"{'unbounded':>11} {'knit(g4)*':>16} {'refresh(g1)':>15}")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        mk, nk, xk = depths(calib, lam, p, S, seeds, touch=touch)
        fu, _ = bloch_fidelity(fast_unbounded, calib, lam, C, p, S, seeds,
                               touch=touch)
        fd, sd = bloch_fidelity(fast_refresh, calib, lam, C_refresh, p, S, seeds,
                                touch=touch)
        if knit_ok:
            fq, sq = bloch_fidelity(fast_knit, calib, lam, C, p, S, seeds,
                                    touch=touch, clamp=True)
            knit_cell = f"{fq:8.4f}±{sq:.4f}"
            fq_csv, sq_csv = round(fq, 4), round(sq, 4)
        else:
            fq_csv = sq_csv = ""
            knit_cell = f"{'n/a':>15}"
        print(f"{tag:<{tw}} | {lam:5.2f} | {B:10d} "
              f"{nk:9d} {mk:10.2f} {xk:9d} {mk*B:14.0f} | "
              f"{fu:11.4f} {knit_cell:>16} {fd:8.4f}±{sd:.4f}")
        w.writerow([args.bench, lam, p, (C if knit_ok else ""), C_refresh, B, B,
                    nk, round(mk, 3), xk, round(mk * B, 1),
                    round(fu, 4), fq_csv, round(fd, 4), sq_csv, round(sd, 4),
                    int(knit_ok)])
    fh.close()

    print("\nlegend:")
    print("  lam            global noise scale (0 = noiseless, 1 = calibrated "
          "device, 2/4 = noisier)")
    print(f"  depth/iter     B = {B} gate-layers per loop body (the pass's "
          f"purl.body_layers)")
    print("  number_of_iterations  realized trip count per shot = the carried "
          "qubit's RUNTIME coherent")
    print("                 depth in loop iterations (unbounded arm); "
          "min_iters / mean_iters / max_iters over shots")
    print(f"  runtime_depth  mean_iters x depth/iter, the mean runtime coherent "
          f"depth in gate-layers")
    print("  fidelity       delivered-state Bloch fidelity vs the ideal state "
          "(higher = better),")
    print("                 shown mean +/- seed-std over seeds")
    print("  arms           unbounded = no cutting (control);  "
          "refresh(g1) = deterministic gamma=1 cut")
    print("                 (proven known state, zero-variance);  knit(g4) = "
          "general gamma^2=16 cut")
    print("  knit(g4)*      *reconstructed: the weight-summed Bloch vector is "
          "clipped to |a|<=1 (spec 5.1);")
    print("                 shown n/a where the KNIT variance window is empty "
          "(C_min > C_max)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
