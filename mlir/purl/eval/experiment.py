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

from sim.fast_target import fast_unbounded, fast_knit, fast_refresh, _prep_psi0
from eval.run_eval import window, variance_V, EVAL_CALIB

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
IDEAL_BLOCH = np.array([1 / math.sqrt(2), 0.5, 0.5])  # H T H T H |0>

# body depth B in gate-layers per loop body (pass `purl.body_layers` on the
# Catalyst-emitted IR). rus_rx_ibm = the IBM 2-control Toffoli body.
B_LAYERS = {"rus_rx_ibm": 12, "rus_lowp": 12, "ipe": 12}

# a good-memory device so the window opens at low p (rus_lowp); rus uses EVAL_CALIB.
# p_leak is the per-2q-gate leakage the carried wire accrues (spec 4.1) -- it only
# bites on a target-entangling benchmark (rus_lowp's touch), zero when idle.
GOOD_MEM = dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                p_leak=1e-3)


def bloch_fidelity(runner, calib, lam, C, p, S, seeds, touch=False, clamp=False,
                   ideal=None, prep=_prep_psi0):
    """Delivered-state Bloch fidelity vs the ideal held state (S/3 per axis).
    Returns (mean fidelity, seed-std of fidelity) -- the std exposes knit's
    sampling variance vs refresh's lack of it. `clamp` (knit arm) clips the
    reconstructed Bloch vector to |a|<=1 (spec 5.1). `ideal`/`prep` select the
    benchmark's held state (default the H T H T H |0> magic state)."""
    if ideal is None:
        ideal = IDEAL_BLOCH
    per_seed = []
    for sd in range(seeds):
        comp = {}
        for basis in ("X", "Y", "Z"):
            rng = np.random.default_rng(1300 + sd)
            vals = []
            for _ in range(S // 3):
                if runner is fast_unbounded:
                    v = runner(rng, calib, lam, basis=basis, p=p, touch=touch,
                               prep=prep)[0]
                else:
                    v = runner(rng, calib, lam, C, basis=basis, p=p, touch=touch,
                               prep=prep)[0]
                vals.append(v)
            comp[basis] = float(np.mean(vals))
        a = np.array([comp["X"], comp["Y"], comp["Z"]])
        if clamp:                               # spec 5.1: clip |a| <= 1
            norm = float(np.linalg.norm(a))
            if norm > 1.0:
                a = a / norm
        per_seed.append(0.5 * (1.0 + float(a @ ideal)))
    return float(np.mean(per_seed)), float(np.std(per_seed))


def depths(calib, lam, p, S, seeds, touch=False, prep=_prep_psi0):
    """Realized trip counts on the unbounded arm -> mean/max coherent depth."""
    ks = []
    for sd in range(seeds):
        rng = np.random.default_rng(1400 + sd)
        for _ in range(S):
            ks.append(fast_unbounded(rng, calib, lam, p=p, touch=touch, prep=prep)[1])
    return float(np.mean(ks)), int(np.min(ks)), int(np.max(ks))


def refresh_c_sweep(calib, p, S, seeds, C_lo, C_hi, touch, ideal=None,
                    prep=_prep_psi0):
    """Spec 8.1 step 4: at lam=1 sweep C over the window on the refresh arm and
    return the empirical best-fidelity C* (what S4 checks the window brackets)."""
    best_C, best_F = None, -1.0
    for C in range(max(1, C_lo), C_hi + 1):
        F, _ = bloch_fidelity(fast_refresh, calib, 1.0, C, p, S, seeds, touch=touch,
                              ideal=ideal, prep=prep)
        if F > best_F:
            best_F, best_C = F, C
    return best_C, best_F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="rus_rx_ibm",
                    choices=["rus_rx_ibm", "rus_lowp", "ipe"])
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
    elif args.bench == "ipe":
        import benchmarks.ipe as bench
    else:
        import benchmarks.rus_rx_ibm as bench
    p = bench.P_ANALYTIC

    # rus_lowp / ipe entangle the carried target each iteration (spec 5.1) -> per-2q
    # leakage accrues on it; rus_rx_ibm holds it idle (no 2q gate) -> zero leakage.
    touch = args.bench in ("rus_lowp", "ipe")
    # per-benchmark held state (default the H T H T H |0> magic state; ipe holds |+>)
    ideal = getattr(bench, "IDEAL_BLOCH", IDEAL_BLOCH)
    prep = getattr(bench, "prep_fast", _prep_psi0)

    if args.ibm:
        # real IBM Eagle r3 data for the carried qubit; leakage is the separate
        # published per-2q-gate estimate (spec 4.1), charged by qsim on each 2q gate.
        from sim.ibm_dataset import carried_calib, IBM_LEAK_PER_2Q
        from sim.qsim import load_calib
        leak = args.leak if args.leak is not None else IBM_LEAK_PER_2Q
        calib = load_calib(carried_calib(args.carry_qubit, p_leak=leak))
        src = f"IBM Eagle r3 (qubit {args.carry_qubit}, p_leak/2q={leak:g})"
    else:
        good = args.bench in ("rus_lowp", "ipe")
        calib = dict(GOOD_MEM) if good else dict(EVAL_CALIB)
        src = "held-memory device" if good else "leakage-dominated"
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
    Cstar, Fstar = refresh_c_sweep(calib, p, S, seeds, 1, max(1, C_max), touch,
                                   ideal=ideal, prep=prep)
    brackets = 1 <= Cstar <= C_max
    print(f"  refresh C-sweep (lam=1): best C* = {Cstar} (F={Fstar:.4f}); "
          f"window [1,{C_max}] {'brackets' if brackets else 'does NOT bracket'} "
          f"C*  [S4]")

    # analytical sampling-cost / resource metrics (metric families 2 & 3, spec 8.2.1)
    q_ref = (1.0 - p) ** C_refresh
    ecuts = q_ref / (1.0 - q_ref) if q_ref < 1.0 else float("inf")
    # KNIT sampling cost at its operating point: the variance-cap floor C_min if the
    # window is non-empty, else the tightest coherence-allowed cut C_max (where the
    # variance diverges -- that is *why* knit is inadmissible).
    knit_C = C_min if knit_ok else C_max
    v_knit = variance_V(knit_C, p) if knit_C >= 1 else float("inf")
    v_str = "inf (diverges)" if math.isinf(v_knit) else f"{v_knit:.1f}"
    v_note = "admissible" if knit_ok else "> C_max -> KNIT inadmissible"
    print(f"  sampling cost:  V(C={knit_C}) = {v_str}  (KNIT variance inflation; "
          f"C_min={C_min} {v_note});  E[#cuts]@C_refresh = {ecuts:.2f}")
    print(f"  bounded cap:    refresh coherent depth <= C_refresh*B = "
          f"{C_refresh*B} gate-layers (compile-time guarantee)")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "experiment.csv")
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["benchmark", "lam", "p", "C_knit", "C_refresh", "B_layers",
                "depth_per_iter", "min_iters", "mean_iters", "max_iters",
                "runtime_depth", "bounded_cap_layers", "fidelity_unbounded",
                "fidelity_knit", "fidelity_refresh", "std_knit", "std_refresh",
                "knit_admissible", "rmse_unbounded", "rmse_refresh",
                "delta_rmse_ref_minus_unb", "best_arm"])

    def _rmse(F, s):
        # delivered-state RMSE (accuracy, family 1): systematic infidelity (bias)
        # combined in quadrature with the statistical seed-std.
        return float(((1.0 - F) ** 2 + s * s) ** 0.5)

    tag = f"{args.bench}[p={p:g},Cr={C_refresh},B={B}]"
    tw = max(len(tag), 22)

    # --- table 1: runtime coherent depth + delivered fidelity ---
    print(f"\n{'bench[config]':<{tw}} | {'lam':>5} | "
          f"{'runtime coherent depth (unbounded)':>50} | "
          f"{'fidelity (mean, seed-std)':>42}")
    print(f"{'':<{tw}} | {'':>5} | {'depth/iter':>10} "
          f"{'min_iters':>9} {'mean_iters':>10} {'max_iters':>9} "
          f"{'runtime_depth':>14} | "
          f"{'unbounded':>11} {'knit(g4)*':>16} {'refresh(g1)':>15}")
    rows = []
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        mk, nk, xk = depths(calib, lam, p, S, seeds, touch=touch, prep=prep)
        fu, su = bloch_fidelity(fast_unbounded, calib, lam, C, p, S, seeds,
                                touch=touch, ideal=ideal, prep=prep)
        fd, sd = bloch_fidelity(fast_refresh, calib, lam, C_refresh, p, S, seeds,
                                touch=touch, ideal=ideal, prep=prep)
        if knit_ok:
            fq, sq = bloch_fidelity(fast_knit, calib, lam, C, p, S, seeds,
                                    touch=touch, clamp=True, ideal=ideal, prep=prep)
            knit_cell = f"{fq:8.4f}±{sq:.4f}"
            fq_csv, sq_csv = round(fq, 4), round(sq, 4)
        else:
            fq_csv = sq_csv = ""
            knit_cell = f"{'n/a':>15}"
        print(f"{tag:<{tw}} | {lam:5.2f} | {B:10d} "
              f"{nk:9d} {mk:10.2f} {xk:9d} {mk*B:14.0f} | "
              f"{fu:11.4f} {knit_cell:>16} {fd:8.4f}±{sd:.4f}")
        rows.append((lam, mk, nk, xk, fu, su, fd, sd, fq_csv, sq_csv))

    # --- table 2: tradeoff metrics -- accuracy (RMSE) + decision quality ---
    print(f"\n{'bench[config]':<{tw}} | {'lam':>5} | {'bounded_cap':>11} | "
          f"{'RMSE = delivered-state error (lower better)':>44}")
    print(f"{'':<{tw}} | {'':>5} | {'(layers)':>11} | "
          f"{'unbounded':>11} {'refresh(g1)':>13} {'d(ref-unb)':>11} {'best':>9}")
    for (lam, mk, nk, xk, fu, su, fd, sd, fq_csv, sq_csv) in rows:
        ru, rr = _rmse(fu, su), _rmse(fd, sd)
        d = rr - ru
        best = "refresh" if rr < ru else "unbounded"
        print(f"{tag:<{tw}} | {lam:5.2f} | {C_refresh*B:11d} | "
              f"{ru:11.4f} {rr:13.4f} {d:+11.4f} {best:>9}")
        w.writerow([args.bench, lam, p, (C if knit_ok else ""), C_refresh, B, B,
                    nk, round(mk, 3), xk, round(mk * B, 1), C_refresh * B,
                    round(fu, 4), fq_csv, round(fd, 4), sq_csv, round(sd, 4),
                    int(knit_ok), round(ru, 4), round(rr, 4), round(d, 4), best])
    fh.close()

    print("\nlegend (metrics beyond fidelity -- spec 8.2.1):")
    print("  lam            global noise scale (0 = noiseless, 1 = device, 2/4 = "
          "noisier)")
    print(f"  depth/iter     B = {B} gate-layers per loop body (purl.body_layers)")
    print("  *_iters        realized trip count = RUNTIME coherent depth (unbounded "
          "arm); max_iters is")
    print("                 the decohering TAIL; runtime_depth = mean_iters x B "
          "[resource, family 3]")
    print(f"  bounded_cap    C_refresh x B = {C_refresh*B} layers -- the compile-time "
          f"coherent-depth CAP the cut")
    print("                 guarantees, independent of any noise model [resource, "
          "family 3]")
    print("  fidelity       delivered-state Bloch fidelity F (higher=better), mean "
          "+/- seed-std. arms:")
    print("                 unbounded (control); refresh(g1) = deterministic gamma=1 "
          "cut (proven state,")
    print("                 zero variance); knit(g4) = gamma^2=16 quasi cut")
    print("  knit(g4)*      *reconstructed (|a|<=1 clip, spec 5.1); n/a where the "
          "KNIT variance window is")
    print("                 empty -- see V(C_min) in the header [sampling cost, "
          "family 2]")
    print("  RMSE           delivered-state error sqrt((1-F)^2 + seed-std^2): "
          "systematic infidelity (bias)")
    print("                 + statistical spread [accuracy, family 1]. Lower better.")
    print("  d(ref-unb)     RMSE(refresh) - RMSE(unbounded): <0 => cutting wins "
          "(fire refresh); >=0 =>")
    print("                 cutting does not pay (decline) [decision quality, "
          "family 4]")
    print("  best           arm with min measured RMSE (oracle); regret = "
          "RMSE(pass-chosen) - RMSE(best)")
    print("  header:        V(C_min) = KNIT variance inflation [family 2]; "
          "E[#cuts] = expected cuts/shot")
    print("                 [family 3]; refresh C-sweep vs window bracketing "
          "[family 4, S4]")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
