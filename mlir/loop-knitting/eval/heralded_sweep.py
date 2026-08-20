"""
heralded_sweep.py -- does a low-p heralded loop (heavy trip-count tail) let
knitting beat unbounded/discard? Sweeps the crossover for benchmarks/heralded at
p = P_HERALD, and reports the pass cut-period window for a realistic vs a
long-coherence (good-memory) device.

Run:  PYTHONPATH=. python3 eval/heralded_sweep.py
"""

import math
import numpy as np

from sim.fast_target import fast_unbounded, fast_truncated, fast_knit
import benchmarks.heralded as herald

# A good-memory device (long coherence) so the window can even open at low p;
# leakage is the reset-clearable error the transform targets.
GOOD_MEM = dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                T_leak=4e-3)
REALISTIC = dict(GOOD_MEM, T1=200e-6, T2=200e-6, T_leak=150e-6)


def window(p, calib, gamma2=16.0, f=0.05):
    C_min = math.ceil(math.log(gamma2) / math.log(1.0 / (1.0 - p)))
    B_plus_tau = calib["readout"] * 3 + calib["tau"]
    C_max = int(f * calib["T2"] / B_plus_tau)
    return C_min, C_max


def arms(p, calib, C, lam, S, seeds):
    """(E_unb, E_disc, E_knit, cut_rate, discard_rate) mean over seeds."""
    Eu, Ed, Ek, cr, dr = [], [], [], [], []
    for sd in range(seeds):
        r = np.random.default_rng(500 + sd)
        zu = [fast_unbounded(r, calib, lam, p=p)[0] for _ in range(S)]
        r = np.random.default_rng(600 + sd)
        kt = [fast_truncated(r, calib, lam, C, p=p) for _ in range(S)]
        kept = [z for z, _, tr in kt if not tr]
        ntr = sum(1 for _, _, tr in kt if tr)
        r = np.random.default_rng(700 + sd)
        kn = [fast_knit(r, calib, lam, C, p=p) for _ in range(S)]
        zk = [e for e, _, _, _ in kn]
        ncut = sum(1 for _, _, nc, _ in kn if nc > 0)
        Eu.append(abs(np.mean(zu) - herald.Z_IDEAL))
        Ed.append(abs(np.mean(kept) - herald.Z_IDEAL) if kept else np.nan)
        Ek.append(abs(np.mean(zk) - herald.Z_IDEAL))
        cr.append(ncut / S)
        dr.append(ntr / S)
    return (np.nanmean(Eu), np.nanmean(Ed), np.nanmean(Ek),
            np.nanmean(cr), np.nanmean(dr), np.nanstd(Ek))


def main():
    p = herald.P_HERALD
    mean_k = 1.0 / p
    print(f"heralded benchmark: p = {p}  (mean trip count 1/p = {mean_k:.0f})")

    cmin_r, cmax_r = window(p, REALISTIC)
    cmin_g, cmax_g = window(p, GOOD_MEM)
    print(f"\npass window (variance floor C_min, coherence ceiling C_max):")
    print(f"  realistic device (T2=200us):  [C_min={cmin_r}, C_max={cmax_r}]  "
          f"-> {'EMPTY (pass refuses to fire)' if cmin_r > cmax_r else 'ok'}")
    print(f"  good-memory device (T2=8ms):  [C_min={cmin_g}, C_max={cmax_g}]  "
          f"-> {'EMPTY' if cmin_g > cmax_g else 'ok'}")
    print(f"  C_min / mean_k = {cmin_g / mean_k:.2f}  "
          f"(prediction ~2.77 for small p)")
    print(f"  fraction of shots reaching a cut  P(k>C_min) = "
          f"{(1 - p) ** cmin_g:.3f}")

    S, seeds = 5000, 6
    C = cmin_g  # the only feasible C on the good-memory device
    print(f"\ncrossover on the good-memory device, C = C_min = {C}, "
          f"S={S}, seeds={seeds}:")
    print(f"{'lam':>5} | {'E_unb':>12} | {'E_disc':>12} | {'E_knit':>16} | "
          f"cut% disc%")
    for lam in (0.0, 0.5, 1.0, 2.0):
        eu, ed, ek, cr, dr, sk = arms(p, GOOD_MEM, C, lam, S, seeds)
        best = min(("unb", eu), ("disc", ed), ("knit", ek), key=lambda t: t[1])[0]
        print(f"{lam:5.2f} | {eu:12.4f} | {ed:12.4f} | {ek:8.4f}±{sk:.4f} | "
              f"{cr*100:4.1f} {dr*100:4.1f}   best={best}")

    # tight-bound demo: force C below C_min -> divergent knit variance
    print(f"\ntight bound C=5 (< C_min={C}) at lam=1 -> divergent variance:")
    eu, ed, ek, cr, dr, sk = arms(p, GOOD_MEM, 5, 1.0, S, seeds)
    print(f"  E_knit = {ek:.4f} ± {sk:.4f}   (cut% {cr*100:.0f})  vs "
          f"E_disc = {ed:.4f}, E_unb = {eu:.4f}")


if __name__ == "__main__":
    main()
