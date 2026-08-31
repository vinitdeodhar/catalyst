"""
eval/rus_data.py -- evaluation for the rus_data benchmark (spec 6.5).

Paetznick-Svore V3 RUS applied to program data (FLAGGED best-effort reconstruction,
see benchmarks/rus_data.py). Knit-only benchmark (carried state is non-Clifford ->
unknown -> refresh unsound). The delivered ideal is the FIXED V3|psi> (failure is
identity), so fidelity is plain 3-basis tomography against one target -- no per-shot
reference. Arms: unbounded and knit (cut every C failing iterations, quasi-probability
reconstruction, |a|<=1 clamp). Reports fidelity, RMSE, cut-fraction, E[#cuts]; a
leakage sweep tests whether the knit gain SCALES with leakage (spec 6.5, not that it
is large -- the p=5/8 tail is thin, ~few % of shots reach a cut).

  PYTHONPATH=. python3 eval/rus_data.py
"""

import os

import numpy as np

import benchmarks.rus_data as rd
from sim.qsim import QSim
from sim.knit_runtime import cut_and_reprepare

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
KNIT_C = 4          # in-window at p=5/8 (C_min~3 on the deployed calib)


def _calib(leak):
    return dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                p_leak=leak)


def _read(sim, q, basis):
    if basis == "X":
        sim.h(q)
    elif basis == "Y":
        sim.sdg(q); sim.h(q)
    return 1 - 2 * sim.measure(q)


def _shot(rng, calib, lam, C):
    """One RUS shot; returns (weight, {basis: read}, cut_count). C=None -> unbounded.
    The 3 basis reads are done on independent shots by the caller (one basis here)."""
    sim = QSim(rd.N_WIRES, calib=calib, lam=lam, rng=rng)
    rd.prepare_input(sim)
    w = 1.0
    k, fail, ncut = 0, True, 0
    while fail and k < 500:
        fail = rd.attempt(sim)
        k += 1
        if C is not None and fail and k % C == 0:
            w *= cut_and_reprepare(sim, rd.DATA, rng)   # knit cut on the held data
            ncut += 1
    return w, sim, ncut


def _fidelity(calib, lam, C, seeds, S, clamp):
    """Weighted 3-basis tomography fidelity vs the fixed ideal V3|psi>. Returns
    (per-seed F array, mean cut-fraction, mean E[#cuts])."""
    Fs, cfrac, ecuts = [], [], []
    for sd in range(seeds):
        comp, ncut_tot, ncut_shots = {}, 0, 0
        for basis in ("X", "Y", "Z"):
            rng = np.random.default_rng(1300 + sd)
            num, den = 0.0, 0.0
            for _ in range(S // 3):
                w, sim, nc = _shot(rng, calib, lam, C)
                num += w * _read(sim, rd.DATA, basis)
                den += 1.0
                if basis == "X":                 # count cuts once per (shot) triple
                    ncut_tot += nc
                    ncut_shots += 1 if nc > 0 else 0
            comp[basis] = num / den
        a = np.array([comp["X"], comp["Y"], comp["Z"]])
        if clamp:
            n = float(np.linalg.norm(a))
            if n > 1.0:
                a = a / n
        Fs.append(0.5 * (1.0 + float(a @ rd.IDEAL_BLOCH)))
        cfrac.append(ncut_shots / (S // 3))
        ecuts.append(ncut_tot / (S // 3))
    return np.array(Fs), float(np.mean(cfrac)), float(np.mean(ecuts))


def rmse(F):
    return float(np.sqrt(np.mean((1.0 - F) ** 2)))


def main(seeds=8, S=6000):
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit("=== rus_data (spec 6.5): Paetznick-Svore V3 RUS on program data ===")
    emit("*** FLAGGED best-effort reconstruction: V3 channel exact, gate circuit "
         "unverified vs paper ***")
    emit(f"p={rd.P_ANALYTIC}, knit C={KNIT_C}, seeds={seeds}, S={S}; "
         f"ideal V3|psi> Bloch={rd.IDEAL_BLOCH.round(3)}")

    # noise sweep at nominal leakage (1e-3): unbounded vs knit
    cal = _calib(1e-3)
    emit(f"\n  {'lam':>5} | {'F_unbounded':>12} | {'F_knit':>14} | {'cut frac':>8} "
         f"| {'E[#cuts]':>8}")
    for lam in LAMS:
        fu, _, _ = _fidelity(cal, lam, None, seeds, S, clamp=False)
        fk, cf, ec = _fidelity(cal, lam, KNIT_C, seeds, S, clamp=True)
        emit(f"  {lam:5.2f} | {fu.mean():.4f}±{fu.std(ddof=1):.4f} | "
             f"{fk.mean():.4f}±{fk.std(ddof=1):.4f} | {cf:8.3f} | {ec:8.3f}")

    # leakage sweep @ lam=1: does the knit gain SCALE with leakage? (spec 6.5)
    emit(f"\n  leakage sweep @ lam=1 (knit gain over unbounded):")
    emit(f"  {'leak/2q':>8} | {'F_unbounded':>12} | {'F_knit':>14} | {'gain±SEM':>16}")
    for leak in (1e-4, 1e-3, 1e-2):
        c = _calib(leak)
        fu, _, _ = _fidelity(c, 1.0, None, seeds, S, clamp=False)
        fk, _, _ = _fidelity(c, 1.0, KNIT_C, seeds, S, clamp=True)
        g = fk.mean() - fu.mean()
        sem = float(np.hypot(fu.std(ddof=1), fk.std(ddof=1)) / np.sqrt(seeds))
        emit(f"  {leak:8.0e} | {fu.mean():.4f}±{fu.std(ddof=1):.4f} | "
             f"{fk.mean():.4f}±{fk.std(ddof=1):.4f} | {g:+.4f}±{sem:.4f}")
    emit("  (claim under test: gain SCALES with leakage, not that it is large; the "
         "p=5/8 tail is thin so the cut-touched pool is small.)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "rus_data.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
