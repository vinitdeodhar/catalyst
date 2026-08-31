"""
eval/migrate.py -- migrate strategy evaluation (spec 13.7) on rus_data.

Migrate (gamma=1) SWAPs the carried state onto a fresh partner every C iterations,
clearing carrier-stuck leakage WITHOUT measuring/re-preparing/knowing the state, and
WITHOUT the quasi-probability variance knit pays (gamma^2=16). Three arms on rus_data
(the clean fixed-ideal V3|psi> metric, no per-shot reference):
  unbounded : the held loop, no cut.
  knit      : quasi-probability cut (gamma=4) every C -- the comparison arm.
  migrate   : swap-based carrier replacement (gamma=1) every C.

The central figure (spec 13.7): migrate MATCHES knit's fidelity (same cleared leakage)
at LOWER seed variance (no gamma^2 blow-up). Also a zero-leakage ablation: with no
leakage migrate's benefit reduces to age-capping and the gap to nominal is the
leakage-clearing component. rus_data has a thin p=5/8 tail (~2% cut fraction), so the
absolute gain is small -- reported whatever it is (spec 13.9 #5).

  PYTHONPATH=. python3 eval/migrate.py
"""

import os

import numpy as np

import benchmarks.rus_data as rd
from sim.qsim import QSim
from sim.knit_runtime import cut_and_reprepare

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
KNIT_C = 4
ANC = 1                 # ancilla wire; data ping-pongs between wires 0 and 2
PAIR_LEAK = 1e-4        # cheapest incident edge (spec 13.1) -- the SWAP is charged
                        # this, well below the body's global-median leakage


def _calib(leak):
    return dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                p_leak=leak)


def _attempt(sim, live):
    """One rus_data RUS attempt on the given live data wire; returns fail flag."""
    sim.force_zero(ANC)
    sim.ctrl_branch(live, ANC, rd.U0, rd.U1)   # the V3 gadget (data controls ancilla)
    m = sim.measure(ANC)
    sim.feedback(active=[live])
    return bool(m == 1)


def _read(sim, q, basis):
    if basis == "X":
        sim.h(q)
    elif basis == "Y":
        sim.sdg(q); sim.h(q)
    return 1 - 2 * sim.measure(q)


def _shot(rng, calib, lam, C, arm):
    """One shot; returns (weight, live_qubit, sim, n_cut). arm in {unbounded,knit,
    migrate}. The state ping-pongs 0<->2 for migrate; stays on 0 otherwise."""
    sim = QSim(3, calib=calib, lam=lam, rng=rng)
    live, partner = 0, 2
    sim.ry(live, rd.PREP_RY); sim.rz(live, rd.PREP_RZ)
    w, k, fail, nc = 1.0, 0, True, 0
    while fail and k < 500:
        fail = _attempt(sim, live)
        k += 1
        if fail and C is not None and k % C == 0:
            if arm == "knit":
                w *= cut_and_reprepare(sim, live, rng)
                nc += 1
            elif arm == "migrate":
                new = sim.migrate(live, partner, pair_leak=PAIR_LEAK)
                partner, live = live, new          # ping-pong
                nc += 1
    return w, live, sim, nc


def _fidelity(calib, lam, C, arm, seeds, S, clamp):
    Fs, cf = [], []
    for sd in range(seeds):
        comp, ncut = {}, 0
        for basis in ("X", "Y", "Z"):
            rng = np.random.default_rng(1300 + sd)
            num = den = 0.0
            for _ in range(S // 3):
                w, live, sim, nc = _shot(rng, calib, lam, C, arm)
                num += w * _read(sim, live, basis)
                den += 1.0
                if basis == "X":
                    ncut += nc
            comp[basis] = num / den
        a = np.array([comp["X"], comp["Y"], comp["Z"]])
        if clamp:
            n = float(np.linalg.norm(a))
            if n > 1.0:
                a = a / n
        Fs.append(0.5 * (1.0 + float(a @ rd.IDEAL_BLOCH)))
        cf.append(ncut / (S // 3))
    return np.array(Fs), float(np.mean(cf))


def rmse(F):
    return float(np.sqrt(np.mean((1.0 - F) ** 2)))


def main(seeds=8, S=6000):
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit("=== Migrate strategy on rus_data (spec 13.7), lam=1, C=%d ===" % KNIT_C)
    emit(f"seeds={seeds}, S={S}; migrate (gamma=1) vs knit (gamma^2=16) vs unbounded")

    for leak, label in ((1e-2, "elevated 1e-2"), (0.0, "zero (ablation)")):
        cal = _calib(leak)
        fu, _ = _fidelity(cal, 1.0, None, "unbounded", seeds, S, clamp=False)
        fk, cfk = _fidelity(cal, 1.0, KNIT_C, "knit", seeds, S, clamp=True)
        fm, cfm = _fidelity(cal, 1.0, KNIT_C, "migrate", seeds, S, clamp=False)
        # migrate has NO variance floor (no C_min), so it can cut aggressively at a
        # small C where knit is inadmissible -- clearing more leakage per shot.
        fm2, cfm2 = _fidelity(cal, 1.0, 2, "migrate", seeds, S, clamp=False)
        emit(f"\n  leakage = {label}")
        emit(f"    {'arm':<16} | {'F (mean+-std)':>18} | {'RMSE':>7} | {'cutfrac':>7}")
        emit(f"    {'unbounded':<16} | {fu.mean():.4f}±{fu.std(ddof=1):.4f} | "
             f"{rmse(fu):.4f} | {'--':>7}")
        emit(f"    {'knit(g^2=16,C=4)':<16} | {fk.mean():.4f}±{fk.std(ddof=1):.4f} | "
             f"{rmse(fk):.4f} | {cfk:7.3f}")
        emit(f"    {'migrate(g=1,C=4)':<16} | {fm.mean():.4f}±{fm.std(ddof=1):.4f} | "
             f"{rmse(fm):.4f} | {cfm:7.3f}")
        emit(f"    {'migrate(g=1,C=2)':<16} | {fm2.mean():.4f}±{fm2.std(ddof=1):.4f} | "
             f"{rmse(fm2):.4f} | {cfm2:7.3f}   (C=2 < knit C_min: knit inadmissible)")
        vr = ((fm.std(ddof=1) / fk.std(ddof=1)) ** 2
              if fk.std(ddof=1) > 0 else float("nan"))
        emit(f"    migrate/knit seed-variance ratio @C=4 = {vr:.3f} "
             f"(migrate is gamma=1: no quasi-probability weights)")

    emit("\n  (central figure: migrate matches knit's cleared-leakage fidelity at "
         "lower variance; the zero-leakage rows isolate age-capping from clearing.)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "migrate.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
