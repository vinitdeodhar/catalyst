"""
eval/qwalk.py -- migrate evaluation on the fat-tailed qwalk benchmark (spec 13.7).

qwalk is built to give migrate a REAL gain: a random-walk hitting-time herald (fat
tail -> ~50% of shots cross a C=1 cut, vs ~2% for the thin p=5/8 RUS benchmarks) and
a 2q-heavy (n2q=6) net-identity non-Clifford body (fast leakage accumulation, unknown
state). Three arms against the FIXED held |psi> (net-identity -> no per-shot reference):
  unbounded : the held loop.
  knit      : quasi-probability cut (gamma^2=16) -- but its variance floor forbids
              small C, so it can only cut at C >= C_min.
  migrate   : swap-based (gamma=1), NO variance floor -> cuts at C=1, clearing leakage
              on the whole fat tail. The SWAP is charged the cheap pair edge (spec 13.1).

  PYTHONPATH=. python3 eval/qwalk.py
"""

import os

import numpy as np

import benchmarks.qwalk as qw
from sim.qsim import QSim
from sim.knit_runtime import cut_and_reprepare

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
ANC = 1                 # ancilla; data ping-pongs between wires 0 and 2
PAIR_LEAK = 1e-4        # cheapest incident edge (spec 13.1)


def _calib(leak):
    return dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                p_leak=leak)


def _touch(sim, live):
    """qwalk's net-identity 2q-heavy coupling on the given live data wire."""
    sim.force_zero(ANC)
    sim.h(ANC)
    for _ in range(qw.N_BLOCKS):
        sim.cnot(ANC, live)
        sim.t(ANC)
        sim.cnot(ANC, live)
    sim.measure(ANC)


def _read(sim, q, basis):
    if basis == "X":
        sim.h(q)
    elif basis == "Y":
        sim.sdg(q); sim.h(q)
    return 1 - 2 * sim.measure(q)


def _shot(rng, calib, lam, C, arm):
    sim = QSim(3, calib=calib, lam=lam, rng=rng)
    live, partner = 0, 2
    sim.ry(live, qw.PREP_RY); sim.rz(live, qw.PREP_RZ)
    pos, k, w, nc = 1, 0, 1.0, 0
    while pos != 0 and k < 500:
        _touch(sim, live)
        pos = qw.walk_step(rng, pos)
        k += 1
        if pos != 0 and C is not None and k % C == 0:
            if arm == "knit":
                w *= cut_and_reprepare(sim, live, rng)
                nc += 1
            elif arm == "migrate":
                new = sim.migrate(live, partner, pair_leak=PAIR_LEAK)
                partner, live = live, new
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
        Fs.append(0.5 * (1.0 + float(a @ qw.IDEAL_BLOCH)))
        cf.append(ncut / (S // 3))
    return np.array(Fs), float(np.mean(cf))


def rmse(F):
    return float(np.sqrt(np.mean((1.0 - F) ** 2)))


def main(seeds=8, S=6000):
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit("=== Migrate on the fat-tailed qwalk benchmark (spec 13.7), lam=1 ===")
    emit(f"seeds={seeds}, S={S}; random-walk hitting-time herald, n2q=6, "
         f"ideal held Bloch={qw.IDEAL_BLOCH.round(3)}")

    for leak, label in ((1e-2, "elevated 1e-2"), (0.0, "zero (ablation)")):
        cal = _calib(leak)
        fu, _ = _fidelity(cal, 1.0, None, "unbounded", seeds, S, clamp=False)
        # knit at its variance-floor period C=4; migrate at C=1 (no floor).
        fk, cfk = _fidelity(cal, 1.0, 4, "knit", seeds, S, clamp=True)
        fm, cfm = _fidelity(cal, 1.0, 1, "migrate", seeds, S, clamp=False)
        emit(f"\n  leakage = {label}")
        emit(f"    {'arm':<18} | {'F (mean+-std)':>18} | {'RMSE':>7} | {'cutfrac':>7}")
        emit(f"    {'unbounded':<18} | {fu.mean():.4f}±{fu.std(ddof=1):.4f} | "
             f"{rmse(fu):.4f} | {'--':>7}")
        emit(f"    {'knit(g^2=16,C=4)':<18} | {fk.mean():.4f}±{fk.std(ddof=1):.4f} | "
             f"{rmse(fk):.4f} | {cfk:7.3f}")
        emit(f"    {'migrate(g=1,C=1)':<18} | {fm.mean():.4f}±{fm.std(ddof=1):.4f} | "
             f"{rmse(fm):.4f} | {cfm:7.3f}")
        g = fm.mean() - fu.mean()
        sem = float(np.hypot(fu.std(ddof=1), fm.std(ddof=1)) / np.sqrt(seeds))
        emit(f"    migrate gain over unbounded = {g:+.4f} ± {sem:.4f}  "
             f"({g/sem:+.1f} sigma)")

    emit("\n  (fat tail -> high cut fraction -> migrate clears leakage on most shots "
         "at gamma=1; knit's variance floor forbids the small C that would match it.)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "qwalk.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
