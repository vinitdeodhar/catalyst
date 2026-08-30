"""
eval/ipe_project.py -- evaluation for the ipe_project benchmark (spec 6.3).

ipe_project needs its own runner (not the fixed-state fast path): the carried wire
PROJECTS as outcomes arrive, so delivered fidelity is measured against a PER-SHOT
reference (the posterior winner, benchmarks.ipe_project.delivered_fidelity). Two
configs: `ipe_project` (faithful, knit-inadmissible negative control -> only the
unbounded arm) and `ipe_project_fast` (knit-admissible). Refresh is never applicable
(unknown carried state); the falsification arm forces an INVALID refresh to show it
destroys the computation.

  PYTHONPATH=. python3 eval/ipe_project.py
"""

import os

import numpy as np

import benchmarks.ipe_project as ip

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
KNIT_C = 6                      # in-window cut period for the fast config (C_min~5)
FAST_CAP = 10                  # bounded IPE precision -> bounds the knit cut cascade


def _calib(leak):
    # good-memory device (as experiment.py GOOD_MEM) with the given per-2q leakage.
    return dict(gate_1q=30e-9, gate_2q=60e-9, readout=1e-6, tau=1e-6,
                T1=8e-3, T2=8e-3, p1=2e-4, p2=1e-3, p_ro=2e-3, p_meas=1e-3,
                p_leak=leak)


def _fid(runner_shots):
    return ip.delivered_fidelity(runner_shots)


def _arm(cfg, calib, lam, C, seeds, S, cap):
    """Mean +- seed-std delivered fidelity for one arm (C=None -> unbounded)."""
    Fs = []
    for sd in range(seeds):
        rng = np.random.default_rng(700 + sd)
        if C is None:
            shots = [(w, z, 1.0) for w, z, _ in
                     (ip.run(rng, cfg, calib=calib, lam=lam, max_rounds=cap)[1:]
                      for _ in range(S))]
        else:
            shots = [(w, z, wt) for _, w, z, wt in
                     (ip.run(rng, cfg, calib=calib, lam=lam, C=C, max_rounds=cap)
                      for _ in range(S))]
        Fs.append(_fid(shots))
    return float(np.mean(Fs)), float(np.std(Fs))


def main(seeds=8, S=3000):
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit("=== ipe_project (spec 6.3): phase estimation as eigenstate projection ===")
    emit(f"U=Rz(2pi/7), input cos(pi/8)|0>+sin(pi/8)|1>; per-shot reference = winner")

    # faithful: knit inadmissible (empty window) -> negative control, unbounded only.
    emit("\n[faithful] p~0.12, knit window EMPTY -> strategy NONE (unbounded arm only)")
    emit(f"  {'lam':>5} | {'F_unbounded':>12}")
    for lam in LAMS:
        fu, su = _arm(ip.FAITHFUL, _calib(1e-3), lam, None, seeds, S, 400)
        emit(f"  {lam:5.2f} | {fu:6.4f}±{su:.4f}")

    # fast: knit admissible. Leakage ablation at nominal noise (lam=1): the reset in
    # each knit cut clears leakage, so the knit gain over unbounded grows with it.
    emit(f"\n[fast] p~0.4, knit C={KNIT_C}, round cap={FAST_CAP}. Leakage ablation @ lam=1:")
    emit(f"  {'leak/2q':>8} | {'F_unbounded':>12} | {'F_knit':>14} | {'gain':>14}")
    for leak in (0.0, 1e-3, 1e-2):
        cal = _calib(leak)
        fu, su = _arm(ip.FAST, cal, 1.0, None, seeds, S, FAST_CAP)
        fk, sk = _arm(ip.FAST, cal, 1.0, KNIT_C, seeds, S, FAST_CAP)
        g = fk - fu
        ge = float(np.hypot(su, sk))
        tag = "positive" if g > ge else "~0 (within noise)"
        emit(f"  {leak:8.0e} | {fu:6.4f}±{su:.4f} | {fk:6.4f}±{sk:.4f} | "
             f"{g:+.4f}±{ge:.4f} {tag}")
    emit("  (knit is an UNBIASED estimator here -- refresh is unsound; its variance is")
    emit("   high because ipe_project's posterior stop couples to the cuts.)")

    # falsification arm: an INVALID forced refresh collapses fidelity toward the
    # input state's overlap with the winning eigenstate (~0.75).
    emit("\n[falsification] faithful @ lam=1: a DELIBERATELY INVALID forced refresh")
    fv, _ = _arm(ip.FAITHFUL, _calib(1e-3), 1.0, None, seeds, S, 400)
    Ff = []
    for sd in range(seeds):
        rng = np.random.default_rng(900 + sd)
        shots = [(w, z, 1.0) for w, z, _ in
                 (ip.run_forced_refresh(rng, ip.FAITHFUL, calib=_calib(1e-3),
                                        lam=1.0, C=2)[1:] for _ in range(S))]
        Ff.append(_fid(shots))
    emit(f"  F_valid(unbounded)={fv:.4f}   F_forced_refresh={np.mean(Ff):.4f}"
         f"±{np.std(Ff):.4f}  (collapses toward input overlap ~0.75)")
    emit("  -> refresh on an unknown carried state changes the computation (the no-go).")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ipe_project.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
