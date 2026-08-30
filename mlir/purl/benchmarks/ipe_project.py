"""
ipe_project -- phase estimation as eigenstate PROJECTION (spec 6.3).

The benchmark where REFRESH is unsound and KNIT is the only valid cut. Unlike
`ipe` (which holds a KNOWN eigenstate -> provable -> refresh), here the carried
data wire starts in a SUPERPOSITION of the two eigenstates of U = Rz(THETA), and
each round's ancilla measurement partially PROJECTS it toward one eigenstate. The
carried state at any iteration boundary depends on the outcome history, so there
is no fixed state to re-prepare (known_state = unknown, asserted in the pass); a
refresh would erase the projection that IS the computation. Knit's gamma=4
identity decomposition is still unbiased for ANY carried state, so knit is valid.

Physics (single ancilla, m=1 per round):
  * U = Rz(THETA), eigenstates |0>,|1> (Z-eigenstates), eigenphases -/+ THETA/2.
  * data prepared in cos(ALPHA)|0> + sin(ALPHA)|1> = Ry(2 ALPHA)|0>, ALPHA away
    from an eigenstate (prior populations = cos^2 ALPHA, sin^2 ALPHA).
  * round: ancilla |0> -H-> |+>; controlled-Rz(THETA) kickback (non-Clifford);
    an S feedback on the ancilla breaks the +/-THETA/2 symmetry; H; measure. The
    ancilla outcome b is a weak Z-measurement of the data -- its backaction nudges
    the data toward |0> or |1>, and a classical Bayesian posterior over {0,1} is
    updated from the same outcome. Continue until the posterior is confident.

Delivered-fidelity (the subtle point, spec 6.3): the reference is chosen PER SHOT
-- the eigenstate the shot's own posterior selects (the winner). Because the
eigenstates are Z-eigenstates, fidelity of the delivered qubit to |winner> is just
(1 + (-1)^winner <Z>)/2, so a single Z read per shot suffices (no X/Y tomography).
At lam=0 the data collapses to the winner, so this is 1.0 within statistics. For
the knit arm the quasi-probability weight multiplies the conditioned estimate and
the reconstructed <Z> is clamped to [-1,1] (as the standard knit tomography does).

Keep in lockstep with mlir/test/Quantum/Purl/ipe_project_{unknown,knit}.mlir.
"""

import math

import numpy as np

from sim.qsim import QSim
from sim.knit_runtime import cut_and_reprepare

N_WIRES = 2
DATA = 0                    # carried, never measured in the loop
ANCILLA = 1                # phase-estimation ancilla (measured each round)
THETA = 2.0 * math.pi / 7.0   # U = Rz(THETA); no special structure (non-Clifford)
ALPHA = math.pi / 8.0         # input away from an eigenstate
FEEDBACK = math.pi / 2.0      # ancilla feedback (S gate) breaking the +/- symmetry

PRIOR = (math.cos(ALPHA) ** 2, math.sin(ALPHA) ** 2)


class Config:
    """A benchmark configuration. `thresh` = posterior-confidence stop; `B_layers`
    and `f` (coherence-budget fraction) are pass inputs reported for the window."""

    def __init__(self, name, thresh, B_layers, f, p_nominal):
        self.name = name
        self.thresh = thresh
        self.B_layers = B_layers
        self.f = f
        self.p_nominal = p_nominal  # effective per-round stop rate ~ 1/mean_rounds


# faithful: confident stop -> mean ~10 rounds (p~0.1, near ipe's 0.12), knit window
# empty. High threshold => the data is (near) fully collapsed at exit, so the lam=0
# per-shot-reference fidelity -> 1 (0.997 at thresh 0.995).
FAITHFUL = Config("ipe_project", thresh=0.995, B_layers=12, f=0.05, p_nominal=0.12)
# fast: coarse stop just above the prior confidence -> mean ~2.5 rounds (p~0.4),
# trimmed body, generous f -> the knit window opens. (Its lam=0 fidelity is only
# ~thresh: the fast arm exists to exhibit the KNIT rewrite/gain, not full collapse.)
FAST = Config("ipe_project_fast", thresh=0.87, B_layers=4, f=0.15, p_nominal=0.45)


def _likelihood(b, i):
    """P(ancilla bit = b | data eigenstate i) for one m=1 round (see module doc):
    P(b=0|i) = cos^2(chi_i/2 + FEEDBACK/2), chi_0=-THETA/2, chi_1=+THETA/2."""
    chi = -THETA / 2.0 if i == 0 else THETA / 2.0
    p0 = math.cos(chi / 2.0 + FEEDBACK / 2.0) ** 2
    return p0 if b == 0 else 1.0 - p0


def prepare_input(sim):
    """cos(ALPHA)|0> + sin(ALPHA)|1> on the carried data wire (a superposition of
    the U-eigenstates -- NOT an eigenstate, so the carried state is unknown)."""
    sim.ry(DATA, 2.0 * ALPHA)


def _round(sim):
    """One projecting IPE round; returns the ancilla bit. The controlled-Rz is the
    non-Clifford controlled-U^1; its measurement backaction projects the data."""
    a = ANCILLA
    sim.h(a)
    sim.crz(a, DATA, THETA)     # controlled-U (non-Clifford)
    sim.s(a)                    # feedback -> asymmetric which-eigenstate likelihood
    sim.h(a)
    b = sim.measure(a)
    sim.feedback(active=[])
    sim.force_zero(a)           # fresh ancilla next round
    return b


def _run_core(rng, cfg, calib, lam, C, max_rounds):
    """Drive the projecting IPE loop to posterior confidence. Returns
    (sim, rounds, posterior, weight). C=None -> unbounded; C=int -> a knit cut on
    the data every C rounds (weight accumulates the gamma=4 term factor)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    post = list(PRIOR)
    w = 1.0
    r = 0
    while max(post) < cfg.thresh and r < max_rounds:
        b = _round(sim)
        l0, l1 = post[0] * _likelihood(b, 0), post[1] * _likelihood(b, 1)
        tot = l0 + l1
        if tot > 0:
            post = [l0 / tot, l1 / tot]
        r += 1
        if C is not None and max(post) < cfg.thresh and r % C == 0:
            w *= cut_and_reprepare(sim, DATA, rng)   # knit cut (identity in expectation)
    return sim, r, post, w


def run(rng, cfg=FAITHFUL, calib=None, lam=1.0, C=None, max_rounds=400):
    """Run one shot. Returns (rounds, winner, z_sample, weight); `winner` is the
    posterior's selected eigenstate (per-shot reference), `z_sample` the data's Z
    read at exit."""
    sim, r, post, w = _run_core(rng, cfg, calib, lam, C, max_rounds)
    winner = 0 if post[0] >= post[1] else 1
    z = 1 - 2 * sim.measure(DATA)                    # data Z read (per-shot reference)
    return r, winner, z, w


def run_debug(rng, cfg=FAITHFUL, calib=None, lam=0.0, max_rounds=400):
    """Like run() but returns (winner, data_p1) with the data's P(|1>) measured
    BEFORE the (destructive) read -- used to check the posterior winner matches the
    collapsed data eigenstate on noiseless shots (spec 6.3 validation)."""
    sim, _, post, _ = _run_core(rng, cfg, calib, lam, None, max_rounds)
    winner = 0 if post[0] >= post[1] else 1
    return winner, sim._prob_one(DATA)


def run_forced_refresh(rng, cfg=FAITHFUL, calib=None, lam=1.0, C=2, max_rounds=400):
    """DELIBERATELY INVALID (spec 6.3 falsification arm): every C rounds, measure +
    reset the data and re-prepare the INITIAL superposition -- a refresh the pass
    correctly REFUSES here (known_state = unknown). It erases the projection, so the
    delivered state at exit is (re-)projected only over the last partial segment and
    its fidelity to the per-shot reference collapses toward the input's overlap with
    the winning eigenstate. Returns (rounds, winner, z_sample, weight=1)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    post = list(PRIOR)
    r = 0
    while max(post) < cfg.thresh and r < max_rounds:
        b = _round(sim)
        l0, l1 = post[0] * _likelihood(b, 0), post[1] * _likelihood(b, 1)
        tot = l0 + l1
        if tot > 0:
            post = [l0 / tot, l1 / tot]
        r += 1
        if r % C == 0:                 # mechanical refresh every C rounds (unconditional)
            sim.measure(DATA)          # end the segment (invalid: carried wire read)
            sim.force_zero(DATA)       # reset
            prepare_input(sim)         # re-prepare the INPUT -> erases the projection
    winner = 0 if post[0] >= post[1] else 1
    z = 1 - 2 * sim.measure(DATA)
    return r, winner, z, 1.0


def delivered_fidelity(shots):
    """Per-shot-reference delivered fidelity from (winner, z, weight) samples.

    Each shot's fidelity to its own reference |winner> is (1 + (-1)^winner z)/2
    (the eigenstates are Z-eigenstates, so a single Z read suffices). The knit
    quasi-probability weight multiplies this CONDITIONED estimate: because
    E_cut[w h(record, z)] = E_true[h(record, z)] for any function h of the shot's
    record and read, the unbiased estimator is the plain mean of
    weight * (1 + (-1)^winner z)/2 over all shots. The reconstructed value is
    clamped to [0,1] (the |a|<=1 Bloch clamp). For the unbounded arm weight = 1."""
    vals = [w * 0.5 * (1.0 + (1 if wn == 0 else -1) * z) for wn, z, w in shots]
    F = float(np.mean(vals))
    return max(0.0, min(1.0, F))


if __name__ == "__main__":
    for cfg in (FAITHFUL, FAST):
        rng = np.random.default_rng(11)
        N = 4000
        rounds, shots = [], []
        for _ in range(N):
            r, wn, z, w = run(rng, cfg, lam=0.0)
            rounds.append(r)
            shots.append((wn, z, w))
        mean_r = sum(rounds) / len(rounds)
        F = delivered_fidelity(shots)
        win1 = sum(1 for wn, _, _ in shots if wn == 1) / N
        p_eff = 1.0 / mean_r if mean_r > 0 else float("nan")
        print(f"{cfg.name}: mean_rounds={mean_r:.2f} (p_eff={p_eff:.3f}, "
              f"target {cfg.p_nominal})  P(winner=1)={win1:.3f}  "
              f"lam=0 delivered F={F:.4f}")
