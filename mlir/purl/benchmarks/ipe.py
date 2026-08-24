"""
ipe -- Adaptive / Bayesian Iterative Phase Estimation as a carry-type benchmark.

A genuinely different algorithm from RUS (not a retry-until-success coin): the
carried wire holds the EIGENSTATE |psi> of a unitary U while the loop iteratively
extracts phase bits and continues until the posterior is confident (a
measurement-conditioned, runtime-variable number of rounds). Each round applies a
controlled-U^(2^k) between a fresh ancilla and the held eigenstate: by phase
kickback the eigenphase is deposited on the ANCILLA and the eigenstate is left
UNCHANGED (U^c|psi> = e^{i c phi}|psi>, a global phase that factors onto the
ancilla). So the held wire's per-iteration action is exactly the identity ->
provably known -> Purl can REFRESH it.

Why it exercises Purl differently from rus_*:
  * the carried wire is an *eigenstate* (a held reference), not a magic state to be
    delivered by retries;
  * the loop is information-accumulating (phase bits) with an *adaptive* stopping
    rule, not identical retries;
  * the controlled-U is the 2q "touch" that charges per-2q-gate leakage on the
    held wire (spec 4.1) -- here it is the algorithm's own gate, not a bolted-on
    maintenance op.

Held eigenstate: |psi> = |+> (eigenstate of U = Rx(.)); ideal Bloch (1,0,0),
<X>_ideal = 1. |+> is sensitive to T2 dephasing, so holding it through many rounds
decoheres it and refresh (re-prepare |+>) restores it.
"""

import numpy as np

from sim.qsim import QSim

N_WIRES = 2
TARGET = 0            # the held eigenstate |psi> = |+>
ANCILLA = 1           # the phase-estimation ancilla (measured each round)
# per-round stopping probability of the adaptive rule; mean rounds = 1/P (~8),
# a moderately heavy tail -- the regime where holding the eigenstate decoheres.
P_ANALYTIC = 0.12
IDEAL_BLOCH = np.array([1.0, 0.0, 0.0])  # |+>
Z_IDEAL = 0.0        # <Z> of |+>


def prep_fast(sim):
    """Prepare the held eigenstate |+> on the carried wire (1-qubit fast model)."""
    sim.h(0)


def prepare_input(sim):
    """Prepare the held eigenstate |psi> = |+> of U on the carried target."""
    sim.h(TARGET)


def attempt(sim):
    """One adaptive-IPE round. The controlled-U^(2^k) is net-identity on the held
    eigenstate (phase kickback onto the ancilla) but a physical 2q gate on it, so
    per-2q-gate leakage accrues; the ancilla is measured to extract a phase bit and
    the target idles through the readout. Returns the not-yet-confident flag."""
    a = ANCILLA
    # controlled-U^(2^k) phase kickback: identity on the eigenstate, 2q gate on it.
    sim.touch_2q(TARGET)
    # phase-estimation ancilla round: |+>, (feedforward Rz on the ancilla -- not the
    # held target -- would go here), H, measure the phase bit.
    sim.h(a)
    sim.h(a)
    sim.measure(a)
    sim.feedback(active=[])
    sim.force_zero(a)  # fresh ancilla next round
    # adaptive stopping: continue until the posterior variance < target. Modeled as
    # a per-round confidence probability P_ANALYTIC (mean 1/P rounds to precision).
    return bool(sim.rng.random() < (1.0 - P_ANALYTIC))


def run_unbounded(rng, calib=None, lam=1.0, max_trips=500):
    """Unbounded adaptive IPE: iterate until confident. Returns (rounds, x_sample)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    k, cont = 0, True
    while cont and k < max_trips:
        cont = attempt(sim)
        k += 1
    # deliver the held eigenstate read in X (its ideal axis)
    sim.h(TARGET)
    x = 1 - 2 * sim.measure(TARGET)
    return k, x


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N = 6000
    rounds, xs = [], []
    for _ in range(N):
        k, x = run_unbounded(rng, lam=0.0)
        rounds.append(k)
        xs.append(x)
    p_hat = 1.0 / (sum(rounds) / len(rounds))
    mu = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / np.sqrt(N))
    print(f"ipe (lam=0, N={N}): p_hat={p_hat:.4f} (analytic {P_ANALYTIC})")
    print(f"  <X> = {mu:+.4f} +- {se:.4f}  (ideal +1.000, |+> eigenstate)")
