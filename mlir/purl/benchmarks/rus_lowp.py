"""
rus_lowp -- low-p carry-type benchmark: a memory qubit held while a heralded
process retries (the quantum-repeater / heralded-entanglement regime, spec Sec 6).
Same carried magic state |psi0> = H T H T H |0> as rus_rx_ibm, but the herald
succeeds with a LOW probability p (default 0.1, mean trip count 10), so the
carried qubit is held through many attempts -- the regime the transform is meant
for.

DIFFERENCE FROM rus_rx_ibm (spec 5.1 / review A4): rus_lowp's loop body applies a
net-identity 2q "touch" to the carried target each iteration (a CZ whose fresh
partner ancilla is |0> -- logically identity, so the `--purl` known-state proof
still certifies the carried state and REFRESH fires -- but a physical 2q gate, so
per-2q-gate leakage (spec 4.1) accrues on the target). This is the reset-clearable
error refresh removes: unbounded holding lets it accumulate over ~1/p touches,
refresh caps it at C. rus_rx_ibm holds the target idle (no 2q gate on it), so its
per-2q-gate leakage is zero and the pass selects NONE.
"""

import numpy as np

from sim.qsim import QSim
from benchmarks.rus_rx_ibm import (  # noqa: F401  (shared carried-qubit model)
    N_WIRES, TARGET, ANCILLAS, Z_IDEAL, prepare_input,
)

# heralding success probability per attempt (low: repeater / heralded regime)
P_LOWP = 0.1
P_ANALYTIC = P_LOWP


def attempt(sim):
    """One low-p attempt. Unlike rus_rx_ibm, the carried target is touched by a
    net-identity 2q gate (leakage source, still provably identity); the herald is
    a target-independent low-p coin and the target idles through its window.
    Returns the fail flag (bool)."""
    # net-identity 2q maintenance on the carried target (spec 5.1): logically I,
    # charges one 2q gate's depolarizing + per-2q-gate leakage on the target.
    sim.touch_2q(TARGET)
    # target-independent low-p herald; the target idles through the coin window
    # (3 ancilla readouts + feedback), matching rus_rx_ibm's per-iteration idle.
    a, b, c = ANCILLAS
    sim.h(a)
    sim.h(b)
    sim.h(c)
    sim.measure(a)
    sim.measure(b)
    sim.measure(c)
    sim.feedback(active=[])
    sim.force_zero(a)
    sim.force_zero(b)
    sim.force_zero(c)
    return bool(sim.rng.random() < (1.0 - P_LOWP))


def run_unbounded(rng, calib=None, lam=1.0, max_trips=500):
    """Unbounded low-p RUS: retry until success. Returns (trip_count, z_sample)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    k, fail = 0, True
    while fail and k < max_trips:
        fail = attempt(sim)
        k += 1
    z = sim.measure(TARGET)
    return k, (1 - 2 * z)


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    N = 6000
    trips, zs = [], []
    for _ in range(N):
        k, z = run_unbounded(rng, lam=0.0)
        trips.append(k)
        zs.append(z)
    p_hat = 1.0 / (sum(trips) / len(trips))
    mu = float(np.mean(zs))
    se = float(np.std(zs, ddof=1) / np.sqrt(N))
    print(f"rus_lowp (lam=0, N={N}): p_hat={p_hat:.4f} (analytic {P_LOWP})")
    print(f"  <Z> = {mu:+.4f} +- {se:.4f}  (ideal {Z_IDEAL})")
