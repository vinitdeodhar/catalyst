"""
rus_rx_ibm -- primary carry-type benchmark (RUS-shaped, physically faithful).

Python mirror driving the Track-B trajectory simulator. Wire 0 = TARGET (the
carried, never-measured qubit); wires 1,2,3 = coin ancillas (measured + reset
each attempt). Per-attempt success probability p = 5/8 (geometric trip counts).

WHY THIS DIFFERS FROM loop_knitting_benchmarks.py (see NOTES.md):
The provided schematic gadget (H,H,Toffoli,S,Toffoli,H,H + X byproduct, target
in |0>) is *physically degenerate* for the noise study: it applies only a
Z-diagonal action to the target, so with target |0> it is trivial, its +0.46
<Z> is a trip-count parity artifact, and "failure composes to identity" is
violated (each failure applies a net X). Crucially a Z-eigenstate is immune to
the dephasing the transform targets, so that gadget can exhibit NO crossover.

This mirror instead implements a faithful carry-type loop with all the
properties the transform's cost model and the experiment require:
  * the TARGET holds a fixed non-Clifford magic state |psi0> = H T H T H |0>
    (Bloch ~ (0.707, 0.5, 0.5); <Z>_ideal = 0.5) coherently ACROSS iterations;
  * each attempt runs a Clifford coin on fresh ancillas (p = 5/8 exact,
    geometric) and measures/reset only the ancillas -- the target is never
    measured and receives IDENTITY on failure (exact), so the delivered state
    is trip-count-independent at lam = 0;
  * the carried qubit therefore accrues (readout + tau) idle decoherence PER
    held iteration -- the mechanism (spec 4.2 idle decay) the transform bounds.
This is the quantum-memory / repeater-cutoff carry-type instance (writeup Sec 5).

Op set is Clifford+T only (spec 4.1); |psi0| uses two T gates.
Keep this file in lockstep with benchmarks/rus_rx_ibm.mlir.
"""

import numpy as np

from sim.qsim import QSim

N_WIRES = 4
TARGET = 0
ANCILLAS = (1, 2, 3)
P_ANALYTIC = 5.0 / 8.0
Z_IDEAL = 0.5  # <Z> of |psi0> = H T H T H |0>


def prepare_input(sim):
    """Program input state on the carried target: magic state H T H T H |0>."""
    q = TARGET
    sim.h(q)
    sim.t(q)
    sim.h(q)
    sim.t(q)
    sim.h(q)


def _coin_fail(ma, mb, mc):
    """Fail predicate true on exactly 3 of 8 uniform outcomes -> P(fail)=3/8."""
    return (ma and mb) or (mc and not ma and not mb)


def attempt(sim):
    """One RUS attempt. Ancillas provide the p=5/8 geometric coin; the target
    is held (idle) and never measured. Returns the fail flag (bool)."""
    a, b, c = ANCILLAS
    sim.h(a)
    sim.h(b)
    sim.h(c)
    ma = sim.measure(a)
    mb = sim.measure(b)
    mc = sim.measure(c)
    fail = bool(_coin_fail(ma, mb, mc))
    # classical-feedback window: the carried target idles for tau here
    sim.feedback(active=[])
    # fresh ancillas next attempt (leakage-free); the target is untouched
    sim.force_zero(a)
    sim.force_zero(b)
    sim.force_zero(c)
    return fail


def run_unbounded(rng, calib=None, lam=1.0, max_trips=500):
    """Unbounded RUS: retry until success. Returns (trip_count, z_sample)."""
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
    print(f"rus_rx_ibm (lam=0, N={N}): p_hat={p_hat:.4f} (analytic 0.625)")
    print(f"  <Z> = {mu:+.4f} +- {se:.4f}  (ideal {Z_IDEAL})")
