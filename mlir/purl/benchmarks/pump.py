"""
pump -- entanglement-pumping proxy (carry-type, leakage-heavy), spec Sec 6.1.

Python mirror driving the Track-B trajectory simulator. Wire 0 = DATA (the held,
never-measured resource); wire 1 = ANCILLA (reset + reused every iteration). The
held wire carries the same non-Clifford magic state |psi0> = H T H T H |0> as the
rus benchmarks (a SINGLE-QUBIT PROXY for the true two-qubit purified resource --
the same simplification rus uses, spec 6.1).

WHY THIS BENCHMARK (spec 6.1): entanglement pumping retries a purification round
against one held resource until a herald succeeds, so the resource ages through a
geometric number of TWO-QUBIT-GATE-HEAVY rounds -- exactly Purl's carry shape, but
with MANY 2q gates per iteration (here 6), so per-2q-gate leakage (spec 4.1) is a
first-order term rather than a sliver. It is the primary consumer of the per-gate
leakage model and gives the benchmark section a citable purification/pumping anchor.

Loop body, per iteration (all Clifford -> the known-state proof certifies identity):
  1. reset the ANCILLA and prepare the coin (Hadamard; the p=0.1 herald is realized
     as an exact classical draw, exactly as rus_lowp does -- the Ry(theta) coin of
     spec 6.1 step 1 is this same p-proxy, keeping the trip count noise-independent);
  2. the entangling sandwich CNOT(a->d), S(a), CNOT(a->d) repeated N_BLOCKS=3 times
     (6 two-qubit gates on d). Both CNOTs share the control a and S touches only a,
     so the two flips on d cancel: the NET action on d is the IDENTITY and the
     herald outcome is independent of the data state (proven in the pass, spec 3.4);
  3. measure the ancilla (herald readout); success exits, failure repeats.

The 6 CNOTs charge 6x per-2q-gate leakage + depolarizing on the held wire each
iteration (qsim._leak_2q), the reset-clearable error a refresh cut removes.

Keep this file in lockstep with mlir/test/Quantum/Purl/pump_refresh.mlir.
"""

import numpy as np

from sim.qsim import QSim

N_WIRES = 2
DATA = 0            # carried, never measured in the loop
ANCILLA = 1        # reset + reused each iteration
P_PUMP = 0.1       # herald success probability (mean 10 trips, matching rus_lowp)
P_ANALYTIC = P_PUMP
Z_IDEAL = 0.5      # <Z> of |psi0> = H T H T H |0>
N_BLOCKS = 3       # sandwich repeats -> N2Q_PER_ITER = 2 * N_BLOCKS two-qubit gates
N2Q_PER_ITER = 2 * N_BLOCKS  # 6: the per-iteration 2q-gate count the pass reports


def prepare_input(sim):
    """Program input on the carried data wire: magic state H T H T H |0>."""
    q = DATA
    sim.h(q)
    sim.t(q)
    sim.h(q)
    sim.t(q)
    sim.h(q)


def dt_iter(calib):
    """Idle time the held data wire accrues per iteration from the ANCILLA's
    spectator ops: the coin Hadamard + the N_BLOCKS S gates (1q each), the herald
    readout, and the feedback window. The 6 CNOTs are charged as 2q gates
    (leakage + depolarizing), not idle -- d is active during its own gate."""
    return calib["readout"] + (1 + N_BLOCKS) * calib["gate_1q"] + calib["tau"]


def attempt(sim):
    """One pumping round. The ancilla runs a Clifford coin and the CNOT-S-CNOT
    sandwich (net identity on d, 6 two-qubit gates), then is measured (herald).
    The herald is an exact classical p-draw (as in rus_lowp). Returns fail (bool)."""
    a, d = ANCILLA, DATA
    sim.force_zero(a)                 # reset the ancilla (leakage-free)
    sim.h(a)                          # coin prep (Clifford proxy for the Ry coin)
    for _ in range(N_BLOCKS):         # 3x sandwich = 6 two-qubit gates on d
        sim.cnot(a, d)
        sim.s(a)
        sim.cnot(a, d)
    sim.measure(a)                    # herald readout (timing/noise on the held d)
    sim.feedback(active=[])           # classical-feedback window: d idles for tau
    return bool(sim.rng.random() < (1.0 - P_PUMP))


def run_unbounded(rng, calib=None, lam=1.0, max_trips=500):
    """Unbounded entanglement pumping: retry until the herald succeeds.
    Returns (trip_count, z_sample)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    k, fail = 0, True
    while fail and k < max_trips:
        fail = attempt(sim)
        k += 1
    z = sim.measure(DATA)
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
    print(f"pump (lam=0, N={N}): p_hat={p_hat:.4f} (analytic {P_PUMP}), "
          f"n2q/iter={N2Q_PER_ITER}")
    print(f"  <Z> = {mu:+.4f} +- {se:.4f}  (ideal {Z_IDEAL})")
