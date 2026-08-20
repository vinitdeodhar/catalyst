"""
fast_target.py -- exact fast executors for the carry-type sweeps.

The rus benchmarks were designed with a TARGET-INDEPENDENT coin (the ancillas
never touch the carried qubit), so the carried qubit is a 1-qubit system and the
trip count is a classical geometric draw. Simulating just that 1-qubit
trajectory -- with the same per-iteration idle time the target accrues in the
full circuit (3 ancilla readouts + feedback), the same qsim noise/leakage model,
and the same cut protocol -- is *exact* for the delivered-state statistics and
~10x faster than the full 4-qubit circuit. Used for the evaluation sweeps; the
full-circuit runners in knit_runtime remain for validation and lockstep with the
MLIR.

Validated against the full sim in sim/validate.py (gate_fast).
"""

import numpy as np

from sim.qsim import QSim, load_calib
from sim.knit_runtime import cut_and_reprepare, _read

P = 5.0 / 8.0  # rus success probability


def _prep_psi0(sim):
    sim.h(0); sim.t(0); sim.h(0); sim.t(0); sim.h(0)  # H T H T H |0>


def _dt_iter(calib):
    """Idle time the carried target accrues per attempt: 3 ancilla readouts,
    3 ancilla prep gates, and the feedback window."""
    return 3 * calib["readout"] + 3 * calib["gate_1q"] + calib["tau"]


def _draw_k(rng, p=P, max_trips=2000):
    return min(int(rng.geometric(p)), max_trips)


def fast_unbounded(rng, calib, lam, N=1, basis="Z", p=P):
    c = load_calib(calib)
    sim = QSim(1, calib=c, lam=lam, rng=rng)
    _prep_psi0(sim)
    dt = _dt_iter(c)
    ktot = 0
    for _ in range(N):
        k = _draw_k(rng, p)
        for _ in range(k):
            sim._idle_qubit(0, dt)
        ktot += k
    return _read(sim, 0, basis), ktot


def fast_truncated(rng, calib, lam, C, N=1, basis="Z", p=P):
    c = load_calib(calib)
    sim = QSim(1, calib=c, lam=lam, rng=rng)
    _prep_psi0(sim)
    dt = _dt_iter(c)
    ktot = 0
    for _ in range(N):
        k = _draw_k(rng, p)
        nrun = min(k, C)
        for _ in range(nrun):
            sim._idle_qubit(0, dt)
        ktot += nrun
        if k > C:
            return None, ktot, True
    return _read(sim, 0, basis), ktot, False


def fast_knit(rng, calib, lam, C, N=1, basis="Z", p=P):
    c = load_calib(calib)
    sim = QSim(1, calib=c, lam=lam, rng=rng)
    _prep_psi0(sim)
    dt = _dt_iter(c)
    w = 1.0
    ktot, ncut = 0, 0
    for _ in range(N):
        k = _draw_k(rng, p)
        for i in range(1, k + 1):
            sim._idle_qubit(0, dt)
            if i < k and i % C == 0:      # failing attempt at a cut boundary
                w *= cut_and_reprepare(sim, 0, rng)
                ncut += 1
        ktot += k
    return w * _read(sim, 0, basis), ktot, ncut, w


def fast_knit_det(rng, calib, lam, C, N=1, basis="Z", p=P):
    """The cheap deterministic gamma=1 cut (proven-known-state): every C failing
    iterations, measure (end segment) + force |0> (clear leakage) + re-prepare
    the KNOWN state |psi0>. Weight is identically 1 -> ZERO sampling variance,
    and each cut refreshes the carried qubit clean. Valid only because the pass
    proved the carried state is |psi0>."""
    c = load_calib(calib)
    sim = QSim(1, calib=c, lam=lam, rng=rng)
    _prep_psi0(sim)
    dt = _dt_iter(c)
    ktot, ncut = 0, 0
    for _ in range(N):
        k = _draw_k(rng, p)
        for i in range(1, k + 1):
            sim._idle_qubit(0, dt)
            if i < k and i % C == 0:
                sim.measure(0)            # end the coherent segment
                sim.force_zero(0)         # fresh |0>, clears leakage
                _prep_psi0(sim)           # re-prepare the known |psi0>
                ncut += 1
        ktot += k
    return _read(sim, 0, basis), ktot, ncut, 1.0
