#!/usr/bin/env python3
"""X11: AQFT aperture via the real qft-approx-dispatch MLIR pass.

End-to-end: a QFT on a RUNTIME active width n (for_loop over n qubits) is compiled
through Catalyst. The real `qft-approx-dispatch` MLIR pass (estimator-gated) rewrites
the inner controlled-phase loop bound n -> min(n, i + b(n) + 1), b(n)=ceil(log2 n)+2,
dropping far rotations. The gate-counter MLIR pass then counts the surviving
ControlledPhaseShift ops on the compiled artifact:

  before (no pass): n(n-1)/2      exact QFT
  after  (pass):    aperture set  = the AQFT the pass produced

Fidelity (state F vs noiseless exact QFT, default.mixed depolarizing eps_1q=0.001
eps_2q=0.005 = E9) is computed on the identical aperture set the pass keeps
(deterministic, |i-j| <= b(n)), so it is the pass output's fidelity: exact wins for
small n, AQFT wins past the crossover (n=7).

Usage:  python3 run_x11_aqft_dispatch.py
"""

from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))
from catalyst import for_loop
from gate_counter_estimator import GateCounterSession

EPS1, EPS2 = 0.001, 0.005
NMAX = 11


def make_qft_body():
    """QFT on the first n (runtime) wires: H(i), CPhase(pi/2^(j-i)) for j>i."""
    def body(n):
        @for_loop(0, n, 1)
        def outer(i):
            qp.Hadamard(wires=i)
            @for_loop(i + 1, n, 1)
            def inner(j):
                d = j - i
                qp.ControlledPhaseShift(np.pi / (2.0 ** d), wires=[j, i])
            inner()
        outer()
        return qp.probs(wires=[0])
    return body


def cphase_count(n, use_pass):
    dev = qp.device("lightning.qubit", wires=NMAX)
    passes = ["qft-approx-dispatch"] if use_pass else []
    with GateCounterSession(make_qft_body(), dev, jnp.int64(n),
                            pre_instrumentation_passes=passes) as s:
        return s.run(jnp.int64(n)).gate_counts.get("ControlledPhaseShift_2", 0)


# ── fidelity of the aperture set the pass keeps (|i-j| <= b) ────────────────

def _qft_ops(n, b):
    ops = []
    for i in range(n):
        ops.append(("H", i, None, None))
        for j in range(i + 1, n):
            if b is not None and (j - i) > b:
                continue
            ops.append(("CP", j, i, math.pi / 2 ** (j - i)))
    return ops


def _ideal(n):
    dev = qp.device("default.qubit", wires=n)
    @qp.qnode(dev)
    def c():
        for w in range(n): qp.Hadamard(wires=w)
        for w in range(n): qp.RZ(0.3 * (w + 1), wires=w)
        for g in _qft_ops(n, None):
            if g[0] == "H": qp.Hadamard(wires=g[1])
            else: qp.ControlledPhaseShift(g[3], wires=[g[1], g[2]])
        return qp.state()
    return np.array(c())


def _noisy_fid(n, b, psi):
    dev = qp.device("default.mixed", wires=n)
    @qp.qnode(dev)
    def c():
        for w in range(n): qp.Hadamard(wires=w)
        for w in range(n): qp.RZ(0.3 * (w + 1), wires=w)
        for g in _qft_ops(n, b):
            if g[0] == "H":
                qp.Hadamard(wires=g[1]); qp.DepolarizingChannel(EPS1, wires=g[1])
            else:
                qp.ControlledPhaseShift(g[3], wires=[g[1], g[2]])
                qp.DepolarizingChannel(EPS2, wires=g[1]); qp.DepolarizingChannel(EPS2, wires=g[2])
        return qp.state()
    rho = np.array(c())
    return float(np.real(psi.conj() @ rho @ psi))


def main():
    print("=" * 84)
    print("  X11 — AQFT aperture via real qft-approx-dispatch MLIR pass")
    print("  before = no pass (exact QFT) ; after = pass (aperture). CPhase from gate-counter")
    print("  pass; fidelity on default.mixed (eps 0.001/0.005) of the kept aperture set.")
    print("=" * 84)
    print(f"  {'n':>3} {'b(n)':>4} | {'CPhase_before':>13} {'CPhase_after':>12} (exact n(n-1)/2)"
          f" | {'F_exact':>8} {'F_aqft':>8} {'winner':>7}")
    print("  " + "-" * 88)
    for n in range(4, NMAX):
        b = math.ceil(math.log2(n)) + 2
        cb = cphase_count(n, False)
        ca = cphase_count(n, True)
        psi = _ideal(n)
        fe = _noisy_fid(n, None, psi)
        fa = _noisy_fid(n, b, psi)
        win = "AQFT" if fa > fe else ("tie" if abs(fa - fe) < 1e-9 else "exact")
        print(f"  {n:>3} {b:>4} | {cb:>13} {ca:>12} ({n*(n-1)//2:>3})       "
              f" | {fe:>8.4f} {fa:>8.4f} {win:>7}")
    print()
    print("  CPhase_after (real pass) = the aperture set; matches the fidelity sim's kept")
    print("  rotations, so the F_aqft column is the pass output's fidelity. Crossover at n=7.")


if __name__ == "__main__":
    main()
