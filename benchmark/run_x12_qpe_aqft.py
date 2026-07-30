#!/usr/bin/env python3
"""X12: qft-approx-dispatch on a real QPE circuit.

Quantum phase estimation of a known eigenphase phi, with a RUNTIME number of
counting qubits n. The inverse QFT on the counting register is an explicit
H + controlled-phase nest (runtime-bound) -- the archetypal AQFT target. The real
`qft-approx-dispatch` MLIR pass fires on it (estimator-gated), rewriting the inner
bound n -> min(n, i+b(n)+1), b(n)=ceil(log2 n)+2.

Two things measured:
  1. CPhase count on the compiled QPE (gate-counter pass): before = full inverse
     QFT n(n-1)/2, after = aperture -- proves the pass fires on a real QPE.
  2. Phase-estimation accuracy under noise (default.mixed, eps 0.001/0.005 = E9):
     P(correct readout) for exact-invQFT vs AQFT-invQFT, per n. AQFT is more
     noise-robust, so it wins past a crossover.

phi is chosen dyadic (k/2^n) so the ideal readout is an exact bitstring.

Usage:  python3 run_x12_qpe_aqft.py
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
NMAX = 11              # device wires (n counting + 1 eigenstate, n <= NMAX-1)
PHI_NUM = 3           # phi = 2*pi * PHI_NUM / 2^n  (dyadic -> exact readout)


# ── QPE body for Catalyst (runtime n counting qubits) ───────────────────────

def make_qpe_body():
    def body(n):
        anc = NMAX - 1                                   # eigenstate qubit (fixed wire)
        qp.PauliX(wires=anc)                             # |1> eigenstate of PhaseShift
        # superposition on counting register
        @for_loop(0, n, 1)
        def sup(k):
            qp.Hadamard(wires=k)
        sup()
        # phase kickback: controlled-PhaseShift(2^k * phi) from counting k onto anc
        @for_loop(0, n, 1)
        def kick(k):
            ang = (2.0 * np.pi * PHI_NUM / (2.0 ** n)) * (2.0 ** k)
            qp.ControlledPhaseShift(ang, wires=[k, anc])
        kick()
        # inverse QFT on counting register (the AQFT target): H(i) + CPhase cascade
        @for_loop(0, n, 1)
        def iqft(i):
            qp.Hadamard(wires=i)
            @for_loop(i + 1, n, 1)
            def inner(j):
                d = j - i
                qp.ControlledPhaseShift(-np.pi / (2.0 ** d), wires=[j, i])
            inner()
        iqft()
        return qp.probs(wires=[0])
    return body


def cphase_count(n, use_pass):
    dev = qp.device("lightning.qubit", wires=NMAX)
    passes = ["qft-approx-dispatch"] if use_pass else []
    with GateCounterSession(make_qpe_body(), dev, jnp.int64(n),
                            pre_instrumentation_passes=passes) as s:
        return s.run(jnp.int64(n)).gate_counts.get("ControlledPhaseShift_2", 0)


# ── noisy QPE accuracy in PennyLane (exact vs AQFT inverse QFT) ──────────────

def _qpe_probs(n, b, noisy):
    """Return counting-register probability distribution; b=None -> exact iQFT."""
    dev = qp.device("default.mixed" if noisy else "default.qubit", wires=n + 1)
    anc = n
    phi = 2.0 * np.pi * PHI_NUM / (2.0 ** n)

    @qp.qnode(dev)
    def c():
        qp.PauliX(wires=anc)
        for k in range(n):
            qp.Hadamard(wires=k)
            if noisy: qp.DepolarizingChannel(EPS1, wires=k)
        for k in range(n):
            qp.ControlledPhaseShift(phi * 2 ** k, wires=[k, anc])
            if noisy:
                qp.DepolarizingChannel(EPS2, wires=k); qp.DepolarizingChannel(EPS2, wires=anc)
        # inverse QFT
        for i in range(n):
            qp.Hadamard(wires=i)
            if noisy: qp.DepolarizingChannel(EPS1, wires=i)
            for j in range(i + 1, n):
                if b is not None and (j - i) > b:
                    continue
                qp.ControlledPhaseShift(-np.pi / 2 ** (j - i), wires=[j, i])
                if noisy:
                    qp.DepolarizingChannel(EPS2, wires=j); qp.DepolarizingChannel(EPS2, wires=i)
        return qp.probs(wires=list(range(n)))
    return np.array(c())


def _success(n, b):
    """P(correct readout): probability mass on the ideal bitstring."""
    p = _qpe_probs(n, b, noisy=True)
    ideal = int(np.argmax(_qpe_probs(n, None, noisy=False)))  # noiseless exact peak
    return float(p[ideal])


def main():
    print("=" * 82)
    print("  X12 — qft-approx-dispatch on a real QPE (inverse QFT), runtime n counting bits")
    print("  CPhase from gate-counter pass; P(correct) on default.mixed (eps 0.001/0.005)")
    print("=" * 82)
    print(f"  {'n':>3} {'b(n)':>4} | {'CPhase_bef':>10} {'CPhase_aft':>10} | "
          f"{'P_exact':>8} {'P_aqft':>8} {'winner':>7}")
    print("  " + "-" * 70)
    for n in range(4, NMAX):
        b = math.ceil(math.log2(n)) + 2
        cb = cphase_count(n, False)     # full: kick n + iQFT n(n-1)/2
        ca = cphase_count(n, True)      # aperture on the iQFT part
        pe = _success(n, None)
        pa = _success(n, b)
        win = "AQFT" if pa > pe + 1e-6 else ("tie" if abs(pa - pe) <= 1e-6 else "exact")
        print(f"  {n:>3} {b:>4} | {cb:>10} {ca:>10} | {pe:>8.4f} {pa:>8.4f} {win:>7}")
    print()
    print("  CPhase_bef includes n kickback + n(n-1)/2 inverse-QFT rotations; the pass cuts")
    print("  only the inverse-QFT tail (kickback loop body is not a bare CPhase cascade match).")
    print("  P(correct) is QPE readout accuracy under noise: AQFT inverse QFT wins past crossover.")


if __name__ == "__main__":
    main()
