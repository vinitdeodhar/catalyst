#!/usr/bin/env python3
"""X13: qft-approx-dispatch on Shor's final QFT.

Shor's period finding ends with an inverse QFT on the exponent register. After
modular exponentiation and measuring the second register, the exponent register
holds a PERIODIC superposition; the final QFT turns its period into readable
peaks. We prepare that periodic state directly (standard isolation of Shor's
final QFT -- full modexp is not simulable at useful sizes) and apply the inverse
QFT, which is the runtime-bound H + controlled-phase nest the real
`qft-approx-dispatch` MLIR pass rewrites (aperture b(n)=ceil(log2 n)+2).

Periodic input: period r = 2^(n-s) (s high qubits in uniform superposition, low
bits fixed) -> exact dyadic peaks, so the ideal readout is a sharp bitstring.

Measured:
  1. CPhase count on the compiled circuit (gate-counter pass): before = full
     inverse QFT n(n-1)/2, after = aperture -- the pass firing on Shor's final QFT.
  2. Period-recovery accuracy under noise (default.mixed, eps 0.001/0.005 = E9):
     P(correct peak) exact-invQFT vs AQFT-invQFT, per n.

Usage:  python3 run_x13_shor_qft.py
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
S = 2                    # high qubits in superposition -> period r = 2^(n-S)


def _prep_periodic(n):
    """State Shor leaves in the exponent register: period-2^(n-S) comb."""
    for w in range(S):              # top S wires: uniform superposition
        qp.Hadamard(wires=w)
    qp.PauliX(wires=n - 1)          # fix a low bit (nonzero offset)


def make_shor_qft_body():
    def body(n):
        # periodic input (stands in for post-modexp register)
        @for_loop(0, S, 1)
        def sup(w):
            qp.Hadamard(wires=w)
        sup()
        qp.PauliX(wires=n - 1)
        # Shor's final inverse QFT (the AQFT target)
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
    with GateCounterSession(make_shor_qft_body(), dev, jnp.int64(n),
                            pre_instrumentation_passes=passes) as s:
        return s.run(jnp.int64(n)).gate_counts.get("ControlledPhaseShift_2", 0)


def _probs(n, b, noisy):
    dev = qp.device("default.mixed" if noisy else "default.qubit", wires=n)
    @qp.qnode(dev)
    def c():
        for w in range(S):
            qp.Hadamard(wires=w)
            if noisy: qp.DepolarizingChannel(EPS1, wires=w)
        qp.PauliX(wires=n - 1)
        if noisy: qp.DepolarizingChannel(EPS1, wires=n - 1)
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


def _success(n, b, ideal_peaks):
    p = _probs(n, b, noisy=True)
    return float(sum(p[k] for k in ideal_peaks))


def main():
    print("=" * 82)
    print("  X13 — qft-approx-dispatch on Shor's final (inverse) QFT, runtime n exponent bits")
    print("  periodic input prepared directly (modexp isolated). CPhase from gate-counter pass;")
    print("  P(correct peaks) on default.mixed (eps 0.001/0.005).")
    print("=" * 82)
    print(f"  {'n':>3} {'b(n)':>4} | {'CPhase_bef':>10} {'CPhase_aft':>10} | "
          f"{'P_exact':>8} {'P_aqft':>8} {'winner':>7}")
    print("  " + "-" * 70)
    for n in range(4, NMAX):
        b = math.ceil(math.log2(n)) + 2
        cb = cphase_count(n, False)
        ca = cphase_count(n, True)
        # ideal period peaks = the top-2^S noiseless outcomes of exact inverse QFT
        p0 = _probs(n, None, noisy=False)
        ideal_peaks = list(np.argsort(p0)[-(2 ** S):])
        pe = _success(n, None, ideal_peaks)
        pa = _success(n, b, ideal_peaks)
        win = "AQFT" if pa > pe + 1e-6 else ("tie" if abs(pa - pe) <= 1e-6 else "exact")
        print(f"  {n:>3} {b:>4} | {cb:>10} {ca:>10} | {pe:>8.4f} {pa:>8.4f} {win:>7}")
    print()
    print("  CPhase = inverse-QFT rotations only (periodic prep is X+H, no CPhase); pass cuts")
    print("  the tail (matches run_x11 iQFT aperture). P = mass on the period peaks under noise.")


if __name__ == "__main__":
    main()
