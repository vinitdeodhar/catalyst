#!/usr/bin/env python3
"""X14: compile-time fresh-ancilla vs MCMR allocation via the real MLIR pass.

Syndrome extraction: N static rounds, weight-4 stabilizer, one MCMR ancilla.
The real `fresh-ancilla-alloc` MLIR pass (estimator-gated) replaces the reused
ancilla + reset with a fresh ancilla per round (index c+i, reset dropped) when
`c + N <= qubit-budget`.

  before = pass with qubit-budget=0  (never fires -> MCMR: 1 ancilla, N resets)
  after  = pass with qubit-budget=big (fires -> fresh: N ancillas, 0 resets)

Confirmed by the gate-counter pass: the reset PauliX ops disappear. Data-qubit
fidelity is modeled (MCMR data idles through N reset/feedback rounds; fresh data
does not) -- the feasibility benefit the go/no-go established, now driven by the
real compiler pass.

Usage:  python3 run_x14_fresh_ancilla.py
"""

from __future__ import annotations
import math, sys
from pathlib import Path
import pennylane as qp
from catalyst import for_loop, measure

sys.path.insert(0, str(Path(__file__).parent))
from gate_counter_estimator import GateCounterSession

# timing model (us): syndrome round gates + reset/feedback latency
T_EXTRACT, T_FB, T2 = 0.20, 3.0, 30.0


def make_syndrome_body(N):
    def body():
        @for_loop(0, N, 1)
        def rnd(i):
            anc = 4
            qp.Hadamard(wires=anc)
            for d in range(4):
                qp.CNOT(wires=[anc, d])
            qp.Hadamard(wires=anc)
            measure(anc, reset=True)          # MCMR
        rnd()
        return qp.probs(wires=[0])
    return body


def reset_count(N, budget):
    dev = qp.device("lightning.qubit", wires=4 + N + 1)
    passes = [f"fresh-ancilla-alloc{{qubit-budget={budget}}}"]
    with GateCounterSession(make_syndrome_body(N), dev,
                            pre_instrumentation_passes=passes) as s:
        gc = s.run().gate_counts
        return gc.get("PauliX_1", 0)          # reset PauliX ops


def data_fidelity(N, fresh):
    idle = N * T_EXTRACT if fresh else N * (T_EXTRACT + T_FB)
    return math.exp(-idle / T2)


def main():
    print("=" * 78)
    print("  X14 — fresh-ancilla vs MCMR via real fresh-ancilla-alloc MLIR pass")
    print("  reset PauliX from gate-counter pass; data fidelity modeled (idle/T2).")
    print("=" * 78)
    print(f"  {'N':>3} {'ancillas':>9} | {'reset_MCMR':>10} {'reset_fresh':>11} | "
          f"{'F_MCMR':>7} {'F_fresh':>7} {'gain':>7}")
    print("  " + "-" * 66)
    for N in [4, 8, 12, 15]:
        r_mcmr = reset_count(N, 0)            # never fires -> MCMR baseline
        r_fresh = reset_count(N, 4 + N)       # fires (c=4, fits) -> fresh
        fm, ff = data_fidelity(N, False), data_fidelity(N, True)
        print(f"  {N:>3} {('1 -> '+str(N)):>9} | {r_mcmr:>10} {r_fresh:>11} | "
              f"{fm:>7.4f} {ff:>7.4f} {ff/fm-1:>+6.1%}")
    print()
    print("  Real pass removes all N reset PauliX ops (reset_fresh=0), confirming the")
    print("  transform; the modeled data fidelity is the feasibility benefit (fresh wins")
    print("  whenever the N ancillas fit -- a COMPILE-TIME choice, not a runtime crossover).")


if __name__ == "__main__":
    main()
