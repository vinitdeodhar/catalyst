#!/usr/bin/env python3
"""X8: Cost-gated runtime dispatch between two MCX decompositions.

Demonstrates the symbolic-cost / runtime-versioning pass end-to-end:

  * The app provides a `for_loop(0, n, 1)` (runtime bound `n`) whose body applies a
    c-control MultiControlledX.
  * ResourceAnalysis classifies the loop as dynamic and gives the per-iteration body.
  * The `width-guarded-mcx-decomp` MLIR pass synthesizes TWO versions of the loop —
    ancilla-free (native, O(c^2) gates) and V-chain (2c-3 Toffolis on c-2 ancillas) —
    and guards them with a SYMBOLIC cost expression evaluated at runtime:

        saving(n) = (g_af(c) - g_vc(c)) * trip(n)          // symbolic in n
        scf.if saving(n) > cost-budget { V-chain } else { ancilla-free }

  * ONE compiled artifact then dispatches differently as `n` changes at runtime:
    small n -> ancilla-free (don't spend the ancillas), large n -> V-chain (the
    accumulated gate saving justifies the qubits). Crossover at n* = budget / delta.

Everything is real: the gate counts come from the gate-counter MLIR pass over the
circuit rewritten by the width-guarded-mcx-decomp MLIR pass; runtime is the timing
model over those counts (modeled QPU duration, not wall-clock).

Usage::
    python3 run_x8_cost_gated_dispatch.py [--c 3] [--cost-budget 30] [--device garnet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import for_loop
from gate_counter_estimator import GateCounterSession
from runtime_model import IQM_GARNET, IBM_HERON


def gate_saving_per_iter(c: int) -> int:
    """Must match the pass: g_af = 24c^2-116c+156, g_vc = 12c-18."""
    return (24 * c * c - 116 * c + 156) - (12 * c - 18)


def make_body(c: int):
    ctrl = list(range(c))
    tgt = c

    def body(n):
        for i in range(c):
            qp.Hadamard(wires=i)

        @for_loop(0, n, 1)
        def loop(_):
            qp.MultiControlledX(wires=ctrl + [tgt])

        loop()
        return qp.probs(wires=[tgt])

    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=3, help="number of MCX controls")
    ap.add_argument("--cost-budget", type=int, default=30)
    ap.add_argument("--device", choices=["garnet", "heron"], default="garnet")
    ap.add_argument("--n-values", type=int, nargs="+",
                    default=[2, 4, 6, 8, 12, 16, 24])
    args = ap.parse_args()

    device = IQM_GARNET if args.device == "garnet" else IBM_HERON
    c = args.c
    delta = gate_saving_per_iter(c)
    n_star = args.cost_budget / delta if delta else float("inf")

    # native width c+1; the V-chain branch alloc_qb's c-2 ancillas at runtime.
    dev = qp.device("lightning.qubit", wires=c + 1)
    body = make_body(c)
    passes = [f"width-guarded-mcx-decomp{{cost-budget={args.cost_budget}}}"]

    print("=" * 82)
    print("  X8 — Cost-gated runtime dispatch (one artifact, symbolic guard in n)")
    print(f"  c={c} controls   delta=g_af-g_vc={delta}   cost-budget={args.cost_budget}"
          f"   => n* = {n_star:.2f}")
    print(f"  guard: saving(n) = {delta}*n  ;  fire V-chain iff saving(n) > {args.cost_budget}")
    print(f"  device={device.name}   (runtime = timing-model output, not wall-clock)")
    print("=" * 82)
    print(f"  {'n':>4} {'saving':>7} {'dispatch':>13} {'runtime':>10}  gate counts")
    print("  " + "-" * 72)

    # Compile once (fixed arg only sets the shape); dispatch varies with run(n).
    with GateCounterSession(body, dev, jnp.int64(args.n_values[0]),
                            timing_model=device, pre_instrumentation_passes=passes) as sess:
        for n in args.n_values:
            r = sess.run(jnp.int64(n))
            gc = {k: v for k, v in r.gate_counts.items() if v > 0}
            fired = gc.get("Toffoli_3", 0) > 0
            pick = "V-chain" if fired else "ancilla-free"
            print(f"  {n:>4} {delta*n:>7} {pick:>13} {r.runtime_ns/1e3:>8.2f}us  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(gc.items())))

    print()
    print(f"  The SAME compiled artifact flips decomposition at n* = {n_star:.0f}: below it the")
    print("  ancilla-free body runs (native MCX, no ancillas); above it the V-chain body runs")
    print("  (Toffoli ladder). The guard condition is symbolic in the runtime bound n.")


if __name__ == "__main__":
    main()
