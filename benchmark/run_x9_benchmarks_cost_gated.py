#!/usr/bin/env python3
"""X9: Before/after runtime across benchmarks, via the cost-gated MCX pass.

Each benchmark is a `for_loop(0, n, 1)` (runtime bound) whose body holds the
benchmark's multi-controlled X gate(s). The real `width-guarded-mcx-decomp` MLIR
pass rewrites the loop; gate counts come from the gate-counter MLIR pass; runtime
from the timing model.

before = pass with a huge cost-budget  -> guard never fires -> native (ancilla-free) MCX
after  = pass with cost-budget 0       -> guard always fires -> V-chain MCX

Both are measured at the same runtime bound n, so the delta is purely the
decomposition the pass dispatches to. (The runtime *flip* on one artifact as n
crosses n* is shown separately in run_x8_cost_gated_dispatch.py.)

Usage::
    python3 run_x9_benchmarks_cost_gated.py [--n 8] [--device garnet]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import for_loop
from gate_counter_estimator import GateCounterSession
from runtime_model import IQM_GARNET, IBM_HERON

HUGE = 10 ** 12  # cost-budget that never fires (native baseline)


def _mcz(c, ctrl, tgt):
    qp.Hadamard(wires=tgt)
    qp.MultiControlledX(wires=ctrl + [tgt])
    qp.Hadamard(wires=tgt)


def build_synthetic(n_data=4):
    """1 MCX per iteration (bare oracle)."""
    c = n_data - 1

    def body(n):
        for i in range(n_data):
            qp.Hadamard(wires=i)

        @for_loop(0, n, 1)
        def loop(_):
            qp.MultiControlledX(wires=list(range(c)) + [n_data - 1])
        loop()
        return qp.probs(wires=[n_data - 1])
    return body, n_data, c, 1


def _grover_step_builder(n_data):
    """2 MCX per iteration (oracle MCZ + diffuser MCZ) — Grover/BBHT/nested shape."""
    c = n_data - 1
    ctrl = list(range(c))
    tgt = n_data - 1

    def body(n):
        for i in range(n_data):
            qp.Hadamard(wires=i)

        @for_loop(0, n, 1)
        def loop(_):
            _mcz(c, ctrl, tgt)                                   # oracle
            for i in range(n_data):
                qp.Hadamard(wires=i); qp.PauliX(wires=i)
            _mcz(c, ctrl, tgt)                                   # diffuser
            for i in range(n_data):
                qp.PauliX(wires=i); qp.Hadamard(wires=i)
        loop()
        return qp.probs(wires=list(range(n_data)))
    return body, n_data, c, 2


BENCHMARKS = {
    "Synthetic MCX (n_data=4)":  lambda: build_synthetic(4),
    "Grover (n_data=4)":         lambda: _grover_step_builder(4),
    "Nested oracle (n_data=5)":  lambda: _grover_step_builder(5),
    "BBHT oracle (n_data=6)":    lambda: _grover_step_builder(6),
}


def run_once(body, wires, budget, n, device):
    dev = qp.device("lightning.qubit", wires=wires)
    passes = [f"width-guarded-mcx-decomp{{cost-budget={budget}}}"]
    with GateCounterSession(body, dev, jnp.int64(n), timing_model=device,
                            pre_instrumentation_passes=passes) as s:
        r = s.run(jnp.int64(n))
        gc = {k: v for k, v in r.gate_counts.items() if v > 0}
        return r.runtime_ns, gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="runtime loop bound")
    ap.add_argument("--device", choices=["garnet", "heron"], default="garnet")
    args = ap.parse_args()
    device = IQM_GARNET if args.device == "garnet" else IBM_HERON
    n = args.n

    print("=" * 84)
    print(f"  X9 — Before/after runtime via cost-gated MCX pass   (n={n}, {device.name})")
    print("  before = native (budget never fires) ; after = V-chain (budget 0) ; same n")
    print("=" * 84)
    print(f"  {'Benchmark':<26} {'c':>3} {'MCX/it':>6} {'t_before':>10} {'t_after':>10} {'speedup':>8}")
    print("  " + "-" * 70)

    for name, mk in BENCHMARKS.items():
        body, n_data, c, nmcx = mk()
        wires = n_data  # native width; V-chain alloc_qb adds c-2 at runtime
        t_before, gcb = run_once(body, wires, HUGE, n, device)
        t_after, gca = run_once(body, wires, 0, n, device)
        spd = t_before / t_after if t_after else float("inf")
        print(f"  {name:<26} {c:>3} {nmcx:>6} {t_before/1e3:>8.2f}us {t_after/1e3:>8.2f}us {spd:>7.2f}x")

    print()
    print("  All gate counts from the gate-counter MLIR pass over IR rewritten by the")
    print("  width-guarded-mcx-decomp MLIR pass; runtime = timing-model output, not wall-clock.")


if __name__ == "__main__":
    main()
