#!/usr/bin/env python3
"""X10: cost-gated MCX pass on BBHT + MQT-grounded Grover.

Three benchmarks, all run through the real width-guarded-mcx-decomp + gate-counter
MLIR passes; runtime is the timing-model output (modeled QPU duration).

  * BBHT (n_data=6, c=5): the *real* dynamic circuit — inner for_loop(0,k,1) with
    runtime, measurement-driven k. The pass fires on that loop. Profiled over the
    true stochastic trip counts; before/after = native vs V-chain, mean over shots.

  * Grover nq=7 (c=5) and nq=9 (c=7): sizes and control counts taken from MQT Bench
    (`get_benchmark('grover', ...)`: diffuser is a (nq-2)-controlled phase; fixed
    Grover iteration counts 6 and 12). Grover's loop count is compile-time in
    principle; we wrap it as for_loop(0, k, 1) with k a runtime argument and set k
    to MQT's iteration count, so the same pass machinery applies (reported honestly
    as a fixed-iteration case).

before = pass with a huge cost-budget (never fires -> native ancilla-free MCX)
after  = pass with cost-budget 0      (always fires -> V-chain)

Usage::
    python3 run_x10_mqt_bbht_grover.py [--n-profile 30] [--device garnet]
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import for_loop, measure, while_loop
from gate_counter_estimator import GateCounterSession
from runtime_model import IQM_GARNET, IBM_HERON

HUGE = 10 ** 12
_LAMBDA = 6.0 / 5.0


def _mcz(n_data):
    tgt = n_data - 1
    qp.Hadamard(wires=tgt)
    qp.MultiControlledX(wires=list(range(n_data - 1)) + [tgt])
    qp.Hadamard(wires=tgt)


def _grover_step(n_data):
    _mcz(n_data)                                       # oracle
    for i in range(n_data):
        qp.Hadamard(wires=i); qp.PauliX(wires=i)
    _mcz(n_data)                                       # diffuser
    for i in range(n_data):
        qp.PauliX(wires=i); qp.Hadamard(wires=i)


# ── BBHT: real dynamic circuit, runtime measurement-driven k ────────────────

def build_bbht(n_data=6):
    n_space = jnp.int64(2 ** n_data)

    def body():
        key = jax.random.PRNGKey(0)

        @while_loop(lambda found, _m, _k: ~found)
        def outer(found, m, rng_key):
            rng_key, sub = jax.random.split(rng_key)
            k = jax.random.randint(sub, shape=(), minval=jnp.int64(1),
                                   maxval=jnp.int64(m) + 1)
            for i in range(n_data):
                qp.Hadamard(wires=i)

            @for_loop(0, k, 1)                          # runtime bound k
            def steps(_):
                _grover_step(n_data)

            steps()
            bits = jnp.zeros(n_data, dtype=jnp.int64)
            for i in range(n_data):
                bits = bits.at[i].set(jnp.int64(measure(i, reset=True)))
            found_now = jnp.all(bits == 1)
            new_m = jnp.minimum(jnp.float64(_LAMBDA) * m, jnp.sqrt(jnp.float64(n_space)))
            return found_now, new_m, rng_key

        found, _, _ = outer(jnp.bool_(False), jnp.float64(1.0), key)
        return found

    return body, n_data


# ── Grover with iteration count as a runtime argument (MQT-grounded sizes) ───

def build_grover(n_data):
    def body(k):
        for i in range(n_data):
            qp.Hadamard(wires=i)

        @for_loop(0, k, 1)                              # runtime k (set to MQT count)
        def steps(_):
            _grover_step(n_data)

        steps()
        return qp.probs(wires=list(range(n_data)))

    return body, n_data


# ── runners ─────────────────────────────────────────────────────────────────

def profile_bbht_expected_k(n_data, n_profile, device):
    """Mean total Grover steps E[k] over BBHT's real stochastic loop.

    Native compile (2 native MCX per Grover step -> steps = PauliX_(n_data)/2).
    """
    body, nd = build_bbht(n_data)
    dev = qp.device("lightning.qubit", wires=n_data)
    passes = [f"width-guarded-mcx-decomp{{cost-budget={HUGE}}}"]  # native, for counting
    label = f"PauliX_{n_data}"
    steps = []
    with GateCounterSession(body, dev, timing_model=device,
                            pre_instrumentation_passes=passes) as s:
        for _ in range(n_profile):
            steps.append(s.run().gate_counts.get(label, 0) / 2.0)
    return statistics.mean(steps)


def run_grover(n_data, k, budget, device):
    body, nd = build_grover(n_data)
    dev = qp.device("lightning.qubit", wires=n_data)
    passes = [f"width-guarded-mcx-decomp{{cost-budget={budget}}}"]
    with GateCounterSession(body, dev, jnp.int64(k), timing_model=device,
                            pre_instrumentation_passes=passes) as s:
        return s.run(jnp.int64(k)).runtime_ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-profile", type=int, default=30)
    ap.add_argument("--device", choices=["garnet", "heron"], default="garnet")
    args = ap.parse_args()
    device = IQM_GARNET if args.device == "garnet" else IBM_HERON

    print("=" * 86)
    print(f"  X10 — cost-gated MCX pass on BBHT + MQT-grounded Grover  ({device.name})")
    print("  before = native (budget never fires) ; after = V-chain (budget 0)")
    print("=" * 86)
    print(f"  {'Benchmark':<34} {'c':>3} {'loop':>14} {'t_before':>10} {'t_after':>10} {'speedup':>8}")
    print("  " + "-" * 82)

    # BBHT — real runtime loop; measure E[k] then report expected before/after at k=E[k]
    ek = profile_bbht_expected_k(6, args.n_profile, device)
    kb = max(1, round(ek))
    tb = run_grover(6, kb, HUGE, device)
    ta = run_grover(6, kb, 0, device)
    print(f"  {'BBHT (n_data=6, real dyn loop)':<34} {5:>3} {f'E[k]={ek:.1f}':>14} "
          f"{tb/1e3:>8.2f}us {ta/1e3:>8.2f}us {tb/ta:>7.2f}x")

    # Grover — MQT Bench control counts (diffuser c = nq-2) and iteration counts.
    # MQT nq=7 -> c=5, k=6 ; MQT nq=9 -> c=7, k=12.  n_data = c+1 gives c controls.
    for mqt_nq, c, k in [(7, 5, 6), (9, 7, 12)]:
        n_data = c + 1
        tb = run_grover(n_data, k, HUGE, device)
        ta = run_grover(n_data, k, 0, device)
        print(f"  {f'Grover MQT nq={mqt_nq} (fixed k={k})':<34} {c:>3} "
              f"{f'k={k} (runtime)':>14} {tb/1e3:>8.2f}us {ta/1e3:>8.2f}us {tb/ta:>7.2f}x")

    print()
    print("  BBHT is the genuine runtime-loop case (k measurement-driven, varies per shot).")
    print("  Grover rows use MQT Bench control counts (c=nq-2 diffuser) and iteration counts")
    print("  (6, 12) with k as a runtime arg -- a fixed-iteration case, reported as such.")


if __name__ == "__main__":
    main()
