#!/usr/bin/env python3
"""X8: End-to-end width-guarded MCX decomposition via the real MLIR pass.

Every number here comes from the real Catalyst compilation pipeline:

  * Circuits use the *native* ``MultiControlledX`` (kept as a single
    ``quantum.custom "PauliX" ctrls(...)`` op in the IR).
  * The ``width-guarded-mcx-decomp`` MLIR pass (C++, in
    mlir/lib/Catalyst/Transforms/WidthGuardedMcxDecompPass.cpp) is inserted into
    the compilation pipeline before the gate-counter pass.  When the V-chain
    width 2N-1 fits ``qubit-budget`` it rewrites the MCX into a clean-ancilla
    Toffoli ladder (allocating N-2 ancillas via quantum.alloc_qb); otherwise it
    leaves the native op in place.
  * The ``gate-counter-instrumentation`` MLIR pass counts the *resulting*
    primitives on each real execution (carrying true stochastic trip counts for
    the dynamic loops).
  * ``runtime_model`` converts those counts to modeled device runtime (ns) —
    a native multi-controlled X is costed at its ancilla-free O(N^2) cost, the
    V-chain at its 2N-3 Toffolis.

before = pass with qubit-budget=0  (never fires -> native ancilla-free MCX)
after  = pass with qubit-budget=Q  (fires -> V-chain where 2N-1 <= Q)

The pass is the ONLY difference between before and after: same circuit, same
pipeline, only the budget parameter changes.

Usage::
    python3 run_x8_width_guard_e2e.py [--q-dev 20] [--n-profile 25]
"""

from __future__ import annotations

import argparse
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

_LAMBDA = 6.0 / 5.0


# ── benchmark circuit builders (all use NATIVE MultiControlledX) ─────────────
# Each returns (body_fn, n_ctrl, total_wires, n_profile, is_dynamic).
# total_wires is sized for the V-chain (2N-1 around the MCX) so both budgets run
# on the same device.

def build_synthetic(N=6):
    def body():
        for i in range(N):
            qp.Hadamard(wires=i)
        qp.MultiControlledX(wires=list(range(N)) + [N])
        return qp.probs(wires=[N])
    return body, N, 2 * N - 1, 1, False


def _mcz_native(n_data):
    """MCZ on all n_data qubits via H . MCX . H (native MCX, n_data-1 controls)."""
    tgt = n_data - 1
    qp.Hadamard(wires=tgt)
    qp.MultiControlledX(wires=list(range(n_data - 1)) + [tgt])
    qp.Hadamard(wires=tgt)


def build_grover(n_data=6, k=2):
    """Static Grover: k steps, oracle+diffuser MCZ (stand-in for Grover-SAT)."""
    N = n_data - 1

    def body():
        for i in range(n_data):
            qp.Hadamard(wires=i)
        for _ in range(k):
            _mcz_native(n_data)                       # oracle
            for i in range(n_data):
                qp.Hadamard(wires=i); qp.PauliX(wires=i)
            _mcz_native(n_data)                       # diffuser
            for i in range(n_data):
                qp.PauliX(wires=i); qp.Hadamard(wires=i)
        return qp.probs(wires=list(range(n_data)))
    return body, N, 2 * n_data - 3, 1, False


def build_bbht(n_data=6):
    """Real BBHT search with native MCX oracle/diffuser (dynamic loops)."""
    N = n_data - 1
    n_space = jnp.int64(2 ** n_data)

    def body():
        key = jax.random.PRNGKey(0)

        @while_loop(lambda found, _m, _k: ~found)
        def bbht_loop(found, m, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            kk = jax.random.randint(subkey, shape=(), minval=jnp.int64(1),
                                    maxval=jnp.int64(m) + 1)
            for i in range(n_data):
                qp.Hadamard(wires=i)

            @for_loop(0, kk, 1)
            def grover_step(_):
                _mcz_native(n_data)
                for i in range(n_data):
                    qp.Hadamard(wires=i); qp.PauliX(wires=i)
                _mcz_native(n_data)
                for i in range(n_data):
                    qp.PauliX(wires=i); qp.Hadamard(wires=i)

            grover_step()
            bits = jnp.zeros(n_data, dtype=jnp.int64)
            for i in range(n_data):
                m_i = measure(i, reset=True)
                bits = bits.at[i].set(jnp.int64(m_i))
            found_now = jnp.all(bits == 1)
            new_m = jnp.minimum(jnp.float64(_LAMBDA) * m,
                                jnp.sqrt(jnp.float64(n_space)))
            return found_now, new_m, rng_key

        found, _, _ = bbht_loop(jnp.bool_(False), jnp.float64(1.0), key)
        return found
    return body, N, 2 * n_data - 3, 25, True


def build_nested(n_data=5):
    """Nested: outer BBHT-style search (native MCX) with an inner RUS gadget."""
    N = n_data - 1
    n_space = jnp.int64(2 ** n_data)
    anc = 2 * n_data - 3          # RUS ancilla placed above the V-chain region

    def body():
        key = jax.random.PRNGKey(0)

        @while_loop(lambda found, _m, _k: ~found)
        def outer(found, m, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            kk = jax.random.randint(subkey, shape=(), minval=jnp.int64(1),
                                    maxval=jnp.int64(m) + 1)
            for i in range(n_data):
                qp.Hadamard(wires=i)

            @for_loop(0, kk, 1)
            def step(_):
                _mcz_native(n_data)                    # oracle (native MCX)
                for i in range(n_data):
                    qp.Hadamard(wires=i); qp.PauliX(wires=i)
                _mcz_native(n_data)                    # diffuser
                for i in range(n_data):
                    qp.PauliX(wires=i); qp.Hadamard(wires=i)

            step()

            # inner RUS gadget on the ancilla wire
            @while_loop(lambda s: s == 0)
            def rus(s):
                qp.Hadamard(wires=anc)
                qp.CNOT(wires=[0, anc])
                qp.T(wires=anc)
                qp.CNOT(wires=[0, anc])
                qp.Hadamard(wires=anc)
                mm = measure(anc, reset=True)
                return jnp.int64(mm)
            rus(jnp.int64(0))

            bits = jnp.zeros(n_data, dtype=jnp.int64)
            for i in range(n_data):
                m_i = measure(i, reset=True)
                bits = bits.at[i].set(jnp.int64(m_i))
            found_now = jnp.all(bits == 1)
            new_m = jnp.minimum(jnp.float64(_LAMBDA) * m,
                                jnp.sqrt(jnp.float64(n_space)))
            return found_now, new_m, rng_key

        found, _, _ = outer(jnp.bool_(False), jnp.float64(1.0), key)
        return found
    return body, N, anc + 1, 20, True


BENCHMARKS = {
    "Synthetic MCX (N=6)":        build_synthetic,
    "Grover (n_data=6, k=2)":     build_grover,
    "BBHT (n_data=6)":            build_bbht,
    "Nested RUS-BBHT (n_data=5)": build_nested,
}


# ── profiling ────────────────────────────────────────────────────────────────

def profile(body, total_wires, budget, n_profile, device):
    dev = qp.device("lightning.qubit", wires=total_wires)
    passes = [f"width-guarded-mcx-decomp{{qubit-budget={budget}}}"]
    runtimes = []
    with GateCounterSession(body, dev, timing_model=device,
                            pre_instrumentation_passes=passes) as sess:
        for _ in range(n_profile):
            runtimes.append(sess.run().runtime_ns)
    return runtimes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-dev", type=int, default=20)
    ap.add_argument("--device", choices=["garnet", "heron"], default="garnet")
    args = ap.parse_args()
    device = IQM_GARNET if args.device == "garnet" else IBM_HERON
    Q = args.q_dev

    print("=" * 90)
    print("  X8 — End-to-end width-guarded MCX decomposition via the real MLIR pass")
    print(f"  device={device.name}  Q_dev={Q}   before: qubit-budget=0   after: qubit-budget={Q}")
    print("  (native MCX -> width-guarded-mcx-decomp pass -> gate-counter pass -> timing model)")
    print("=" * 90)
    print(f"  {'Benchmark':<28} {'N':>3} {'W_af':>5} {'W_vc':>5} {'fits':>5} "
          f"| {'t_before':>10} {'t_after':>10} {'speedup':>8}")
    print("  " + "-" * 84)

    for name, builder in BENCHMARKS.items():
        body, N, wires, n_profile, _dyn = builder()
        w_af, w_vc = N + 1, 2 * N - 1
        fits = w_vc <= Q
        before = profile(body, wires, 0, n_profile, device)
        after = profile(body, wires, Q if fits else 0, n_profile, device)
        tb, ta = statistics.mean(before) / 1e3, statistics.mean(after) / 1e3
        spd = tb / ta if ta > 0 else float("inf")
        print(f"  {name:<28} {N:>3} {w_af:>5} {w_vc:>5} {('yes' if fits else 'NO'):>5} "
              f"| {tb:>9.2f}us {ta:>9.2f}us {spd:>7.2f}x")

    print("\n  All gate counts come from the real gate-counter MLIR pass over circuits")
    print("  rewritten by the real width-guarded-mcx-decomp MLIR pass; runtime is the")
    print("  timing-model output (modeled QPU duration), never wall-clock.")


if __name__ == "__main__":
    main()
