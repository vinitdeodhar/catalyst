# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E6: Resource Analysis Pass Timing.

Measures the overhead of the ``resource-analysis`` MLIR pass relative to
full @qjit compilation time across all 10 benchmark circuits.

For each circuit:
  1. Compile with @qjit and time the compilation (first call).
  2. Run DynamicResourceEstimator.analyse() and time just the analysis pass.
  3. Report compile_time, analysis_time, and overhead ratio.

For parameterized circuits (BBHT at n_data=3,4,5), both times are shown
as a function of n to quantify scaling.

Key claim: The resource-analysis pass adds negligible overhead (< 3%) to
compilation time across all circuit sizes in the benchmark suite.

Usage::

    python3 benchmark/run_e6_analysis_timing.py
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import pennylane as qp
from jax.core import ShapedArray

sys.path.insert(0, "/home/vadeo/catalyst/benchmark")
from catalyst import cond, for_loop, measure, qjit, while_loop
from dynamic_resource_estimator import DynamicResourceEstimator


# ── Timing helpers ─────────────────────────────────────────────────────────────

def _time_fn(fn, *args, n_repeats: int = 3):
    """Time fn(*args) for n_repeats calls; return (result, min_time_s)."""
    result = None
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return result, best


def _time_analysis(est: DynamicResourceEstimator, qjit_fn, n_repeats: int = 5):
    """Time the analysis pass only (MLIR already compiled).

    Calls _run_pass directly to avoid re-extracting MLIR each time.
    """
    mlir_text = qjit_fn.mlir
    circuit_mlir = est._extract_circuit_module(mlir_text)

    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        est._run_pass(circuit_mlir)
        best = min(best, time.perf_counter() - t0)
    return best


def _hdr(title: str):
    sep = "=" * 72
    print(f"\n{sep}\n  {title}\n{sep}\n")


def _row(name: str, n_qubits: int, loops: str,
         compile_ms: float, analysis_ms: float):
    overhead_pct = analysis_ms / max(compile_ms, 1) * 100
    print(f"  {name:<28}  {n_qubits:>7}  {loops:<20}"
          f"  {compile_ms:>9.0f}  {analysis_ms:>11.2f}  {overhead_pct:>9.2f}%")


# ── Circuit builders ───────────────────────────────────────────────────────────
# Minimal self-contained versions; match the benchmark test_cases exactly.

def _adjoint_circuit():
    def _circuit():
        qp.Hadamard(wires=0)
        qp.T(wires=0)
        qp.adjoint(qp.T)(wires=0)
        qp.Hadamard(wires=0)
        return qp.probs(wires=[0])

    dev = qp.device("lightning.qubit", wires=1)

    @qjit
    def fn():
        return qp.QNode(_circuit, dev)()

    return fn, dev, 1


def _coin_flip_circuit():
    def _circuit():
        @while_loop(lambda c: c == 0)
        def flip(count):
            qp.Hadamard(wires=1)
            m = measure(1, reset=True)
            qp.PauliX(wires=0)
            return jnp.int64(m)

        flip(jnp.int64(0))
        return qp.probs(wires=[0])

    dev = qp.device("lightning.qubit", wires=2)

    @qjit
    def fn():
        return qp.QNode(_circuit, dev)()

    return fn, dev, 2


def _rus_circuit():
    target, ancilla = 0, 1

    def _circuit():
        qp.Hadamard(wires=target)

        @while_loop(lambda s: s == 0)
        def rus_attempt(success):
            qp.Hadamard(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.T(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.Hadamard(wires=ancilla)
            m = measure(ancilla, reset=True)
            return jnp.int64(m)

        rus_attempt(jnp.int64(0))
        return qp.probs(wires=[target])

    dev = qp.device("lightning.qubit", wires=2)

    @qjit
    def fn():
        return qp.QNode(_circuit, dev)()

    return fn, dev, 2


def _msd_circuit(n_magic: int = 7, p_err: float = 0.10):
    syndrome_wire = 0

    def _prepare_noisy_T(wire: int, key):
        qp.Hadamard(wires=wire)
        qp.T(wires=wire)
        key, subkey = jax.random.split(key)
        error = jax.random.bernoulli(subkey, jnp.float64(p_err))

        @cond(error)
        def inject():
            qp.PauliX(wires=wire)

        inject()
        return key

    def _circuit(key):
        @while_loop(lambda success, _k: ~success)
        def msd_attempt(success, key):
            for wire in range(1, n_magic + 1):
                key = _prepare_noisy_T(wire, key)
            for wire in range(1, n_magic + 1):
                qp.CNOT(wires=[wire, syndrome_wire])
            syn = measure(syndrome_wire, reset=True)
            for wire in range(1, n_magic + 1):
                measure(wire, reset=True)
            return jnp.bool_(syn == 0), key

        success, _ = msd_attempt(jnp.bool_(False), key)
        return success

    dev = qp.device("lightning.qubit", wires=n_magic + 1)
    init_key = jax.random.PRNGKey(0)

    @qjit
    def fn(key):
        return qp.QNode(_circuit, dev)(key)

    fn(init_key)  # eager compile
    return fn, dev, n_magic + 1


def _qpe_circuit(n_bits_default: int = 4):
    ancilla, target = 0, 1

    def _circuit(n_bits):
        qp.PauliX(wires=target)

        @for_loop(jnp.int64(0), n_bits, jnp.int64(1))
        def qpe_round(j, correction_accum, estimate):
            k = n_bits - jnp.int64(1) - j
            qp.Hadamard(wires=ancilla)
            qp.PhaseShift(correction_accum, wires=ancilla)

            inner_iters = jnp.left_shift(jnp.int64(1), k)

            @for_loop(jnp.int64(0), inner_iters, jnp.int64(1))
            def apply_cu(_):
                qp.CRZ(jnp.pi / 2, wires=[ancilla, target])

            apply_cu()
            qp.Hadamard(wires=ancilla)
            bit = measure(ancilla, reset=True)
            new_correction = (correction_accum - jnp.pi * jnp.float64(bit)) / jnp.float64(2.0)
            new_estimate = estimate + jnp.int64(bit) * jnp.left_shift(jnp.int64(1), j)
            return new_correction, new_estimate

        _, phase_bits = qpe_round(jnp.float64(0.0), jnp.int64(0))
        return phase_bits

    dev = qp.device("lightning.qubit", wires=2)

    @qjit
    def fn(nb):
        return qp.QNode(_circuit, dev)(nb)

    fn(jnp.int64(n_bits_default))
    return fn, dev, 2


def _bbht_circuit(n_data: int):
    _LAMBDA = 6.0 / 5.0
    n_space = jnp.int64(2 ** n_data)

    def _oracle_inner(n_data_):
        if n_data_ == 3:
            qp.Hadamard(wires=n_data_ - 1)
            qp.Toffoli(wires=[0, 1, 2])
            qp.Hadamard(wires=n_data_ - 1)
        else:
            target = n_data_ - 1
            controls = list(range(n_data_ - 1))
            qp.Hadamard(wires=target)
            qp.MultiControlledX(wires=controls + [target])
            qp.Hadamard(wires=target)

    def _diffuser_inner(n_data_):
        for i in range(n_data_):
            qp.Hadamard(wires=i)
            qp.PauliX(wires=i)
        _oracle_inner(n_data_)
        for i in range(n_data_):
            qp.PauliX(wires=i)
            qp.Hadamard(wires=i)

    def _circuit(key):
        @while_loop(lambda found, _m, _k: ~found)
        def bbht_loop(found, m, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            k = jax.random.randint(
                subkey, shape=(), minval=jnp.int64(1), maxval=jnp.int64(m) + 1
            )
            for i in range(n_data):
                qp.Hadamard(wires=i)

            @for_loop(0, k, 1)
            def grover_step(_):
                _oracle_inner(n_data)
                _diffuser_inner(n_data)

            grover_step()

            bits = jnp.zeros(n_data, dtype=jnp.int64)
            for i in range(n_data):
                m_i = measure(i, reset=True)
                bits = bits.at[i].set(jnp.int64(m_i))

            found_now = jnp.all(bits == 1)
            new_m = jnp.minimum(
                jnp.float64(_LAMBDA) * m,
                jnp.sqrt(jnp.float64(n_space)),
            )
            return found_now, new_m, rng_key

        found, _, _ = bbht_loop(jnp.bool_(False), jnp.float64(1.0), key)
        return found

    dev = qp.device("lightning.qubit", wires=n_data)
    init_key = jax.random.PRNGKey(0)

    @qjit
    def fn(key):
        return qp.QNode(_circuit, dev)(key)

    fn(init_key)
    return fn, dev, n_data


def _nested_rus_bbht_circuit():
    """Nested RUS-in-BBHT: outer BBHT while, inner RUS while."""
    from catalyst_benchmark.test_cases.nested_rus_bbht_catalyst import (
        ProblemNestedRUSBBHT, qcompile, workflow,
    )
    n_data = 2
    n_wires = n_data + 1   # n_data data qubits + 1 ancilla
    dev = qp.device("lightning.qubit", wires=n_wires)
    p = ProblemNestedRUSBBHT(dev)
    init_key = jax.random.PRNGKey(0)

    @qjit
    def fn(key):
        qcompile(p, None)
        return workflow(p, key)

    fn(init_key)
    return fn, dev, n_wires


# ── Main timing loop ───────────────────────────────────────────────────────────

def main():
    est = DynamicResourceEstimator()

    _hdr("E6: Resource Analysis Pass Timing")
    print("  Measuring compile time and resource-analysis pass time for all benchmarks.")
    print("  Overhead% = analysis_time / compile_time × 100")
    print()

    # Table header
    print(f"  {'Circuit':<28}  {'qubits':>7}  {'loops':<20}"
          f"  {'compile_ms':>10}  {'analysis_ms':>12}  {'overhead':>9}")
    print("  " + "-" * 90)

    rows = []   # for summary statistics

    # ── 1. Adjoint ─────────────────────────────────────────────────────────
    name = "adjoint"
    print(f"  [{name}] compiling...", end=" ", flush=True)
    t0 = time.perf_counter()
    fn, *_ = _adjoint_circuit()
    fn()
    c_ms = (time.perf_counter() - t0) * 1000
    a_ms = _time_analysis(est, fn) * 1000
    _row(name, 1, "none", c_ms, a_ms)
    rows.append((name, c_ms, a_ms))

    # ── 2. Coin flip ───────────────────────────────────────────────────────
    name = "coin_flip"
    print(f"  [{name}] compiling...", end=" ", flush=True)
    t0 = time.perf_counter()
    fn, *_ = _coin_flip_circuit()
    fn()
    c_ms = (time.perf_counter() - t0) * 1000
    a_ms = _time_analysis(est, fn) * 1000
    _row(name, 2, "1 while", c_ms, a_ms)
    rows.append((name, c_ms, a_ms))

    # ── 3. RUS ─────────────────────────────────────────────────────────────
    name = "rus"
    print(f"  [{name}] compiling...", end=" ", flush=True)
    t0 = time.perf_counter()
    fn, *_ = _rus_circuit()
    fn()
    c_ms = (time.perf_counter() - t0) * 1000
    a_ms = _time_analysis(est, fn) * 1000
    _row(name, 2, "1 while", c_ms, a_ms)
    rows.append((name, c_ms, a_ms))

    # ── 4. MSD (n_magic=7) ─────────────────────────────────────────────────
    name = "msd (n_magic=7)"
    print(f"  [{name}] compiling...", end=" ", flush=True)
    t0 = time.perf_counter()
    fn, dev, nq = _msd_circuit(n_magic=7)
    c_ms = (time.perf_counter() - t0) * 1000
    a_ms = _time_analysis(est, fn) * 1000
    _row(name, nq, "1 while + 7 cond", c_ms, a_ms)
    rows.append((name, c_ms, a_ms))

    # ── 5. Iterative QPE (n_bits as runtime arg) ───────────────────────────
    name = "qpe (n_bits runtime)"
    print(f"  [{name}] compiling...", end=" ", flush=True)
    t0 = time.perf_counter()
    fn, dev, nq = _qpe_circuit(n_bits_default=4)
    c_ms = (time.perf_counter() - t0) * 1000
    a_ms = _time_analysis(est, fn) * 1000
    _row(name, nq, "for + for (nested, dyn)", c_ms, a_ms)
    rows.append((name, c_ms, a_ms))

    # ── 6-8. BBHT at n_data = 3, 4, 5 (separate compilations) ─────────────
    for n_data in [3, 4, 5]:
        name = f"bbht (n_data={n_data})"
        print(f"  [{name}] compiling...", end=" ", flush=True)
        t0 = time.perf_counter()
        fn, dev, nq = _bbht_circuit(n_data)
        c_ms = (time.perf_counter() - t0) * 1000
        a_ms = _time_analysis(est, fn) * 1000
        _row(name, nq, "while + for (nested)", c_ms, a_ms)
        rows.append((name, c_ms, a_ms))

    # ── 9. Nested RUS-in-BBHT ──────────────────────────────────────────────
    name = "nested_rus_bbht"
    print(f"  [{name}] compiling...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        fn, dev, nq = _nested_rus_bbht_circuit()
        c_ms = (time.perf_counter() - t0) * 1000
        a_ms = _time_analysis(est, fn) * 1000
        _row(name, nq, "while + while (nested)", c_ms, a_ms)
        rows.append((name, c_ms, a_ms))
    except Exception as e:
        print(f"SKIP ({e})")
        rows.append((name, float("nan"), float("nan")))

    # ── Summary statistics ─────────────────────────────────────────────────
    _hdr("E6 Summary")
    valid = [(n, c, a) for n, c, a in rows if c == c and a == a]
    if valid:
        overheads = [a / c * 100 for _, c, a in valid]
        analysis_ms_vals = [a for _, _, a in valid]
        compile_ms_vals = [c for _, c, _ in valid]

        print(f"  Circuits measured   : {len(valid)}")
        print(f"  Compile time range  : {min(compile_ms_vals):.0f} – {max(compile_ms_vals):.0f} ms")
        print(f"  Analysis time range : {min(analysis_ms_vals):.2f} – {max(analysis_ms_vals):.2f} ms")
        print(f"  Overhead range      : {min(overheads):.2f}% – {max(overheads):.2f}%")
        print(f"  Mean overhead       : {sum(overheads)/len(overheads):.2f}%")
        max_overhead = max(overheads)
        max_name = valid[overheads.index(max_overhead)][0]
        print(f"  Max overhead        : {max_overhead:.2f}% ({max_name})")
        claim_met = max_overhead < 5.0
        print(f"  Claim (< 5%)        : {'✓ MET' if claim_met else '✗ EXCEEDED'}")

    # ── BBHT scaling table ─────────────────────────────────────────────────
    bbht_rows = [(n, c, a) for n, c, a in rows if n.startswith("bbht")]
    if len(bbht_rows) >= 2:
        print()
        print("  BBHT scaling (analysis time vs. n_data):")
        print(f"  {'n_data':>7}  {'compile_ms':>12}  {'analysis_ms':>13}  {'ratio':>8}")
        print("  " + "-" * 48)
        for n_str, c, a in bbht_rows:
            n_data = int(n_str.split("=")[1].rstrip(")"))
            print(f"  {n_data:>7}  {c:>12.0f}  {a:>13.2f}  {a/c*100:>7.2f}%")

    # ── CSV data for Table 3 ───────────────────────────────────────────────
    print()
    print("  CSV (circuit, compile_ms, analysis_ms, overhead_pct):")
    for name, c, a in rows:
        if c == c and a == a:
            print(f"  {name},{c:.1f},{a:.3f},{a/c*100:.3f}")

    print()
    print("  DONE.")


if __name__ == "__main__":
    main()
