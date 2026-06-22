#!/usr/bin/env python3
# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runner for dynamic-loop benchmarks with resource estimation.

Usage::

    python run_dynamic_benchmarks.py [--n-rus N] [--n-bbht N] [--n-qpe N] [--json]

    --n-rus N     Number of qubits for the RUS benchmark (default: 2)
    --n-bbht N    Number of qubits for the BBHT benchmark (default: 3)
    --n-qpe N     Precision bits for the iterative QPE benchmark (default: 4)
    --json        Emit machine-readable JSON instead of the human summary

Each benchmark is:
  1. Compiled and executed once via @qjit on lightning.qubit.
  2. Analysed by the ``resource-analysis`` MLIR pass (static).
  3. Annotated with expected iteration counts (analytical) to produce
     estimated total gate counts.

For the RUS circuit the expected iteration count is 1/p = 2 (Bernoulli
with p=1/2).  For BBHT it is ⌈π/4·√(2^n)⌉ total Grover iterations
(the classical BBHT expected-cost theorem).

Why runtime instrumentation would be needed
-------------------------------------------
The static pass reports per-iteration costs for the dynamic loops in
these circuits, but it cannot know trip counts because:
  • RUS: trip count depends on quantum measurement outcomes (random).
  • BBHT: trip count depends on quantum measurements AND a classically
    random k sampled inside the loop.
Only a runtime counter (incrementing on every gate application during
actual execution) can record the exact count for a given run.  The
``expected_iters`` supplied here are analytical expectations, not
observed values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pennylane as qp
from jax.core import ShapedArray

# Make sure the benchmark package is importable when run from the benchmark/ dir.
sys.path.insert(0, str(Path(__file__).parent))

from catalyst import qjit
from dynamic_resource_estimator import DynamicResourceEstimator

# ── Analytical expected-iteration helpers ──────────────────────────────────

def rus_expected_iters() -> int:
    """E[iters] for RUS: p(success) = (1 - 1/√2)/2 ≈ 14.6 % → ~7 iters."""
    return 7


def qpe_expected_total_crz(n_bits: int) -> int:
    """Total CRZ applications in iterative QPE: Σ 2^k for k=0..n_bits-1 = 2^n_bits - 1."""
    return 2**n_bits - 1


def bbht_expected_grover_iters(n_qubits: int) -> int:
    """Expected total Grover iterations for BBHT on 2^n states.

    BBHT terminates in expected O(√N) Grover iterations.
    The classical constant gives ⌈π/4·√N⌉.
    """
    N = 2 ** n_qubits
    return max(1, math.ceil(math.pi / 4 * math.sqrt(N)))


# ── Benchmark helpers ──────────────────────────────────────────────────────

def _separator(title: str) -> str:
    bar = "─" * 60
    return f"\n{bar}\n  {title}\n{bar}"


def run_rus(n_qubits: int, est: DynamicResourceEstimator):
    """Compile, run, and analyse the RUS T-gate benchmark."""
    from catalyst_benchmark.test_cases.rus_catalyst import (
        EXPECTED_ITERATIONS,
        ProblemRUS,
        qcompile,
        workflow,
    )

    print(_separator(f"RUS T-gate  (N={n_qubits} qubits)"))
    dev = qp.device("lightning.qubit", wires=n_qubits)
    p = ProblemRUS(dev)

    @qjit
    def rus_main():
        qcompile(p, None)
        return workflow(p, None)

    # ── Execute ────────────────────────────────────────────────────────────
    print("  Running @qjit circuit … ", end="", flush=True)
    result = rus_main()
    print(f"done  →  target probs = {result}")

    # ── Static analysis ────────────────────────────────────────────────────
    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(rus_main)
    print("done")

    # Identify the dynamic while-loop body (name contains "while_loop").
    dyn_loops = report.dynamic_loops()
    expected_iters = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            expected_iters[child] = EXPECTED_ITERATIONS

    print(report.summary(expected_iters=expected_iters))
    print(f"\n  Expected iterations: {EXPECTED_ITERATIONS}  (success prob ≈ 14.6 %)")

    return report, expected_iters


def run_bbht(n_qubits: int, est: DynamicResourceEstimator):
    """Compile, run, and analyse the BBHT Grover benchmark."""
    from catalyst_benchmark.test_cases.bbht_grover_catalyst import (
        ProblemBBHT,
        qcompile,
        workflow,
    )

    print(_separator(f"BBHT Grover  (N={n_qubits} qubits,  search space 2^{n_qubits}={2**n_qubits})"))
    dev = qp.device("lightning.qubit", wires=n_qubits)
    p = ProblemBBHT(dev)

    key = jax.random.PRNGKey(0)

    @qjit
    def bbht_main(k):
        qcompile(p, None)
        return workflow(p, None)

    # ── Execute ────────────────────────────────────────────────────────────
    print("  Running @qjit circuit … ", end="", flush=True)
    result = bbht_main(key)
    print(f"done  →  found all-ones = {result}")

    # ── Static analysis ────────────────────────────────────────────────────
    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(bbht_main, key)
    print("done")

    # BBHT has two dynamic loops:
    #   • outer while_loop  — one "outer iteration" per successful find
    #   • inner for_loop    — the k Grover steps per outer iteration
    # The expected *total Grover iterations* across all outer steps is O(√N).
    total_grover = bbht_expected_grover_iters(n_qubits)
    dyn_loops = report.dynamic_loops()
    expected_iters = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            # Both loops get the same aggregate estimate; the per-iteration
            # cost × expected count gives estimated total gates.
            expected_iters[child] = total_grover

    print(report.summary(expected_iters=expected_iters))
    print(f"\n  Expected total Grover iterations: ~{total_grover}  (O(√{2**n_qubits}))")

    return report, expected_iters


def run_msd(n_magic: int, p_err: float, est: DynamicResourceEstimator):
    """Compile, run, and analyse the MSD benchmark."""
    from catalyst_benchmark.test_cases.msd_catalyst import (
        ProblemMSD,
        expected_iters,
        expected_gate_count,
        qcompile,
        success_prob,
        workflow,
    )

    p_succ = success_prob(n_magic, p_err)
    e_iters = expected_iters(n_magic, p_err)
    print(_separator(
        f"Magic State Distillation  (n_magic={n_magic}, p={p_err:.2f}, "
        f"E[iters]={e_iters:.2f})"
    ))
    dev = qp.device("lightning.qubit", wires=n_magic + 1)
    prob = ProblemMSD(dev, n_magic=n_magic, p_err=p_err)

    key = jax.random.PRNGKey(0)

    @qjit
    def msd_main(k):
        qcompile(prob, None)
        return workflow(prob, None)

    print("  Running @qjit circuit … ", end="", flush=True)
    result = msd_main(key)
    print(f"done  →  success = {bool(result)}")

    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(msd_main, key)
    print("done")

    # The while_loop trip count is geometric with parameter p_succ.
    dyn_loops = report.dynamic_loops()
    expected_iters_dict = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            expected_iters_dict[child] = max(1, round(e_iters))

    print(report.summary(expected_iters=expected_iters_dict))
    print(f"\n  P(success/attempt) = {p_succ:.4f}")
    print(f"  E[iterations]      = {e_iters:.2f}   (geometric distribution)")
    print(f"  E[T gates]         = {expected_gate_count('T(1)', n_magic, p_err):.1f}")
    print(f"  E[CNOT gates]      = {expected_gate_count('CNOT(2)', n_magic, p_err):.1f}")
    print(f"  E[PauliX gates]    = {expected_gate_count('PauliX(1)', n_magic, p_err):.2f}  "
          f"(static worst-case: {n_magic * round(e_iters)})")

    return report, expected_iters_dict


def run_qpe(n_bits: int, est: DynamicResourceEstimator):
    """Compile, run, and analyse the iterative QPE benchmark."""
    from catalyst_benchmark.test_cases.iterative_qpe_catalyst import (
        ProblemQPE,
        expected_total_crz,
        qcompile,
        workflow,
    )

    total_crz = expected_total_crz(n_bits)
    print(_separator(f"Iterative QPE  ({n_bits} bits precision,  total CRZ = 2^{n_bits}-1 = {total_crz})"))
    dev = qp.device("lightning.qubit", wires=2)
    p = ProblemQPE(dev, n_bits=n_bits)

    @qjit
    def qpe_main(nb):
        qcompile(p, None)
        return workflow(p, None)

    # ── Execute ────────────────────────────────────────────────────────────
    print("  Running @qjit circuit … ", end="", flush=True)
    estimated_phase, phase_bits = qpe_main(jnp.int64(n_bits))
    print(f"done  →  phase ≈ {float(estimated_phase):.6f}  (true: 0.125000)")

    # ── Static analysis ────────────────────────────────────────────────────
    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(qpe_main, jnp.int64(n_bits))
    print("done")

    # QPE has two nested dynamic for_loops:
    #   • dyn_for_loop_1 (outer): n_bits rounds
    #   • dyn_for_loop_2 (inner): 2^(n_bits-1-k) per outer round, total = 2^n_bits - 1
    dyn_loops = report.dynamic_loops()
    expected_iters = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            if "for_loop_1" in child or (len(children) == 1):
                expected_iters[child] = n_bits          # outer: n_bits rounds
            else:
                expected_iters[child] = total_crz       # inner: total across all outer rounds

    # Fallback: if loop naming differs, annotate all with n_bits (outer) and inner total.
    if not expected_iters:
        for fn_name, fn in report.entries.items():
            if fn.is_dynamic():
                expected_iters[fn_name] = n_bits

    print(report.summary(expected_iters=expected_iters))
    print(f"\n  Outer loop: {n_bits} rounds (exact)")
    print(f"  Inner loop total CRZ: {total_crz}  (= 2^{n_bits} - 1,  exponential in n_bits)")
    print(f"  Static analysis cannot compute this sum without knowing n_bits at compile time.")

    return report, expected_iters


# ── JSON output ────────────────────────────────────────────────────────────

def _report_to_dict(report, expected_iters):
    totals = report.with_expected_iters(expected_iters)
    return {
        "entries": {
            name: {
                "num_qubits": fn.num_qubits,
                "operations": fn.operations,
                "measurements": fn.measurements,
                "function_calls": fn.function_calls,
                "var_function_calls": {k: "<dynamic>" for k in fn.var_function_calls},
                "is_dynamic": fn.is_dynamic(),
            }
            for name, fn in report.entries.items()
        },
        "expected_iters": expected_iters,
        "estimated_totals": totals,
    }


# ── Static benchmark runners ───────────────────────────────────────────────

def run_qft(n_qubits: int, n_layers: int, est: DynamicResourceEstimator):
    """Compile, run, and analyse the QFT-style static benchmark."""
    from catalyst_benchmark.test_cases.qft_catalyst import ProblemC as ProblemQFT, qcompile, workflow

    print(_separator(f"QFT-style circuit  (N={n_qubits} qubits, layers={n_layers})"))
    dev = qp.device("lightning.qubit", wires=n_qubits)
    p = ProblemQFT(dev, nlayers=n_layers)
    params = p.trial_params(0)

    @qjit
    def qft_main(params: ShapedArray(params.shape, params.dtype)):
        qcompile(p, params)
        return workflow(p, params)

    print("  Running @qjit circuit … ", end="", flush=True)
    result = qft_main(params)
    print(f"done  →  state shape = {result.shape}")

    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(qft_main, params)
    print("done")

    dyn_loops = report.dynamic_loops()
    expected_iters = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            # Grover-oracle-style countdown while_loops have fixed trip counts
            # but appear dynamic to the analysis; annotate with n_qubits as a
            # conservative estimate so the summary does not leave them blank.
            expected_iters[child] = n_qubits

    print(report.summary(expected_iters=expected_iters or None))
    if not dyn_loops:
        print("\n  All loops static — analysis is exact, no estimation needed.")
    return report, expected_iters


def run_grover_static(n_qubits: int, est: DynamicResourceEstimator):
    """Compile, run, and analyse the standard (static) Grover benchmark."""
    from catalyst_benchmark.test_cases.grover_catalyst import ProblemC as ProblemGrover, qcompile, workflow

    print(_separator(f"Grover search  (N={n_qubits} qubits)"))
    dev = qp.device("lightning.qubit", wires=n_qubits)
    p = ProblemGrover(dev, None)
    weights = p.trial_params(0)

    @qjit
    def grover_main(w: ShapedArray(weights.shape, weights.dtype)):
        qcompile(p, None)
        return workflow(p, w)

    print("  Running @qjit circuit … ", end="", flush=True)
    result = grover_main(weights)
    print(f"done  →  state shape = {result.shape}")

    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(grover_main, weights)
    print("done")

    # The oracle uses a while_loop countdown (while i >= 0: i -= 1).
    # This has a fixed trip count (len(CLAUSE_LIST)) but appears dynamic to
    # the analysis since it is a while_loop.  Annotate with p.nlayers as an
    # approximation so the summary can show estimated totals.
    dyn_loops = report.dynamic_loops()
    expected_iters = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            expected_iters[child] = len(p.CLAUSE_LIST)

    print(report.summary(expected_iters=expected_iters or None))
    if not dyn_loops:
        print("\n  All loops static — analysis is exact, no estimation needed.")
    else:
        print(f"\n  Note: while_loop in oracle is a countdown with fixed trip count "
              f"= {len(p.CLAUSE_LIST)}; classified dynamic by analysis (conservative).")
    return report, expected_iters


def run_adjoint(est: DynamicResourceEstimator):
    """Compile, run, and analyse the adjoint gate circuit."""
    from catalyst_benchmark.test_cases.adjoint_circuit_catalyst import ProblemAdjoint, qcompile, workflow

    print(_separator("Adjoint gate circuit  (N=1 qubit, static)"))
    dev = qp.device("lightning.qubit", wires=1)
    p = ProblemAdjoint(dev)

    @qjit
    def adjoint_main():
        qcompile(p, None)
        return workflow(p, None)

    print("  Running @qjit circuit … ", end="", flush=True)
    result = adjoint_main()
    print(f"done  →  probs = {result}  (expect [1.0, 0.0])")

    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(adjoint_main)
    print("done")

    print(report.summary())
    print("\n  Adjoint(T) should appear in analysis output — validates isAdjoint tracking.")
    return report, {}


def run_coin_flip(est: DynamicResourceEstimator):
    """Compile, run, and analyse the coin-flip accumulator benchmark."""
    from catalyst_benchmark.test_cases.coin_flip_catalyst import (
        EXPECTED_ITERATIONS,
        ProblemCoinFlip,
        qcompile,
        workflow,
    )

    print(_separator("Coin-flip accumulator  (N=1 qubit, Geometric(p=0.5))"))
    dev = qp.device("lightning.qubit", wires=1)
    p = ProblemCoinFlip(dev)

    @qjit
    def coin_flip_main():
        qcompile(p, None)
        return workflow(p, None)

    print("  Running @qjit circuit … ", end="", flush=True)
    result = coin_flip_main()
    print(f"done  →  flip count = {int(result)}")

    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(coin_flip_main)
    print("done")

    dyn_loops = report.dynamic_loops()
    expected_iters = {}
    for _parent, children in dyn_loops.items():
        for child in children:
            expected_iters[child] = EXPECTED_ITERATIONS

    print(report.summary(expected_iters=expected_iters))
    print(f"\n  Expected iterations: {EXPECTED_ITERATIONS}  (p=0.5 → E[k] = 2 exactly)")
    return report, expected_iters


def run_nested_rus_bbht(n_data: int, est: DynamicResourceEstimator):
    """Compile, run, and analyse the nested RUS-in-BBHT benchmark."""
    from catalyst_benchmark.test_cases.nested_rus_bbht_catalyst import (
        EXPECTED_INNER_ITERATIONS,
        EXPECTED_OUTER_ITERATIONS,
        ProblemNestedRUSBBHT,
        qcompile,
        workflow,
    )

    n_qubits = n_data + 1   # ancilla wire
    print(_separator(
        f"Nested RUS-in-BBHT  (n_data={n_data}, search space 2^{n_data}={2**n_data})"
    ))
    dev = qp.device("lightning.qubit", wires=n_qubits)
    p = ProblemNestedRUSBBHT(dev)

    @qjit
    def nested_main():
        qcompile(p, None)
        return workflow(p, None)

    print("  Running @qjit circuit … ", end="", flush=True)
    found, n_attempts = nested_main()
    print(f"done  →  found={bool(found)}, attempts={int(n_attempts)}")

    print("  Running resource-analysis pass … ", end="", flush=True)
    report = est.analyse(nested_main)
    print("done")

    # Two dynamic loops: outer search (while) and inner RUS oracle (while).
    # Assign expected iterations by loop name: outer gets EXPECTED_OUTER_ITERATIONS,
    # inner gets EXPECTED_INNER_ITERATIONS.  Since nested naming may vary, assign
    # the smaller estimate (inner) to the deeper loop and larger to the outer.
    dyn_loops = report.dynamic_loops()
    all_dyn_children: list[str] = []
    for _parent, children in dyn_loops.items():
        all_dyn_children.extend(children)
    all_dyn_children = sorted(set(all_dyn_children))

    expected_iters = {}
    for i, child in enumerate(all_dyn_children):
        # Loop names are dyn_while_loop_1 (outer), dyn_while_loop_2 (inner).
        if i == 0:
            expected_iters[child] = EXPECTED_OUTER_ITERATIONS
        else:
            expected_iters[child] = EXPECTED_INNER_ITERATIONS

    print(report.summary(expected_iters=expected_iters))
    print(f"\n  Outer search E[iters] ≈ {EXPECTED_OUTER_ITERATIONS}  (O(√{2**n_data}))")
    print(f"  Inner RUS    E[iters] ≈ {EXPECTED_INNER_ITERATIONS}  (p ≈ 0.5)")
    print("  Two-level while→while nesting — tests getFlattenedResource depth.")
    return report, expected_iters


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dynamic benchmark runner")
    # ── original dynamic benchmarks ────────────────────────────────────────
    parser.add_argument("--n-rus", type=int, default=2, metavar="N",
                        help="Qubits for RUS benchmark (default: 2)")
    parser.add_argument("--n-bbht", type=int, default=3, metavar="N",
                        help="Qubits for BBHT benchmark (default: 3)")
    parser.add_argument("--n-qpe", type=int, default=4, metavar="N",
                        help="Precision bits for iterative QPE benchmark (default: 4)")
    parser.add_argument("--n-msd", type=int, default=7, metavar="N",
                        help="Magic ancillae for MSD benchmark (default: 7)")
    parser.add_argument("--p-msd", type=float, default=0.1, metavar="P",
                        help="Physical error rate for MSD benchmark (default: 0.1)")
    # ── static baselines ───────────────────────────────────────────────────
    parser.add_argument("--n-qft", type=int, default=4, metavar="N",
                        help="Qubits for QFT-style circuit (default: 4)")
    parser.add_argument("--n-qft-layers", type=int, default=1, metavar="L",
                        help="Layers for QFT-style circuit (default: 1)")
    parser.add_argument("--n-grover", type=int, default=7, metavar="N",
                        help="Qubits for static Grover benchmark (default: 7; must satisfy (N-3)%%2==0, min useful N=7)")
    # ── new dynamic benchmarks ─────────────────────────────────────────────
    parser.add_argument("--n-nested", type=int, default=2, metavar="N",
                        help="Data qubits for nested RUS-in-BBHT (default: 2, total wires=3)")
    # ── flags ──────────────────────────────────────────────────────────────
    parser.add_argument("--skip-static", action="store_true",
                        help="Skip the static baseline benchmarks (QFT, Grover, Adjoint)")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON")
    args = parser.parse_args()

    est = DynamicResourceEstimator()
    results = {}

    # ── Static baselines ───────────────────────────────────────────────────
    if not args.skip_static:
        try:
            r, ei = run_qft(args.n_qft, args.n_qft_layers, est)
            results["qft"] = _report_to_dict(r, ei)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n  [QFT] failed: {exc}", file=sys.stderr)

        try:
            r, ei = run_grover_static(args.n_grover, est)
            results["grover_static"] = _report_to_dict(r, ei)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n  [Grover-static] failed: {exc}", file=sys.stderr)

        try:
            r, ei = run_adjoint(est)
            results["adjoint"] = _report_to_dict(r, ei)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n  [Adjoint] failed: {exc}", file=sys.stderr)

    # ── Original dynamic benchmarks ────────────────────────────────────────
    try:
        rus_report, rus_iters = run_rus(args.n_rus, est)
        results["rus"] = _report_to_dict(rus_report, rus_iters)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n  [RUS] failed: {exc}", file=sys.stderr)

    try:
        bbht_report, bbht_iters = run_bbht(args.n_bbht, est)
        results["bbht"] = _report_to_dict(bbht_report, bbht_iters)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n  [BBHT] failed: {exc}", file=sys.stderr)

    try:
        qpe_report, qpe_iters = run_qpe(args.n_qpe, est)
        results["qpe"] = _report_to_dict(qpe_report, qpe_iters)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n  [QPE] failed: {exc}", file=sys.stderr)

    try:
        msd_report, msd_iters = run_msd(args.n_msd, args.p_msd, est)
        results["msd"] = _report_to_dict(msd_report, msd_iters)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n  [MSD] failed: {exc}", file=sys.stderr)

    # ── New dynamic benchmarks ─────────────────────────────────────────────
    try:
        r, ei = run_coin_flip(est)
        results["coin_flip"] = _report_to_dict(r, ei)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n  [CoinFlip] failed: {exc}", file=sys.stderr)

    try:
        r, ei = run_nested_rus_bbht(args.n_nested, est)
        results["nested_rus_bbht"] = _report_to_dict(r, ei)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n  [NestedRUSBBHT] failed: {exc}", file=sys.stderr)

    if args.json:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
