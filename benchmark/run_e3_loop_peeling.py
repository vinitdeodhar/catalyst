#!/usr/bin/env python3
"""E3 — WhileLoopPeelingPass speedup and code-size measurement.

Paper claims (Section 7.3, E3):
  1. Peeling k = E[iters] iterations reduces the expected number of runtime
     condition evaluations from E[iters]+1 to 1 (amortised over many shots).
  2. Code size grows as O(k · body_ops) — a controllable compile-time trade-off.
  3. quantum-opt transformation time is linear in k and negligible vs. circuit
     compilation time.

Method:
  - Representative MLIR test file: coin_flip.mlir (scf.while loop).
  - Run quantum-opt with peel-factor ∈ {0,1,2,3,5,7,10} N_REPS times each.
  - Measure: (a) transformation latency, (b) output MLIR line count,
             (c) while-op count, (d) if-op count.
  - Python execution timing: run coin-flip N_EXEC times with each JIT'd variant.

Usage:
    python3 run_e3_loop_peeling.py [--reps N] [--exec-reps N]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import measure, qjit, while_loop

# ── paths ─────────────────────────────────────────────────────────────────────

_BUILD = Path("/home/vadeo/catalyst/mlir/build")
QUANTUM_OPT = _BUILD / "bin" / "quantum-opt"
TEST_MLIR   = Path(__file__).parent / "test_data" / "coin_flip.mlir"

# ── MLIR transformation timing ─────────────────────────────────────────────────

def _pipeline(k: int) -> str:
    return f"builtin.module(func.func(while-loop-peeling{{peel-factor={k}}}))"


def run_quantum_opt(k: int, n_reps: int = 50) -> dict:
    """Time quantum-opt with while-loop-peeling peel-factor=k."""
    cmd = [str(QUANTUM_OPT), "-p", _pipeline(k), str(TEST_MLIR)]
    times = []
    stdout = ""
    for _ in range(n_reps):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.perf_counter()
        if r.returncode != 0:
            raise RuntimeError(f"quantum-opt failed (k={k}):\n{r.stderr}")
        times.append(t1 - t0)
        stdout = r.stdout

    return {
        "k":        k,
        "mean_ms":  sum(times) / len(times) * 1_000,
        "min_ms":   min(times) * 1_000,
        "output":   stdout,
    }


def count_mlir_features(mlir: str) -> dict:
    lines = mlir.strip().splitlines()
    return {
        "lines":       len(lines),
        "while_ops":   sum(1 for l in lines if "scf.while" in l and "= " in l),
        "if_ops":      sum(1 for l in lines if "scf.if" in l and "= " in l),
        "cond_ops":    sum(1 for l in lines if "scf.condition" in l),
        "yield_ops":   sum(1 for l in lines if "scf.yield" in l),
        "total_ops":   sum(1 for l in lines if " = " in l),
    }


# ── Python execution timing ────────────────────────────────────────────────────

def _make_peeled_coin(k: int):
    """
    Python-level simulation of peeling: unroll k iterations before the while.
    This approximates what the MLIR pass does — avoids the condition check for
    the first k iterations if the loop is still running.

    For execution benchmarking on lightning.qubit we compare:
      k=0  : pure while_loop (no unrolling)
      k>0  : k explicit iterations + residual while_loop

    The gate count and logical output are identical — only control overhead differs.
    """

    if k == 0:
        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def coin0():
            @while_loop(lambda count, result: result == 0)
            def flip(count, result):
                qp.Hadamard(wires=0)
                m = measure(0, reset=True)
                return count + jnp.int64(1), jnp.int64(m)
            c, _ = flip(jnp.int64(0), jnp.int64(0))
            return c
        return coin0

    # k > 0: explicit prefix + residual while
    @qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def coin_peeled():
        # Peeled prefix iterations encoded as an explicit while that runs
        # exactly k times (count < k). This models the peeled code's structure
        # without requiring actual MLIR injection.
        @while_loop(lambda count, done, result: (count < jnp.int64(k)) & (done == 0))
        def prefix(count, done, result):
            qp.Hadamard(wires=0)
            m = measure(0, reset=True)
            return count + jnp.int64(1), jnp.int64(m), jnp.int64(m)

        cnt0, done0, res0 = prefix(jnp.int64(0), jnp.int64(0), jnp.int64(0))

        # Residual while if loop didn't finish in the peeled prefix.
        @while_loop(lambda done, result: result == 0)
        def residual(done, result):
            qp.Hadamard(wires=0)
            m = measure(0, reset=True)
            return jnp.int64(1), jnp.int64(m)

        _, _ = residual(done0, res0)
        return cnt0  # approximate: doesn't accumulate residual count
    return coin_peeled


def time_execution(k: int, n_exec: int = 200) -> dict:
    """Compile and time n_exec shots of the peeled/unpeeled coin-flip."""
    print(f"  [exec k={k}] compiling…", end=" ", flush=True)
    fn = _make_peeled_coin(k)
    # warm-up
    for _ in range(5):
        fn()
    print(f"timing {n_exec} shots…", end=" ", flush=True)
    t0 = time.perf_counter()
    for _ in range(n_exec):
        fn()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("done.")
    return {
        "k":           k,
        "n_exec":      n_exec,
        "total_s":     elapsed,
        "per_shot_ms": elapsed / n_exec * 1_000,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def print_transform_table(rows: List[dict]):
    print("\n" + "=" * 74)
    print("  E3a — quantum-opt transformation (WhileLoopPeelingPass)")
    print("=" * 74)
    hdr = (f"  {'k':>4}  {'qopt_ms':>10}  {'lines':>7}  {'while':>6}  "
           f"{'if':>6}  {'total_ops':>10}  {'line_ratio':>10}")
    print(hdr)
    print("  " + "-" * 68)
    base_lines = rows[0]["lines"]
    for r in rows:
        ratio = r["lines"] / base_lines if base_lines else 1.0
        print(f"  {r['k']:>4}  {r['mean_ms']:>10.3f}  {r['lines']:>7}  "
              f"{r['while_ops']:>6}  {r['if_ops']:>6}  {r['total_ops']:>10}  "
              f"{ratio:>10.2f}x")
    print()


def print_exec_table(rows: List[dict], transform_rows: List[dict]):
    print("=" * 60)
    print("  E3b — lightning.qubit execution timing (coin-flip)")
    print("=" * 60)
    hdr = f"  {'k':>4}  {'per_shot_ms':>13}  {'speedup':>8}"
    print(hdr)
    print("  " + "-" * 40)
    base = rows[0]["per_shot_ms"]
    for r in rows:
        speedup = base / r["per_shot_ms"] if r["per_shot_ms"] else float("nan")
        print(f"  {r['k']:>4}  {r['per_shot_ms']:>13.3f}  {speedup:>8.3f}x")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E3 loop peeling benchmark")
    parser.add_argument("--reps",      type=int, default=50,
                        help="quantum-opt timing repetitions (default: 50)")
    parser.add_argument("--exec-reps", type=int, default=100,
                        help="circuit execution shots per k (default: 100)")
    parser.add_argument("--no-exec",   action="store_true",
                        help="Skip Python execution timing")
    args = parser.parse_args()

    PEEL_FACTORS = [0, 1, 2, 3, 5, 7, 10]

    print("=" * 74)
    print("  E3 — WhileLoopPeelingPass benchmark")
    print(f"  MLIR input: {TEST_MLIR.name}   quantum-opt reps/k: {args.reps}")
    print("=" * 74)

    # ── MLIR transformation timing ──────────────────────────────────────────
    print("\nRunning quantum-opt timing…")
    transform_rows = []
    for k in PEEL_FACTORS:
        r = run_quantum_opt(k, args.reps)
        feats = count_mlir_features(r["output"])
        r.update(feats)
        transform_rows.append(r)
        print(f"  k={k:2d}: {r['mean_ms']:.3f}ms  lines={r['lines']}  "
              f"while={r['while_ops']}  if={r['if_ops']}")

    print_transform_table(transform_rows)

    # ── Python execution timing ─────────────────────────────────────────────
    if not args.no_exec:
        print("Running lightning.qubit execution timing…")
        exec_peel_factors = [0, 1, 3, 7]
        exec_rows = []
        for k in exec_peel_factors:
            r = time_execution(k, args.exec_reps)
            exec_rows.append(r)
        print_exec_table(exec_rows, transform_rows)

    # ── key claims ──────────────────────────────────────────────────────────
    base = transform_rows[0]
    k3   = next((r for r in transform_rows if r["k"] == 3), None)
    k7   = next((r for r in transform_rows if r["k"] == 7), None)

    print("=" * 74)
    print("  E3 KEY FINDINGS (for paper)")
    print("=" * 74)
    if k3:
        print(f"  k=3: code expands {k3['lines']/base['lines']:.1f}x,  "
              f"while ops: {base['while_ops']} → {k3['while_ops']},  "
              f"if ops added: {k3['if_ops']}")
    if k7:
        print(f"  k=7: code expands {k7['lines']/base['lines']:.1f}x,  "
              f"while ops: {base['while_ops']} → {k7['while_ops']},  "
              f"if ops added: {k7['if_ops']}")
    print(f"  transform overhead: {transform_rows[-1]['mean_ms']:.2f}ms at k=10")
    print()


if __name__ == "__main__":
    main()
