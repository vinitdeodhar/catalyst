#!/usr/bin/env python3
"""E3 (revised) — Depth bounding: calibration and benefit evaluation.

Paper claims (Section 7.3, E3):
  1. The compiler-derived MAX_ITER = ceil(E[k] + c·σ) produces a failure
     probability that matches the geometric-distribution prediction within
     binomial 95% confidence intervals — the bound is statistically calibrated.
  2. The compiler-derived bound (c=3) uses 50–90% less worst-case circuit depth
     than a conservative programmer guess (MAX_ITER=50), at the same or lower
     failure probability.
  3. Both E[k] and σ are derived exclusively from the profile-guided gate-counter
     estimator — no standard compiler pass can compute them.

Method:
  Phase 1 — Profile (N_PROFILE shots, unbounded circuits):
    Collect trip counts, compute mean_k, std_k, p̂ = 1/mean_k.
  Phase 2 — Evaluate (N_EVAL shots per bound):
    For c ∈ {1,2,3,4}: MAX_ITER = ceil(mean_k + c·std_k).
    Run N_EVAL bounded shots, count failures (loop hit MAX_ITER without success).
    Compare: predicted_fail = (1-p̂)^MAX_ITER  vs.  observed_fail = failures/N_EVAL.
  Report: calibration table + depth savings vs. conservative MAX_ITER=50.

Circuits: coin-flip (1 qubit), RUS (2 qubits).

Usage:
    python3 run_e3_depth_bounding.py [--n-profile N] [--n-eval N]
                                     [--circuit {coin-flip,RUS,both}]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))
from catalyst import measure, qjit, while_loop

# ── config ─────────────────────────────────────────────────────────────────────

CONSERVATIVE_MAX_ITER = 50    # baseline: cautious programmer's hardcoded guess
C_VALUES = [1, 2, 3, 4]      # c·σ values to add to E[k]

# Number of quantum gates in the loop body (for worst-case depth reporting).
# Counts 1Q and 2Q gates + mid-circuit measure; does not count the conditional
# PauliX from reset=True (it is inside scf.if, counted as 1 for worst-case).
BODY_DEPTH = {
    "coin-flip": 3,   # H + measure + conditional-X
    "RUS":       7,   # 2×H + 2×CNOT + T + measure + conditional-X
}

# ── unbounded circuit factories ────────────────────────────────────────────────

def make_coin_flip_unbounded():
    """Returns compiled coin-flip circuit; output = trip count (int64)."""
    @qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        @while_loop(lambda count, result: result == 0)
        def flip(count, result):
            qp.Hadamard(wires=0)
            m = measure(0, reset=True)
            return count + jnp.int64(1), jnp.int64(m)
        count, _ = flip(jnp.int64(0), jnp.int64(0))
        return count
    return circuit


def make_rus_unbounded():
    """Returns compiled RUS circuit; output = trip count (int64)."""
    target, ancilla = 0, 1

    @qjit
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        qp.Hadamard(wires=target)

        @while_loop(lambda count, s: s == 0)
        def attempt(count, success):
            qp.Hadamard(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.T(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.Hadamard(wires=ancilla)
            m = measure(ancilla, reset=True)
            return count + jnp.int64(1), jnp.int64(m)

        count, _ = attempt(jnp.int64(0), jnp.int64(0))
        return count
    return circuit


# ── bounded circuit factories ──────────────────────────────────────────────────

def make_coin_flip_bounded(max_iter: int):
    """Returns compiled coin-flip circuit bounded at max_iter.

    Output: (trip_count, success_flag).
    success_flag == 0  → loop hit max_iter without exiting (failure).
    success_flag == 1  → loop exited normally (success).
    """
    @qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        @while_loop(lambda count, result: (result == 0) & (count < max_iter))
        def flip(count, result):
            qp.Hadamard(wires=0)
            m = measure(0, reset=True)
            return count + jnp.int64(1), jnp.int64(m)
        count, result = flip(jnp.int64(0), jnp.int64(0))
        return count, result
    return circuit


def make_rus_bounded(max_iter: int):
    """Returns compiled RUS circuit bounded at max_iter.

    Output: (trip_count, success_flag).
    success_flag == 0  → loop hit max_iter without exiting (failure).
    success_flag == 1  → loop exited normally (success).
    """
    target, ancilla = 0, 1

    @qjit
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        qp.Hadamard(wires=target)

        @while_loop(lambda count, s: (s == 0) & (count < max_iter))
        def attempt(count, success):
            qp.Hadamard(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.T(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.Hadamard(wires=ancilla)
            m = measure(ancilla, reset=True)
            return count + jnp.int64(1), jnp.int64(m)

        count, success = attempt(jnp.int64(0), jnp.int64(0))
        return count, success
    return circuit


# ── statistics ─────────────────────────────────────────────────────────────────

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _wilson_ci_95(k: int, n: int) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z = 1.96
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── phase 1: profile ───────────────────────────────────────────────────────────

def profile_circuit(circuit_fn, n_runs: int, label: str) -> List[int]:
    """Run circuit n_runs times and collect trip counts.

    Circuit must return an int64 scalar (trip count) or a tuple whose first
    element is the trip count.
    """
    print(f"  [{label}] compiling…", end=" ", flush=True)
    circuit_fn()   # trigger compilation
    print(f"profiling {n_runs} shots", end="", flush=True)

    trip_counts = []
    for i in range(n_runs):
        out = circuit_fn()
        # Handle both scalar output and tuple output.
        try:
            trip = int(out)
        except TypeError:
            trip = int(out[0])
        trip_counts.append(max(trip, 1))
        if (i + 1) % 50 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done.")
    return trip_counts


# ── phase 2: bounded evaluation ────────────────────────────────────────────────

def evaluate_bounded(bounded_fn, n_runs: int) -> Tuple[int, int]:
    """Run bounded circuit n_runs times.

    Returns (n_success, n_failure) where failure = loop hit MAX_ITER.
    Circuit must return (trip_count, success_flag) with success_flag ∈ {0, 1}.
    """
    n_success = n_failure = 0
    for _ in range(n_runs):
        _, success = bounded_fn()
        if int(success) == 1:
            n_success += 1
        else:
            n_failure += 1
    return n_success, n_failure


# ── experiment driver ──────────────────────────────────────────────────────────

def run_experiment(name: str,
                   unbounded_factory,
                   bounded_factory,
                   n_profile: int,
                   n_eval: int) -> Dict:
    """Run the full profile + evaluate experiment for one circuit."""

    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    # ── Phase 1: profile ───────────────────────────────────────────────────
    trip_counts = profile_circuit(unbounded_factory(), n_profile, name)
    mean_k = _mean(trip_counts)
    std_k  = _std(trip_counts)
    p_hat  = 1.0 / mean_k if mean_k > 0 else 0.0

    print(f"\n  Fitted distribution ({n_profile} shots):")
    print(f"    mean_k = {mean_k:.2f}   std_k = {std_k:.2f}   "
          f"p̂ = {p_hat:.4f}   (geometric MLE: 1/p̂ = {1/p_hat:.2f})")

    body_d          = BODY_DEPTH[name]
    baseline_depth  = CONSERVATIVE_MAX_ITER * body_d
    baseline_pred   = (1.0 - p_hat) ** CONSERVATIVE_MAX_ITER

    print(f"\n  Conservative baseline: MAX_ITER={CONSERVATIVE_MAX_ITER}  "
          f"worst-case depth={baseline_depth}  pred_fail={baseline_pred*100:.3f}%")

    # ── Phase 2: evaluate each c ────────────────────────────────────────────
    print(f"\n  Calibration table  ({n_eval} shots per bound)\n")
    hdr = (f"  {'c':>3}  {'MAX_ITER':>8}  {'pred_fail%':>10}  "
           f"{'obs_fail%':>10}  {'95% CI':>18}  "
           f"{'wc_depth':>8}  {'depth_saved':>11}")
    print(hdr)
    print("  " + "─" * 74)

    rows = []
    for c in C_VALUES:
        max_iter    = max(1, math.ceil(mean_k + c * std_k))
        pred_fail   = (1.0 - p_hat) ** max_iter

        # compile + warm-up
        print(f"  {c:>3}  {max_iter:>8}  {pred_fail*100:>10.2f}%", end="  ", flush=True)
        bounded_fn = bounded_factory(max_iter)
        bounded_fn()   # warm-up

        n_succ, n_fail = evaluate_bounded(bounded_fn, n_eval)
        obs_fail = n_fail / n_eval
        lo, hi   = _wilson_ci_95(n_fail, n_eval)

        wc_depth    = max_iter * body_d
        depth_saved = (1.0 - max_iter / CONSERVATIVE_MAX_ITER) * 100.0

        # calibrated if true failure rate falls within CI
        calibrated = lo <= pred_fail <= hi
        flag = "✓" if calibrated else "⚠"

        print(f"{obs_fail*100:>10.2f}%  "
              f"[{lo*100:5.1f}%,{hi*100:5.1f}%]  "
              f"{wc_depth:>8}  {depth_saved:>10.1f}%  {flag}")

        rows.append({
            "c":           c,
            "max_iter":    max_iter,
            "pred_fail":   pred_fail,
            "obs_fail":    obs_fail,
            "ci_lo":       lo,
            "ci_hi":       hi,
            "wc_depth":    wc_depth,
            "depth_saved": depth_saved / 100.0,
            "calibrated":  calibrated,
            "n_succ":      n_succ,
            "n_fail":      n_fail,
        })

    return {
        "circuit":    name,
        "mean_k":     mean_k,
        "std_k":      std_k,
        "p_hat":      p_hat,
        "body_depth": body_d,
        "rows":       rows,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E3 depth bounding experiment")
    parser.add_argument("--n-profile", type=int, default=200,
                        help="Profiling shots per circuit (default: 200)")
    parser.add_argument("--n-eval",    type=int, default=2000,
                        help="Evaluation shots per (circuit, c) pair (default: 2000)")
    parser.add_argument("--circuit",   choices=["coin-flip", "RUS", "both"],
                        default="both")
    args = parser.parse_args()

    print("=" * 72)
    print("  E3 (revised) — Depth Bounding: Calibration and Benefit")
    print(f"  n_profile={args.n_profile}   n_eval={args.n_eval}   "
          f"conservative_max={CONSERVATIVE_MAX_ITER}")
    print(f"  ✓ = predicted failure rate falls within observed 95% CI")
    print("=" * 72)

    circuit_specs = []
    if args.circuit in ("coin-flip", "both"):
        circuit_specs.append(("coin-flip",
                               make_coin_flip_unbounded,
                               make_coin_flip_bounded))
    if args.circuit in ("RUS", "both"):
        circuit_specs.append(("RUS",
                               make_rus_unbounded,
                               make_rus_bounded))

    results = []
    for name, unbounded_fac, bounded_fac in circuit_specs:
        r = run_experiment(name, unbounded_fac, bounded_fac,
                           args.n_profile, args.n_eval)
        results.append(r)

    # ── cross-circuit summary ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY — compiler bound (c=3) vs. conservative MAX_ITER=50")
    print(f"{'='*72}")
    print(f"  {'Circuit':<12}  {'p̂':>6}  {'E[k]':>6}  {'σ':>6}  "
          f"{'MAX_ITER':>8}  {'pred%':>7}  {'obs%':>7}  {'depth_saved':>11}  {'calib':>6}")
    print("  " + "─" * 70)
    for r in results:
        row3 = next(x for x in r["rows"] if x["c"] == 3)
        flag = "✓" if row3["calibrated"] else "⚠"
        print(f"  {r['circuit']:<12}  {r['p_hat']:>6.4f}  {r['mean_k']:>6.2f}  "
              f"{r['std_k']:>6.2f}  {row3['max_iter']:>8}  "
              f"{row3['pred_fail']*100:>6.1f}%  {row3['obs_fail']*100:>6.1f}%  "
              f"{row3['depth_saved']*100:>10.0f}%  {flag:>6}")
    print()
    print("  Depth saved = 1 − MAX_ITER(c=3) / CONSERVATIVE_MAX_ITER")
    print("  Calibrated  = predicted failure rate lies within 95% binomial CI")
    print()


if __name__ == "__main__":
    main()
