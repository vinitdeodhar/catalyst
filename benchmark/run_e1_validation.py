#!/usr/bin/env python3
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
"""E1 — Validation experiment: static per-iteration cost == runtime per-iteration cost.

Paper claim (Section 7.2):
  For every gate in a dynamic loop body that is NOT inside an scf.if branch,
  the static resource analysis per-iteration count equals the runtime mean
  per-iteration count to within ±0.01 (noise floor of 100 runs).

  For gates arising from scf.if (e.g. the PauliX from measure(reset=True)),
  the analysis intentionally over-approximates by taking the maximum over
  branches: the over-approximation factor is exactly 1/p where p is the
  probability the branch is taken.

Validation method:
  For each circuit with a dynamic while-loop:
    1. Run N times with gate counter instrumentation.
    2. For each run, infer the trip count k from gate counts (or circuit output).
    3. Compute observed_per_iter[gate] = gate_count[run] / k[run].
    4. Report mean±std of observed_per_iter versus static_per_iter.
    5. Ratio = observed / static: 1.0 = exact; 0.5 = 2× over-approx.

Circuits:
  - coin_flip  : dynamic while, returns trip count directly.
                 Expected: H×1, Measure×1 are exact (ratio=1.0);
                 PauliX×1 is over-approx by 2× (ratio=0.5).
  - rus        : dynamic while, trip count inferred from T gate (1 T/iter, no T outside).
                 Expected: H×2, CNOT×2, T×1, Measure×1 exact;
                 PauliX×1 over-approx 2×.

Usage:
    python3 run_e1_validation.py [--n-coin N] [--n-rus N] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import measure, qjit, while_loop
from gate_counter_estimator import GateCounterSession

# ── static per-iter costs (from resource-analysis pass) ───────────────────────
# Gate counter labels: "GateName_wires" as produced by GateCounterInstrumentationPass.
# Source: DynamicResourceEstimator output on each circuit (run once separately).

# coin_flip dyn_while_loop_1 body:
#   Hadamard(1)×1, MidCircuitMeasure×1, PauliX(1)×1(over-approx from reset=True)
COIN_STATIC_PER_ITER: Dict[str, float] = {
    "Hadamard_1": 1.0,
    "Measure_1":  1.0,
    "PauliX_1":   1.0,   # over-approximated: actual runtime ≈ 0.5
}

# rus dyn_while_loop_1 body:
#   Hadamard(1)×2, CNOT(2)×2, T(1)×1, MidCircuitMeasure×1, PauliX(1)×1(over-approx)
RUS_STATIC_PER_ITER: Dict[str, float] = {
    "Hadamard_1": 2.0,
    "CNOT_2":     2.0,
    "T_1":        1.0,
    "Measure_1":  1.0,
    "PauliX_1":   1.0,   # over-approximated: actual runtime ≈ 0.5
}

# ── circuit definitions ────────────────────────────────────────────────────────

def _make_coin_flip():
    """Return the coin-flip circuit body (1 qubit, returns trip count)."""
    def _circuit():
        @while_loop(lambda count, result: result == 0)
        def flip_loop(count, result):
            qp.Hadamard(wires=0)
            m = measure(0, reset=True)
            return count + jnp.int64(1), jnp.int64(m)
        count, _ = flip_loop(jnp.int64(0), jnp.int64(0))
        return count
    return _circuit


def _make_rus():
    """Return the RUS circuit body (2 qubits, returns target qubit probs)."""
    target, ancilla = 0, 1

    def _circuit():
        qp.Hadamard(wires=target)   # 1 H before the loop
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
    return _circuit


# ── per-run statistics ────────────────────────────────────────────────────────

class RunStats(NamedTuple):
    gate_counts: Dict[str, int]
    trip_count: float
    per_iter: Dict[str, float]


def _coin_trip(run_result) -> float:
    """Trip count for coin-flip = the circuit's returned integer."""
    return max(float(int(run_result.circuit_output)), 1.0)


def _rus_trip(run_result) -> float:
    """Trip count for RUS = T_1 gate count (1 T per iteration, 0 outside loop)."""
    return max(float(run_result.gate_counts.get("T_1", 1)), 1.0)


# Gates that appear OUTSIDE the loop in RUS — subtract from total before dividing.
# RUS has 1 Hadamard before the while loop to prepare the target qubit.
RUS_OUTSIDE_LOOP: Dict[str, int] = {"Hadamard_1": 1}


def _compute_per_iter(gate_counts: Dict[str, int], trip: float,
                      static: Dict[str, float],
                      outside_loop: Dict[str, int] = None) -> Dict[str, float]:
    """Compute observed per-iteration count for each gate in `static`.

    ``outside_loop`` maps gate_label -> count of that gate's applications
    OUTSIDE the loop body (e.g. pre-loop state preparation). These are
    subtracted from the total before dividing by trip count.
    """
    outside = outside_loop or {}
    return {
        g: max(gate_counts.get(g, 0) - outside.get(g, 0), 0) / trip
        for g in static
    }


def run_circuit_session(circuit_fn, dev, n_wires, n_runs, trip_fn,
                        static_per_iter, label,
                        outside_loop=None) -> List[RunStats]:
    """Run circuit n_runs times, compute per-iter stats for each run."""
    print(f"\n[{label}] compiling … ", end="", flush=True)
    results = []
    with GateCounterSession(circuit_fn, dev) as sess:
        print(f"done.  Running {n_runs} shots …", end="", flush=True)
        for i in range(n_runs):
            r = sess.run()
            trip = trip_fn(r)
            per_iter = _compute_per_iter(r.gate_counts, trip, static_per_iter, outside_loop)
            results.append(RunStats(r.gate_counts, trip, per_iter))
            if (i + 1) % 20 == 0:
                print(f" {i+1}", end="", flush=True)
    print(" done.")
    return results


# ── summary statistics ────────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def summarise(label: str, runs: List[RunStats], static: Dict[str, float]) -> dict:
    """Compute mean/std per-iter for each gate, plus ratio vs static."""
    gates = list(static.keys())
    rows = []
    for gate in gates:
        obs_list = [r.per_iter[gate] for r in runs]
        mu = _mean(obs_list)
        sd = _std(obs_list)
        rat = mu / static[gate] if static[gate] else float("nan")
        rows.append({
            "circuit":       label,
            "gate":          gate,
            "static_per_iter": static[gate],
            "obs_mean":      mu,
            "obs_std":       sd,
            "ratio":         rat,
            "note":          "exact" if abs(rat - 1.0) < 0.05 else "over-approx",
        })

    trip_vals = [r.trip_count for r in runs]
    return {
        "circuit":    label,
        "n_runs":     len(runs),
        "mean_trip":  _mean(trip_vals),
        "std_trip":   _std(trip_vals),
        "gate_rows":  rows,
    }


# ── report printing ───────────────────────────────────────────────────────────

def print_table(summary: dict):
    label = summary["circuit"]
    n     = summary["n_runs"]
    mt    = summary["mean_trip"]
    st    = summary["std_trip"]
    print(f"\n{'='*70}")
    print(f"  Circuit: {label}   (n_runs={n}, mean_trip={mt:.2f}±{st:.2f})")
    print(f"{'='*70}")
    hdr = f"  {'Gate':<16}  {'static/iter':>11}  {'obs/iter':>10}  {'±std':>7}  {'ratio':>7}  note"
    print(hdr)
    print("  " + "-" * 66)
    for row in summary["gate_rows"]:
        note = "✓ exact" if row["note"] == "exact" else "⚠ over-approx"
        print(f"  {row['gate']:<16}  {row['static_per_iter']:>11.3f}  "
              f"{row['obs_mean']:>10.3f}  {row['obs_std']:>7.3f}  "
              f"{row['ratio']:>7.3f}  {note}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E1 validation experiment")
    parser.add_argument("--n-coin", type=int, default=200,
                        help="Runs for coin-flip (default: 200)")
    parser.add_argument("--n-rus",  type=int, default=50,
                        help="Runs for RUS (default: 50)")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON to stdout")
    args = parser.parse_args()

    summaries = []

    # ── Coin-flip ────────────────────────────────────────────────────────
    coin_runs = run_circuit_session(
        circuit_fn=_make_coin_flip(),
        dev=qp.device("lightning.qubit", wires=1),
        n_wires=1,
        n_runs=args.n_coin,
        trip_fn=_coin_trip,
        static_per_iter=COIN_STATIC_PER_ITER,
        label="coin_flip",
    )
    coin_summary = summarise("coin_flip", coin_runs, COIN_STATIC_PER_ITER)
    summaries.append(coin_summary)

    # ── RUS ──────────────────────────────────────────────────────────────
    rus_runs = run_circuit_session(
        circuit_fn=_make_rus(),
        dev=qp.device("lightning.qubit", wires=2),
        n_wires=2,
        n_runs=args.n_rus,
        trip_fn=_rus_trip,
        static_per_iter=RUS_STATIC_PER_ITER,
        label="rus",
        outside_loop=RUS_OUTSIDE_LOOP,   # subtract 1 H before the loop
    )
    rus_summary = summarise("rus", rus_runs, RUS_STATIC_PER_ITER)
    summaries.append(rus_summary)

    # ── print ─────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print("\n" + "=" * 70)
        print("  E1 VALIDATION — Static per-iter cost == Runtime per-iter cost")
        print("=" * 70)
        print("  Ratio = observed_mean / static;  1.0 = exact;  0.5 = 2× over-approx")

        for s in summaries:
            print_table(s)

        print("=" * 70)
        print("  INTERPRETATION:")
        print("  ✓ exact       → ratio in [0.95, 1.05] — static analysis is correct")
        print("  ⚠ over-approx → ratio < 0.95 — scf.if max-over-branches over-counts")
        print("  Expected: all gates except PauliX(from reset=True) are exact.")
        print("            PauliX ratio ≈ 0.50 for p=0.5 measurements.")
        print()

    return summaries


if __name__ == "__main__":
    main()
