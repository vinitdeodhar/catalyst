#!/usr/bin/env python3
"""Trip count specialization — coverage-based peel depth selection and evaluation.

Key insight: WhileLoopPeelingPass with peel-factor=m already IS trip count
specialization. The recursive scf.if structure it emits handles exit at exactly
k=0, 1, ..., m-1 iterations in each nested else branch, with the residual
scf.while handling k>=m. No new compiler pass is needed.

What this experiment adds:
  1. Coverage-based peel depth: choose m* = min{m : P(k<=m) >= threshold}
     from the observed trip count PMF — rather than E[k]+3σ (depth bounding).
  2. Empirical validation: fraction of shots hitting specialized paths vs.
     residual while, measured from observed trip counts.
  3. Code size vs. coverage trade-off curve.

Distinction from depth bounding (E3b):
  Depth bounding: choose MAX_ITER to bound worst-case depth; accepts p_fail
    failures (shots where loop hits MAX_ITER without success).
  Trip count specialization: choose m to cover T% of shots with static paths;
    ALL shots produce correct results (residual while handles k>m exactly).
  These are complementary — specialization never drops any shots.

Circuits: coin-flip (p≈0.5), RUS (p≈0.146).

Usage:
    python3 run_trip_count_specialization.py [--n-profile N] [--coverage T]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))
from catalyst import measure, qjit, while_loop

# ── constants ──────────────────────────────────────────────────────────────────

_BUILD      = Path("/home/vadeo/catalyst/mlir/build")
QUANTUM_OPT = _BUILD / "bin" / "quantum-opt"
TEST_MLIR   = Path(__file__).parent / "test_data" / "coin_flip.mlir"

BODY_DEPTH = {"coin-flip": 3, "RUS": 7}

# ── circuit factories (unbounded — for profiling trip counts) ──────────────────

def make_coin_flip():
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


def make_rus():
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


# ── statistics ─────────────────────────────────────────────────────────────────

def _mean(xs):  return sum(xs) / len(xs) if xs else 0.0
def _std(xs):
    n, mu = len(xs), _mean(xs)
    return math.sqrt(sum((x - mu)**2 for x in xs) / (n - 1)) if n > 1 else 0.0


# ── phase 1: profile ───────────────────────────────────────────────────────────

def profile(circuit_fn, n_runs: int, label: str) -> List[int]:
    print(f"  [{label}] compiling…", end=" ", flush=True)
    circuit_fn()
    print(f"profiling {n_runs} shots", end="", flush=True)
    trips = []
    for i in range(n_runs):
        out = circuit_fn()
        trips.append(max(int(out), 1))
        if (i + 1) % 100 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done.")
    return trips


# ── phase 2: coverage analysis ─────────────────────────────────────────────────

def coverage_curve(trips: List[int], max_m: int = 20) -> List[Tuple[int, float]]:
    """Return (m, P(k<=m)) for m = 0..max_m."""
    n = len(trips)
    return [(m, sum(1 for k in trips if k <= m) / n) for m in range(max_m + 1)]


def optimal_peel_depth(trips: List[int], threshold: float) -> Tuple[int, float]:
    """Smallest m such that P(k<=m) >= threshold. Returns (m, achieved_coverage)."""
    for m, cov in coverage_curve(trips, max_m=max(trips) + 5):
        if cov >= threshold:
            return m, cov
    return max(trips), 1.0


def path_distribution(trips: List[int], peel_depth: int) -> Dict[str, float]:
    """Fraction of shots hitting each specialised path vs. residual."""
    n = len(trips)
    counts = {}
    for k in trips:
        label = f"k={k}" if k <= peel_depth else f"residual (k>{peel_depth})"
        counts[label] = counts.get(label, 0) + 1
    return {label: cnt / n for label, cnt in sorted(counts.items(),
            key=lambda kv: (0, int(kv[0][2:])) if kv[0].startswith("k=")
                           else (1, 0))}


# ── phase 3: MLIR code size ────────────────────────────────────────────────────

def mlir_stats(peel_depth: int) -> Dict:
    """Run quantum-opt with peel-factor=peel_depth; return output stats."""
    pipeline = (f"builtin.module(func.func("
                f"while-loop-peeling{{peel-factor={peel_depth}}}))")
    r = subprocess.run([str(QUANTUM_OPT), "-p", pipeline, str(TEST_MLIR)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"quantum-opt failed:\n{r.stderr}")
    lines = r.stdout.strip().splitlines()
    return {
        "lines":     len(lines),
        "if_ops":    sum(1 for l in lines if "scf.if" in l and "=" in l),
        "while_ops": sum(1 for l in lines if "scf.while" in l and "=" in l),
    }


# ── reporting ──────────────────────────────────────────────────────────────────

def run_circuit(name: str, circuit_fn, n_profile: int, coverage_threshold: float):
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    trips = profile(circuit_fn(), n_profile, name)
    mean_k = _mean(trips)
    std_k  = _std(trips)
    p_hat  = 1.0 / mean_k if mean_k else 0.0

    print(f"\n  Distribution ({n_profile} shots):")
    print(f"    mean_k={mean_k:.2f}  std_k={std_k:.2f}  p̂={p_hat:.4f}")

    # ── coverage curve ─────────────────────────────────────────────────────
    max_show = min(max(trips), 15)
    curve = coverage_curve(trips, max_m=max_show)

    print(f"\n  Coverage curve  (fraction of shots using specialized path, no residual while)")
    print(f"  {'m':>4}  {'P(k<=m)':>9}  {'residual%':>10}  {'wc_depth':>9}  {'code_lines':>10}")
    print("  " + "─" * 52)
    body_d = BODY_DEPTH[name]
    for m, cov in curve:
        if m == 0:
            continue
        wc = m * body_d
        # estimate MLIR lines: baseline ~16 lines + 8 lines per if level
        est_lines = 16 + 8 * m
        marker = " ← 90%" if abs(cov - coverage_threshold) < 0.02 else ""
        print(f"  {m:>4}  {cov*100:>8.1f}%  {(1-cov)*100:>9.1f}%  "
              f"{wc:>9}  {est_lines:>10}{marker}")

    # ── optimal peel depth ─────────────────────────────────────────────────
    m_star, achieved = optimal_peel_depth(trips, coverage_threshold)
    print(f"\n  Optimal peel depth for {coverage_threshold*100:.0f}% coverage: "
          f"m*={m_star}  (achieved {achieved*100:.1f}%)")

    # ── path distribution at m* ────────────────────────────────────────────
    dist = path_distribution(trips, m_star)
    print(f"\n  Observed path distribution at m*={m_star}:")
    print(f"  {'Path':<22}  {'fraction':>9}  {'shots':>7}")
    print("  " + "─" * 42)
    for label, frac in dist.items():
        shots = round(frac * n_profile)
        bar = "█" * round(frac * 30)
        print(f"  {label:<22}  {frac*100:>8.1f}%  {shots:>7}  {bar}")

    # ── MLIR stats at m* (for coin-flip test MLIR only) ───────────────────
    if name == "coin-flip":
        print(f"\n  MLIR code stats (quantum-opt peel-factor={m_star}):")
        try:
            stats0 = mlir_stats(0)
            statsM = mlir_stats(m_star)
            print(f"  {'':>20}  {'baseline (m=0)':>14}  {'m*={m_star}':>10}  {'change':>8}")
            print("  " + "─" * 58)
            print(f"  {'IR lines':>20}  {stats0['lines']:>14}  "
                  f"{statsM['lines']:>10}  +{statsM['lines']-stats0['lines']:>6}")
            print(f"  {'scf.if ops':>20}  {stats0['if_ops']:>14}  "
                  f"{statsM['if_ops']:>10}  +{statsM['if_ops']-stats0['if_ops']:>6}")
            print(f"  {'scf.while ops':>20}  {stats0['while_ops']:>14}  "
                  f"{statsM['while_ops']:>10}  "
                  f"{'unchanged' if statsM['while_ops'] == stats0['while_ops'] else ''}")
        except RuntimeError as e:
            print(f"  quantum-opt error: {e}")

    # ── key metrics ────────────────────────────────────────────────────────
    residual_frac = sum(1 for k in trips if k > m_star) / len(trips)
    print(f"\n  Key metrics at m*={m_star}:")
    print(f"    Shots using specialised paths (no while loop): "
          f"{(1-residual_frac)*100:.1f}%")
    print(f"    Shots falling to residual while:               "
          f"{residual_frac*100:.1f}%")
    print(f"    Worst-case depth of specialized paths:         "
          f"{m_star * body_d} gates")
    print(f"    Shots needing dynamic circuit support:         "
          f"{residual_frac*100:.1f}% (down from 100%)")

    return {
        "circuit": name, "mean_k": mean_k, "std_k": std_k, "p_hat": p_hat,
        "m_star": m_star, "achieved_coverage": achieved,
        "residual_frac": residual_frac, "trips": trips,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trip count specialization — coverage-based evaluation")
    parser.add_argument("--n-profile",  type=int,   default=500,
                        help="Profile shots per circuit (default: 500)")
    parser.add_argument("--coverage",   type=float, default=0.90,
                        help="Coverage threshold for m* selection (default: 0.90)")
    parser.add_argument("--circuit",    choices=["coin-flip", "RUS", "both"],
                        default="both")
    args = parser.parse_args()

    print("=" * 72)
    print("  Trip Count Specialization — Coverage-Based Peel Depth")
    print(f"  n_profile={args.n_profile}   coverage_threshold={args.coverage*100:.0f}%")
    print(f"  Pass: WhileLoopPeelingPass (peel-factor=m*)")
    print(f"  Metric: fraction of shots avoiding residual scf.while")
    print("=" * 72)

    specs = []
    if args.circuit in ("coin-flip", "both"):
        specs.append(("coin-flip", make_coin_flip))
    if args.circuit in ("RUS", "both"):
        specs.append(("RUS", make_rus))

    results = []
    for name, factory in specs:
        r = run_circuit(name, factory, args.n_profile, args.coverage)
        results.append(r)

    # ── cross-circuit summary ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY — {args.coverage*100:.0f}% coverage specialization")
    print(f"{'='*72}")
    print(f"  {'Circuit':<12}  {'p̂':>6}  {'m*':>4}  {'coverage':>9}  "
          f"{'specialized%':>13}  {'residual%':>10}  {'needs_dyn_circ':>14}")
    print("  " + "─" * 72)
    for r in results:
        spec = (1 - r["residual_frac"]) * 100
        res  = r["residual_frac"] * 100
        print(f"  {r['circuit']:<12}  {r['p_hat']:>6.4f}  {r['m_star']:>4}  "
              f"{r['achieved_coverage']*100:>8.1f}%  {spec:>12.1f}%  "
              f"{res:>9.1f}%  {res:>13.1f}%")
    print()
    print("  'specialized%' = shots running as straight-line code (no while loop)")
    print("  'needs_dyn_circ' = shots requiring residual scf.while (dynamic circuit)")


if __name__ == "__main__":
    main()
