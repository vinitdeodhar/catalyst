#!/usr/bin/env python3
"""E3b (real pass) — Depth bounding using DepthBoundingPass on real Catalyst MLIR.

This script re-runs the E3b depth-bounding calibration experiment using the
actual compiler pass (``--depth-bounding``) instead of Python circuit editing.

Two-phase approach
------------------
Phase 1  Compile the unbounded circuit and profile N_PROFILE shots with
         GateCounterSession to collect trip counts.  Use ProfileGuidedEstimator
         to derive mean_k, std_k, p_hat.

Phase 2  For each confidence level c, compute MAX_ITER = ceil(mean_k + c*std_k)
         (from ProfileReport.max_iterations(c)).  Annotate the compiled
         circuit's MLIR with ``max_iterations = MAX_ITER : i64`` and run
         ``quantum-opt --depth-bounding`` to produce the bounded MLIR.  Print
         the before/after transformation.

Calibration  obs_fail = fraction of profiled trip counts that exceed MAX_ITER.
             This is statistically equivalent to running the bounded circuit
             because: a bounded circuit fails iff the unbounded trip count
             would have exceeded the bound under the same PRNG sequence.

Circuits: coin-flip (p≈0.5) and RUS (p≈0.146).

Usage::

    python3 run_e3b_real_pass.py [--n-profile N] [--circuit {coin-flip,RUS,both}]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import measure, qjit, while_loop
from gate_counter_estimator import GateCounterSession
from profile_guided_estimator import (
    ProfileGuidedEstimator,
    annotate_mlir_max_iterations,
)

# ── config ────────────────────────────────────────────────────────────────────

CONSERVATIVE_MAX_ITER = 50
C_VALUES = [1, 2, 3, 4]

# Gates per one loop body iteration (from static resource analysis).
# Used as per_iter_costs for ProfileGuidedEstimator.
PER_ITER_COSTS = {
    "coin-flip": {"Hadamard_1": 1, "Measure_1": 1},    # H + measure (+ PauliX in scf.if)
    "RUS":       {"Hadamard_1": 2, "CNOT_2": 2, "T_1": 1, "Measure_1": 1},
}

# Number of quantum gates per loop body iteration (for worst-case depth calc).
BODY_DEPTH = {
    "coin-flip": 3,   # H + measure + conditional-X
    "RUS":       7,   # 2H + 2CNOT + T + measure + conditional-X
}

# ── circuit definitions (unbounded) ──────────────────────────────────────────

def coin_flip_body():
    """Coin-flip loop body — used by GateCounterSession."""
    @while_loop(lambda count, result: result == 0)
    def flip(count, result):
        qp.Hadamard(wires=0)
        m = measure(0, reset=True)
        return count + jnp.int64(1), jnp.int64(m)
    count, _ = flip(jnp.int64(0), jnp.int64(0))
    return count


def rus_body():
    """RUS loop body — used by GateCounterSession."""
    target, ancilla = 0, 1
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


CIRCUIT_SPECS = {
    "coin-flip": {
        "body":     coin_flip_body,
        "dev":      lambda: qp.device("lightning.qubit", wires=1),
        "per_iter": PER_ITER_COSTS["coin-flip"],
        "depth":    BODY_DEPTH["coin-flip"],
    },
    "RUS": {
        "body":     rus_body,
        "dev":      lambda: qp.device("lightning.qubit", wires=2),
        "per_iter": PER_ITER_COSTS["RUS"],
        "depth":    BODY_DEPTH["RUS"],
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _wilson_ci_95(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z = 1.96
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _extract_circuit_module(mlir_text: str) -> str:
    """Extract the nested ``module @module_<name>`` block from Catalyst MLIR.

    Catalyst wraps the quantum circuit inside a nested module named
    @module_<fn_name>.  The `depth-bounding` pass targets func::FuncOp, so
    we extract that inner module (which has the scf.while at its top level).
    """
    import re
    match = re.search(r"\bmodule @module_\w+ \{", mlir_text)
    if not match:
        return mlir_text
    start = match.start()
    depth = 0
    i = start
    while i < len(mlir_text):
        if mlir_text[i] == "{":
            depth += 1
        elif mlir_text[i] == "}":
            depth -= 1
            if depth == 0:
                return mlir_text[start : i + 1].strip()
        i += 1
    return mlir_text


def run_depth_bounding_pass(mlir_text: str, max_iter: int) -> str:
    """Extract circuit module, annotate with max_iterations, run DepthBoundingPass.

    Returns the transformed MLIR of the circuit module.
    """
    from catalyst.compiler import _quantum_opt

    circuit_mlir = _extract_circuit_module(mlir_text)
    annotated = annotate_mlir_max_iterations(circuit_mlir, max_iter)
    return _quantum_opt(
        "--pass-pipeline",
        "builtin.module(func.func(depth-bounding))",
        stdin=annotated,
    )


def _get_circuit_mlir(sess: GateCounterSession) -> str:
    """Return the full Catalyst MLIR of the gate-counter compiled function."""
    return sess._compiled_fn.mlir


# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(name: str, n_profile: int, show_mlir: bool) -> Dict:
    spec = CIRCUIT_SPECS[name]
    dev = spec["dev"]()
    body_fn = spec["body"]
    per_iter = spec["per_iter"]
    body_depth = spec["depth"]

    print(f"\n{'='*72}")
    print(f"  {name}   ({n_profile} profile shots)")
    print(f"{'='*72}")

    # ── Phase 1: profile ──────────────────────────────────────────────────────
    print("  Phase 1: compiling + profiling unbounded circuit…", flush=True)
    with GateCounterSession(body_fn, dev) as sess:
        estimator = ProfileGuidedEstimator(sess, per_iter)
        report = estimator.run(n_profile)

        trip_counts = [int(round(tc)) for tc in report.trip_counts]
        mean_k = report.mean_trip
        std_k = report.std_trip
        p_hat = report.p_hat

        print(f"  Fitted:  mean_k={mean_k:.2f}  std_k={std_k:.2f}  "
              f"p̂={p_hat:.4f}  (1/p̂={1/p_hat:.2f})")

        # ── Phase 2: run DepthBoundingPass on real MLIR ─────────────────────
        circuit_mlir = sess._compiled_fn.mlir
        print(f"\n  Phase 2: running DepthBoundingPass on real Catalyst MLIR…")
        # Use c=3 for the primary demo (canonical recommendation).
        max_iter_demo = report.max_iterations(c=3)

        try:
            bounded_mlir = run_depth_bounding_pass(circuit_mlir, max_iter_demo)
            pass_ok = True
        except Exception as e:
            print(f"  [WARNING] Pass failed: {e}")
            pass_ok = False
            bounded_mlir = ""

        if pass_ok and show_mlir:
            # Show the scf.while section in the transformed output.
            print(f"\n  -- Bounded MLIR (MAX_ITER={max_iter_demo}, c=3, "
                  f"mean_k={mean_k:.1f}+3×{std_k:.1f}) --")
            _print_while_section(bounded_mlir)

    # ── Calibration table ─────────────────────────────────────────────────────
    baseline_depth  = CONSERVATIVE_MAX_ITER * body_depth
    baseline_pred   = (1.0 - p_hat) ** CONSERVATIVE_MAX_ITER
    print(f"\n  Conservative baseline: MAX_ITER={CONSERVATIVE_MAX_ITER}  "
          f"worst-case depth={baseline_depth}  pred_fail={baseline_pred*100:.3f}%")

    print(f"\n  Calibration table  (obs_fail from {n_profile} profile shots)\n")
    print(f"  {'c':>3}  {'MAX_ITER':>8}  {'pass':>4}  {'pred_fail%':>10}  "
          f"{'obs_fail%':>10}  {'95% CI':>16}  {'wc_depth':>8}  "
          f"{'depth_saved':>11}  calib")
    print("  " + "─" * 80)

    rows = []
    for c in C_VALUES:
        max_iter = report.max_iterations(c)
        pred_fail = (1.0 - p_hat) ** max_iter

        # obs_fail from trip count simulation (statistically equivalent to
        # running the bounded circuit: failure iff trip_count > max_iter).
        n_fail = sum(1 for tc in trip_counts if tc > max_iter)
        obs_fail = n_fail / len(trip_counts)
        lo, hi = _wilson_ci_95(n_fail, len(trip_counts))

        wc_depth = max_iter * body_depth
        depth_saved = (1.0 - max_iter / CONSERVATIVE_MAX_ITER) * 100.0

        calibrated = lo <= pred_fail <= hi
        flag = "✓" if calibrated else "⚠"

        # Run DepthBoundingPass for this c value to confirm it runs cleanly.
        try:
            run_depth_bounding_pass(circuit_mlir, max_iter)
            pass_flag = "✓"
        except Exception:
            pass_flag = "✗"

        print(f"  {c:>3}  {max_iter:>8}  {pass_flag:>4}  "
              f"{pred_fail*100:>10.2f}%  {obs_fail*100:>10.2f}%  "
              f"[{lo*100:5.1f}%,{hi*100:5.1f}%]  "
              f"{wc_depth:>8}  {depth_saved:>10.1f}%  {flag}")

        rows.append({
            "c":           c,
            "max_iter":    max_iter,
            "pass_ok":     pass_flag == "✓",
            "pred_fail":   pred_fail,
            "obs_fail":    obs_fail,
            "ci_lo":       lo,
            "ci_hi":       hi,
            "wc_depth":    wc_depth,
            "depth_saved": depth_saved / 100.0,
            "calibrated":  calibrated,
            "n_fail":      n_fail,
            "n_total":     len(trip_counts),
        })

    return {
        "circuit":    name,
        "mean_k":     mean_k,
        "std_k":      std_k,
        "p_hat":      p_hat,
        "body_depth": body_depth,
        "pass_ok":    pass_ok,
        "rows":       rows,
    }


def _print_while_section(mlir_text: str, max_lines: int = 40):
    """Print the scf.while section of an MLIR text, truncated for readability."""
    lines = mlir_text.splitlines()
    in_while = False
    depth = 0
    printed = 0
    for line in lines:
        if "scf.while" in line:
            in_while = True
        if in_while:
            print(f"    {line}")
            printed += 1
            depth += line.count("{") - line.count("}")
            if depth <= 0 and printed > 1:
                break
            if printed >= max_lines:
                print("    … (truncated)")
                break


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E3b depth bounding with real MLIR pass")
    parser.add_argument("--n-profile", type=int, default=200,
                        help="Profile shots (default: 200)")
    parser.add_argument("--circuit", choices=["coin-flip", "RUS", "both"], default="both")
    parser.add_argument("--show-mlir", action="store_true", default=True,
                        help="Print before/after MLIR section (default: on)")
    parser.add_argument("--no-show-mlir", dest="show_mlir", action="store_false")
    args = parser.parse_args()

    print("=" * 72)
    print("  E3b — Depth Bounding with Real DepthBoundingPass")
    print(f"  n_profile={args.n_profile}   conservative_max={CONSERVATIVE_MAX_ITER}")
    print(f"  pass: quantum-opt --depth-bounding (added to Catalyst build)")
    print(f"  obs_fail derived from profile trip counts (equiv. to bounded execution)")
    print("=" * 72)

    names = (["coin-flip", "RUS"] if args.circuit == "both"
             else [args.circuit])

    results = []
    for name in names:
        r = run_experiment(name, args.n_profile, args.show_mlir)
        results.append(r)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY — DepthBoundingPass (c=3) vs. conservative MAX_ITER=50")
    print(f"{'='*72}")
    print(f"  {'Circuit':<12}  {'p̂':>6}  {'E[k]':>5}  {'σ':>5}  "
          f"{'MAX_ITER':>8}  {'pass':>4}  {'pred%':>6}  {'obs%':>6}  "
          f"{'depth_saved':>11}  calib")
    print("  " + "─" * 74)
    for r in results:
        row3 = next(x for x in r["rows"] if x["c"] == 3)
        flag = "✓" if row3["calibrated"] else "⚠"
        pass_icon = "✓" if row3["pass_ok"] else "✗"
        print(f"  {r['circuit']:<12}  {r['p_hat']:>6.4f}  {r['mean_k']:>5.2f}  "
              f"{r['std_k']:>5.2f}  {row3['max_iter']:>8}  {pass_icon:>4}  "
              f"{row3['pred_fail']*100:>5.1f}%  {row3['obs_fail']*100:>5.1f}%  "
              f"{row3['depth_saved']*100:>10.0f}%  {flag}")
    print()
    print("  pass=✓  → DepthBoundingPass ran cleanly on real Catalyst MLIR")
    print("  calib=✓ → predicted failure rate lies within 95% binomial CI")
    print("  Depth saved = 1 − MAX_ITER(c=3) / CONSERVATIVE_MAX_ITER")
    print()


if __name__ == "__main__":
    main()
