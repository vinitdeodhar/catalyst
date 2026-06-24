#!/usr/bin/env python3
"""E3b (all benchmarks) — DepthBoundingPass across the full dynamic benchmark suite.

Runs the real-pass depth-bounding experiment on all 5 while-loop benchmarks:
  coin-flip, RUS, MSD, BBHT Grover (n_data=3), Nested RUS-in-BBHT (n_data=2).

Iterative QPE uses only for_loops — not applicable.

Method (per circuit):
  1. Profile n_profile shots with GateCounterSession (or direct circuit for Nested)
     to collect trip counts and compute mean_k, std_k, p_hat.
  2. Apply DepthBoundingPass (real MLIR pass) to the compiled circuit MLIR.
  3. Compute calibration from profiling data: obs_fail = P(trip_count > MAX_ITER).
  4. Report depth savings vs. conservative MAX_ITER=BASELINE.

Usage::

    python3 run_e3b_all_benchmarks.py [--n-profile N]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import pennylane as qp

sys.path.insert(0, str(Path(__file__).parent))

from catalyst import cond, for_loop, measure, qjit, while_loop
from gate_counter_estimator import GateCounterSession, RunResult
from profile_guided_estimator import (
    ProfileGuidedEstimator,
    annotate_mlir_max_iterations,
)

# ── config ────────────────────────────────────────────────────────────────────

C_VALUES = [1, 2, 3, 4]
CONSERVATIVE_MAX = 50   # conservative iteration cap without profiling (same as original E3b)

# ── _KeyedSession for circuits that need a fresh JAX PRNG key each call ──────

class _KeyedSession:
    """Wraps GateCounterSession to inject a fresh PRNGKey on each run()."""

    def __init__(self, inner: GateCounterSession, seed_offset: int = 7919):
        self._inner = inner
        self._offset = seed_offset
        self._n = 0

    def run(self) -> RunResult:
        key = jax.random.PRNGKey(self._n + self._offset)
        self._n += 1
        return self._inner.run(key)


# ── circuit definitions ───────────────────────────────────────────────────────

# ── coin-flip ─────────────────────────────────────────────────────────────────

def _coin_flip_body():
    @while_loop(lambda count, result: result == 0)
    def flip(count, result):
        qp.Hadamard(wires=0)
        m = measure(0, reset=True)
        return count + jnp.int64(1), jnp.int64(m)
    count, _ = flip(jnp.int64(0), jnp.int64(0))
    return count


# ── RUS ───────────────────────────────────────────────────────────────────────

def _rus_body():
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


# ── MSD ───────────────────────────────────────────────────────────────────────

N_MAGIC = 7
P_ERR   = 0.10

def _msd_body(key):
    @while_loop(lambda success, _k: ~success)
    def msd_attempt(success, key):
        for wire in range(1, N_MAGIC + 1):
            qp.Hadamard(wires=wire)
            qp.T(wires=wire)
            key, subkey = jax.random.split(key)
            error = jax.random.bernoulli(subkey, jnp.float64(P_ERR))

            @cond(error)
            def inject():
                qp.PauliX(wires=wire)

            inject()
        for wire in range(1, N_MAGIC + 1):
            qp.CNOT(wires=[wire, 0])
        syn = measure(0, reset=True)
        for wire in range(1, N_MAGIC + 1):
            measure(wire, reset=True)
        return jnp.bool_(syn == 0), key

    success, _ = msd_attempt(jnp.bool_(False), key)
    return success


# ── BBHT Grover ───────────────────────────────────────────────────────────────

_LAMBDA = 6.0 / 5.0
N_DATA_BBHT = 3

def _bbht_body(key):
    n_data   = N_DATA_BBHT
    n_space  = jnp.int64(2 ** n_data)

    @while_loop(lambda found, _m, _k: ~found)
    def bbht_loop(found, m, rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        k = jax.random.randint(subkey, shape=(), minval=jnp.int64(1),
                               maxval=jnp.int64(m) + 1)

        for i in range(n_data):
            qp.Hadamard(wires=i)

        @for_loop(0, k, 1)
        def grover_step(_):
            # oracle
            qp.Hadamard(wires=n_data - 1)
            qp.Toffoli(wires=[0, 1, 2])
            qp.Hadamard(wires=n_data - 1)
            # diffuser
            for i in range(n_data):
                qp.Hadamard(wires=i)
                qp.PauliX(wires=i)
            qp.Hadamard(wires=n_data - 1)
            qp.Toffoli(wires=[0, 1, 2])
            qp.Hadamard(wires=n_data - 1)
            for i in range(n_data):
                qp.PauliX(wires=i)
                qp.Hadamard(wires=i)

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


# ── Nested RUS-in-BBHT ────────────────────────────────────────────────────────

N_DATA_NESTED = 2

def _nested_body():
    n_data  = N_DATA_NESTED
    ancilla = n_data

    @while_loop(lambda found, _cnt: ~found)
    def search_loop(found, attempt_count):
        for i in range(n_data):
            qp.Hadamard(wires=i)

        @while_loop(lambda oracle_done: ~oracle_done)
        def rus_oracle(oracle_done):
            qp.Hadamard(wires=ancilla)
            for i in range(n_data):
                qp.CNOT(wires=[i, ancilla])
            qp.T(wires=ancilla)
            for i in range(n_data):
                qp.CNOT(wires=[i, ancilla])
            qp.Hadamard(wires=ancilla)
            anc_m = measure(ancilla, reset=True)
            return jnp.bool_(anc_m == 1)

        rus_oracle(jnp.bool_(False))

        # diffuser
        for i in range(n_data):
            qp.Hadamard(wires=i)
            qp.PauliX(wires=i)
        qp.CZ(wires=[0, 1])
        for i in range(n_data):
            qp.PauliX(wires=i)
            qp.Hadamard(wires=i)

        bits = jnp.zeros(n_data, dtype=jnp.int64)
        for i in range(n_data):
            m_i = measure(i, reset=True)
            bits = bits.at[i].set(jnp.int64(m_i))

        found_now = jnp.all(bits == 1)
        return found_now, attempt_count + jnp.int64(1)

    found, n_attempts = search_loop(jnp.bool_(False), jnp.int64(0))
    return found, n_attempts


# ── benchmark specs ───────────────────────────────────────────────────────────

SPECS = {
    "coin-flip": {
        "body":       _coin_flip_body,
        "dev":        lambda: qp.device("lightning.qubit", wires=1),
        "needs_key":  False,
        "per_iter":   {"Hadamard_1": 1, "Measure_1": 1},
        "body_depth": 3,
        "notes":      "p≈0.5, Geom",
    },
    "RUS": {
        "body":       _rus_body,
        "dev":        lambda: qp.device("lightning.qubit", wires=2),
        "needs_key":  False,
        "per_iter":   {"T_1": 1, "Hadamard_1": 2, "CNOT_2": 2, "Measure_1": 1},
        "body_depth": 7,
        "notes":      "p≈0.146, Geom",
    },
    "MSD": {
        "body":       _msd_body,
        "dev":        lambda: qp.device("lightning.qubit", wires=N_MAGIC + 1),
        "needs_key":  True,
        "per_iter":   {"T_1": N_MAGIC, "CNOT_2": N_MAGIC, "Measure_1": N_MAGIC + 1},
        "body_depth": N_MAGIC * 3 + 1,   # H+T+CNOT per magic wire + syndrome meas
        "notes":      f"n_magic={N_MAGIC}, p_err={P_ERR}, Geom≈0.6",
    },
    "BBHT": {
        "body":       _bbht_body,
        "dev":        lambda: qp.device("lightning.qubit", wires=N_DATA_BBHT),
        "needs_key":  True,
        "per_iter":   {"Measure_1": N_DATA_BBHT},  # data meas per outer iter
        "body_depth": 20,   # per outer iter (E[k_grover]≈1 × 20 gates/step)
        "notes":      f"n_data={N_DATA_BBHT}, non-Geom, conc. at k=1",
    },
    "Nested": {
        "body":       _nested_body,
        "dev":        lambda: qp.device("lightning.qubit", wires=N_DATA_NESTED + 1),
        "needs_key":  False,
        "per_iter":   None,   # trip count read from circuit output directly
        "body_depth": 12,
        "notes":      f"n_data={N_DATA_NESTED}, outer BBHT / inner RUS",
    },
}

# ── statistics ────────────────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    n = len(xs)
    if n < 2: return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))

def _wilson_ci(k, n):
    if n == 0: return 0.0, 1.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return max(0.0, c - m), min(1.0, c + m)

# ── pass helpers ──────────────────────────────────────────────────────────────

def _extract_circuit_module(mlir_text: str) -> str:
    import re
    m = re.search(r"\bmodule @module_\w+ \{", mlir_text)
    if not m:
        return mlir_text
    start = m.start()
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


def run_pass(mlir_text: str, max_iter: int) -> str:
    """Extract circuit module, annotate, run DepthBoundingPass. Returns MLIR."""
    from catalyst.compiler import _quantum_opt
    circuit_mlir = _extract_circuit_module(mlir_text)
    annotated = annotate_mlir_max_iterations(circuit_mlir, max_iter)
    return _quantum_opt(
        "--pass-pipeline",
        "builtin.module(func.func(depth-bounding))",
        stdin=annotated,
    )


def _while_result_counts(mlir_text: str) -> Tuple[int, int]:
    """Return (before_count, after_count) of scf.while results from MLIR text."""
    import re
    # before: e.g. %1:2 = scf.while
    m_before = re.search(r"%\w+:(\d+)\s*=\s*scf\.while", mlir_text)
    n_before = int(m_before.group(1)) if m_before else 0
    return n_before, n_before  # will be updated after pass


def _print_while_summary(bounded_mlir: str, name: str, max_iter: int):
    """Print the result-count change and the new condition pattern."""
    import re
    # Count scf.while results
    m = re.search(r"%\w+:(\d+)\s*=\s*scf\.while", bounded_mlir)
    n_results = int(m.group(1)) if m else "?"
    # Show just the before-region condition lines
    lines = bounded_mlir.splitlines()
    in_before = False
    printed = []
    for line in lines:
        if "scf.while" in line:
            in_before = True
            printed.append(f"    {line.strip()}")
        elif in_before:
            if "scf.condition" in line:
                printed.append(f"    …")
                printed.append(f"    {line.strip()}")
                break
            stripped = line.strip()
            if any(kw in stripped for kw in ["cmpi", "andi", "constant"]):
                printed.append(f"    {stripped}")
    print(f"\n  [{name}] bounded MLIR (MAX_ITER={max_iter}, results: {n_results})")
    for l in printed:
        print(l)


# ── per-circuit profiling ─────────────────────────────────────────────────────

def _profile_with_gate_counter(
    spec: dict, n_profile: int
) -> Tuple[List[int], str]:
    """Profile via GateCounterSession. Returns (trip_counts, mlir_text)."""
    body_fn   = spec["body"]
    dev       = spec["dev"]()
    per_iter  = spec["per_iter"]
    needs_key = spec["needs_key"]

    # Circuits that take a JAX PRNGKey need a sample key at compile time
    # so GateCounterSession uses the *args code path during JIT compilation.
    compile_args = (jax.random.PRNGKey(0),) if needs_key else ()

    print(f"  compiling…", end=" ", flush=True)
    with GateCounterSession(body_fn, dev, *compile_args) as sess:
        if needs_key:
            wrapped_sess = _KeyedSession(sess)
            estimator = ProfileGuidedEstimator(wrapped_sess, per_iter)
        else:
            estimator = ProfileGuidedEstimator(sess, per_iter)

        print(f"profiling {n_profile} shots", end="", flush=True)
        report = estimator.run(n_profile)
        print(" done.")
        mlir_text = sess._compiled_fn.mlir

    trip_counts = [max(1, int(round(tc))) for tc in report.trip_counts]
    return trip_counts, mlir_text


def _profile_nested(n_profile: int) -> Tuple[List[int], str]:
    """Profile Nested RUS-BBHT directly from circuit outputs."""
    dev = SPECS["Nested"]["dev"]()

    print(f"  compiling…", end=" ", flush=True)

    @qjit
    @qp.qnode(dev)
    def _circuit():
        return _nested_body()

    _circuit()   # warm-up / trigger compilation
    mlir_text = _circuit.mlir

    print(f"profiling {n_profile} shots", end="", flush=True)
    trip_counts = []
    for i in range(n_profile):
        _, n_attempts = _circuit()
        trip_counts.append(max(1, int(n_attempts)))
        if (i + 1) % 10 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done.")

    return trip_counts, mlir_text


# ── main experiment ───────────────────────────────────────────────────────────

def run_benchmark(name: str, spec: dict, n_profile: int, show_mlir: bool) -> Dict:
    body_depth = spec["body_depth"]
    baseline_wc = CONSERVATIVE_MAX * body_depth

    print(f"\n{'='*72}")
    print(f"  {name}   [{spec['notes']}]   (n_profile={n_profile})")
    print(f"{'='*72}")

    # ── Phase 1: profile ──────────────────────────────────────────────────────
    if spec["per_iter"] is None:
        trip_counts, mlir_text = _profile_nested(n_profile)
    else:
        trip_counts, mlir_text = _profile_with_gate_counter(spec, n_profile)

    mean_k = _mean(trip_counts)
    std_k  = _std(trip_counts)
    p_hat  = 1.0 / mean_k if mean_k > 0 else float("nan")
    geom   = (1.0 - p_hat) ** CONSERVATIVE_MAX

    print(f"  Fitted:  mean_k={mean_k:.2f}  std_k={std_k:.2f}  "
          f"p̂={p_hat:.4f}  baseline wc_depth={baseline_wc}  "
          f"pred_fail(baseline)={geom*100:.3f}%")

    # ── Phase 2: DepthBoundingPass demo (c=3) ────────────────────────────────
    max_iter_demo = math.ceil(mean_k + 3 * std_k)
    try:
        bounded_mlir = run_pass(mlir_text, max_iter_demo)
        pass_ok = True
        if show_mlir:
            _print_while_summary(bounded_mlir, name, max_iter_demo)
    except Exception as e:
        print(f"  [WARNING] Pass failed at c=3: {e}")
        pass_ok = False

    # ── Phase 3: calibration table ────────────────────────────────────────────
    n = len(trip_counts)
    print(f"\n  {'c':>3}  {'MAX_ITER':>8}  {'pass':>4}  {'pred_fail%':>10}  "
          f"{'obs_fail%':>10}  {'95% CI':>16}  {'wc_depth':>8}  "
          f"{'depth_saved':>11}  calib")
    print("  " + "─" * 82)

    rows = []
    for c in C_VALUES:
        max_iter  = math.ceil(mean_k + c * std_k)
        pred_fail = (1.0 - p_hat) ** max_iter

        n_fail    = sum(1 for tc in trip_counts if tc > max_iter)
        obs_fail  = n_fail / n
        lo, hi    = _wilson_ci(n_fail, n)

        wc_depth    = max_iter * body_depth
        depth_saved = max(0.0, (1.0 - max_iter / CONSERVATIVE_MAX)) * 100.0

        calib = lo <= pred_fail <= hi
        calib_flag = "✓" if calib else "⚠"

        try:
            run_pass(mlir_text, max_iter)
            pass_flag = "✓"
        except Exception:
            pass_flag = "✗"

        print(f"  {c:>3}  {max_iter:>8}  {pass_flag:>4}  "
              f"{pred_fail*100:>10.2f}%  {obs_fail*100:>10.2f}%  "
              f"[{lo*100:5.1f}%,{hi*100:5.1f}%]  "
              f"{wc_depth:>8}  {depth_saved:>10.1f}%  {calib_flag}")

        rows.append(dict(c=c, max_iter=max_iter, pass_ok=pass_flag=="✓",
                         pred_fail=pred_fail, obs_fail=obs_fail,
                         ci_lo=lo, ci_hi=hi, wc_depth=wc_depth,
                         depth_saved=depth_saved/100.0, calibrated=calib))

    return dict(circuit=name, mean_k=mean_k, std_k=std_k, p_hat=p_hat,
                body_depth=body_depth, pass_ok=pass_ok, rows=rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-profile", type=int, default=100)
    parser.add_argument("--circuits", nargs="+",
                        default=["coin-flip","RUS","MSD","BBHT","Nested"],
                        choices=list(SPECS))
    parser.add_argument("--show-mlir", action="store_true", default=True)
    parser.add_argument("--no-show-mlir", dest="show_mlir", action="store_false")
    args = parser.parse_args()

    print("=" * 72)
    print("  E3b (all benchmarks) — DepthBoundingPass depth reduction survey")
    print(f"  n_profile={args.n_profile}   conservative_max={CONSERVATIVE_MAX}")
    print(f"  QPE skipped: for_loop only (deterministic trip count)")
    print("=" * 72)

    results = []
    for name in args.circuits:
        r = run_benchmark(name, SPECS[name], args.n_profile, args.show_mlir)
        results.append(r)

    # ── combined summary ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY — all circuits, c=3 (DepthBoundingPass, real MLIR pass)")
    print(f"{'='*72}")
    print(f"  {'Circuit':<12}  {'notes':<28}  {'E[k]':>5}  {'σ':>5}  "
          f"{'MAX':>4}  {'pass':>4}  {'pred%':>6}  {'obs%':>6}  "
          f"{'wc_depth':>8}  {'saved':>6}  calib")
    print("  " + "─" * 90)
    for r in results:
        row3 = next(x for x in r["rows"] if x["c"] == 3)
        flag  = "✓" if row3["calibrated"] else "⚠"
        picon = "✓" if row3["pass_ok"] else "✗"
        notes = SPECS[r["circuit"]]["notes"]
        print(f"  {r['circuit']:<12}  {notes:<28}  "
              f"{r['mean_k']:>5.2f}  {r['std_k']:>5.2f}  "
              f"{row3['max_iter']:>4}  {picon:>4}  "
              f"{row3['pred_fail']*100:>5.1f}%  {row3['obs_fail']*100:>5.1f}%  "
              f"{row3['wc_depth']:>8}  "
              f"{row3['depth_saved']*100:>5.0f}%  {flag}")
    print()
    print("  Depth saved = 1 − MAX_ITER(c=3) / CONSERVATIVE_MAX")
    print("  pass=✓ → DepthBoundingPass ran on real Catalyst MLIR")
    print("  calib=✓ → pred_fail within 95% Wilson CI of obs_fail")
    print()


if __name__ == "__main__":
    main()
