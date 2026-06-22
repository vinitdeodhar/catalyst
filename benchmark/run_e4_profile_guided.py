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
"""E4: Profile-Guided Iteration-Count Refinement — Convergence Experiment.

Demonstrates that N profile shots (gate counter executions) suffice to estimate
E[k] within ±1 of the true value for RUS and MSD circuits.

The profile-guided loop:
  1. Compile circuit with analytical E[k] as the initial estimated_iterations.
  2. Run batch_size shots; back-calculate trip counts k_i = gates_i / per_iter.
  3. Fit mean_k = mean({k_i}), recommended = round(mean_k).
  4. Update estimated_iterations → recompile (modelled here as a running mean).
  5. Repeat until recommended_iters stabilises.

Outputs (stdout):
  · Per-circuit convergence table: batch, n_shots, mean_k, rec_k, true_k, error%
  · Bootstrap analysis: P(|rec_k - true_k| ≤ 1) vs n_shots across 1000 trials
  · Figure-4 data: convergence of mean_k towards true E[k]

Usage::

    cd /home/vadeo/catalyst
    python3 benchmark/run_e4_profile_guided.py

Optional flags::

    --batch-size  INT   shots per batch (default 10)
    --n-batches   INT   total batches   (default 50  → 500 shots)
    --n-bootstrap INT   bootstrap trials (default 1000)
    --seed        INT   RNG seed offset (default 7919)

Circuits:
    RUS  (wire=0 target, wire=1 ancilla): Geom(0.1464), true E[k] ≈ 6.83 → rec = 7
    MSD  (n_magic=7, p_err=0.10):         Geom(0.6049), true E[k] ≈ 1.65 → rec = 2
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import jax
import jax.numpy as jnp
import pennylane as qp

# Catalyst ops
from catalyst import cond, measure, qjit, while_loop

# Project-local modules
sys.path.insert(0, "/home/vadeo/catalyst/benchmark")
from gate_counter_estimator import GateCounterSession, RunResult
from profile_guided_estimator import ProfileGuidedEstimator, ProfileReport


# ── True distribution parameters ──────────────────────────────────────────────

def _rus_true_p() -> float:
    """P(success per RUS attempt) = (1 − 1/√2) / 2 ≈ 0.1464."""
    return (1.0 - 1.0 / math.sqrt(2)) / 2.0


def _msd_success_prob(n_magic: int, p_err: float) -> float:
    """P(even-error syndrome) = ½[(1-2p)^n + 1]."""
    return 0.5 * ((1.0 - 2.0 * p_err) ** n_magic + 1.0)


# ── Circuit builders ───────────────────────────────────────────────────────────

def _make_rus_circuit():
    """Return a zero-arg circuit function for the RUS while-loop."""
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

    return _circuit


def _make_msd_circuit(n_magic: int = 7, p_err: float = 0.10):
    """Return a key-arg circuit function for the MSD while-loop."""
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

    return _circuit


# ── Keyed session wrapper (for circuits that need a fresh JAX key per run) ────

class _KeyedSession:
    """Wraps GateCounterSession.run() to inject a new JAX PRNG key each call.

    ProfileGuidedEstimator calls sess.run() with no arguments; this adapter
    generates a fresh key from an incrementing counter so the MSD circuit
    (which needs JAX randomness) produces independent samples.
    """

    def __init__(self, inner_sess: GateCounterSession, seed_offset: int = 7919):
        self._inner = inner_sess
        self._offset = seed_offset
        self._n = 0

    def run(self) -> RunResult:
        key = jax.random.PRNGKey(self._n + self._offset)
        self._n += 1
        return self._inner.run(key)


# ── Trip-count extraction helpers ─────────────────────────────────────────────

def _estimate_trip_rus(gate_counts: dict) -> float:
    """Infer RUS trip count from per-run gate counts.

    All four non-branch per-iter gates are exact: T=1, H=2, CNOT=2, Measure=1.
    Weighted average: total_weight = 1+2+2+1 = 6; weighted_sum = 6k → tc = k.
    """
    T  = gate_counts.get("T_1", 0)
    H  = gate_counts.get("Hadamard_1", 0)
    CN = gate_counts.get("CNOT_2", 0)
    M  = gate_counts.get("Measure_1", 0)
    total = T + H + CN + M
    return total / 6.0 if total > 0 else 0.0


def _estimate_trip_msd(gate_counts: dict, n_magic: int = 7) -> float:
    """Infer MSD trip count from per-run gate counts.

    Non-branch per-iter: H=n, T=n, CNOT=n, Measure=n+1.
    Weighted average: total_weight = n+n+n+(n+1) = 4n+1; weighted_sum = (4n+1)k.
    """
    T  = gate_counts.get("T_1", 0)
    H  = gate_counts.get("Hadamard_1", 0)
    CN = gate_counts.get("CNOT_2", 0)
    M  = gate_counts.get("Measure_1", 0)
    total = T + H + CN + M
    weight = 4 * n_magic + 1  # 29 for n_magic=7
    return total / weight if weight > 0 else 0.0


# ── Bootstrap analysis ─────────────────────────────────────────────────────────

def bootstrap_accuracy(
    trip_counts: list,
    true_k: int,
    sample_sizes: list,
    n_trials: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """Bootstrap P(|round(mean_k_n) - true_k| ≤ 1) for each sample size.

    Parameters
    ----------
    trip_counts : list of floats — all observed trip counts from N shots
    true_k      : int           — round(true E[k])
    sample_sizes: list of int   — sample sizes to evaluate
    n_trials    : int           — bootstrap draws per sample size
    rng_seed    : int           — for reproducibility

    Returns
    -------
    dict mapping sample_size → P(within ±1)
    """
    rng = random.Random(rng_seed)
    n_total = len(trip_counts)
    result = {}
    for n in sample_sizes:
        n_within = 0
        for _ in range(n_trials):
            sample = rng.choices(trip_counts, k=min(n, n_total))
            mean_k = sum(sample) / len(sample)
            rec = max(1, round(mean_k))
            if abs(rec - true_k) <= 1:
                n_within += 1
        result[n] = n_within / n_trials
    return result


# ── Convergence runner ─────────────────────────────────────────────────────────

def run_convergence(
    circuit_name: str,
    per_iter_costs: dict,
    session_or_keyed,
    true_p: float,
    batch_size: int,
    n_batches: int,
    n_bootstrap: int,
    seed: int,
    trip_fn,
) -> list:
    """Run profile-guided convergence experiment for one circuit.

    Returns list of trip_counts (length = batch_size * n_batches).
    """
    true_E = 1.0 / true_p
    true_k = max(1, round(true_E))

    _hdr("Profile-Guided Convergence", circuit_name)
    print(f"  true p      = {true_p:.4f}")
    print(f"  true E[k]   = {true_E:.3f}    (analytical)")
    print(f"  true rec_k  = {true_k}         (round(E[k]))")
    print(f"  batch_size  = {batch_size}  shots")
    print(f"  n_batches   = {n_batches}  batches  ({batch_size*n_batches} total shots)")
    print()

    pge = ProfileGuidedEstimator(session_or_keyed, per_iter_costs=per_iter_costs)
    report = pge.run_batches(batch_size=batch_size, n_batches=n_batches)

    # Print convergence table
    _convergence_table(report, true_E, true_k)

    # Collect trip counts for bootstrap
    trip_counts = report.trip_counts

    # Bootstrap analysis
    sample_sizes = sorted({5, 10, 20, 50, 100, min(200, len(trip_counts)),
                           len(trip_counts)})
    sample_sizes = [s for s in sample_sizes if s <= len(trip_counts)]
    print()
    print("  Bootstrap accuracy  P(|rec_k − true_k| ≤ 1):")
    print(f"  {'n_shots':>8}  {'P(within ±1)':>14}  {'✓?' :>6}")
    print("  " + "-" * 34)
    acc = bootstrap_accuracy(trip_counts, true_k, sample_sizes, n_bootstrap, seed)
    for n, p_acc in acc.items():
        check = "✓" if p_acc >= 0.90 else ("·" if p_acc >= 0.80 else "✗")
        print(f"  {n:>8}  {p_acc:>14.3f}  {check:>6}")

    print()
    print(f"  Final estimate ({len(trip_counts)} shots):")
    print(f"    mean_k       = {report.mean_trip:.3f}  (true: {true_E:.3f})")
    print(f"    rec_k        = {report.recommended_iters}  (true: {true_k})")
    err_pct = abs(report.mean_trip - true_E) / true_E * 100
    print(f"    error        = {err_pct:.1f}%")

    return trip_counts


def _convergence_table(report: ProfileReport, true_E: float, true_k: int):
    print(f"  {'batch':>5}  {'n_shots':>7}  {'mean_k':>7}  {'±95%CI':>7}  "
          f"{'rec_k':>5}  {'true_k':>6}  {'err%':>6}")
    print("  " + "-" * 58)
    for cp in report.convergence:
        err_pct = abs(cp.mean_trip - true_E) / true_E * 100
        match = "✓" if cp.recommended_iters == true_k else ("~" if abs(cp.recommended_iters - true_k) <= 1 else "✗")
        print(
            f"  {cp.n_runs // (report.convergence[0].n_runs):>5}"
            f"  {cp.n_runs:>7}"
            f"  {cp.mean_trip:>7.3f}"
            f"  {cp.ci_95_half:>7.3f}"
            f"  {cp.recommended_iters:>5}"
            f"  {true_k:>6}"
            f"  {err_pct:>6.1f}"
            f"  {match}"
        )


def _hdr(label: str, name: str):
    width = 68
    sep = "=" * width
    print()
    print(sep)
    print(f"  {label}: {name}")
    print(sep)
    print()


# ── Figure-4 data printer ──────────────────────────────────────────────────────

def print_figure4_data(
    circuit_name: str,
    report_rus: ProfileReport,
    report_msd: ProfileReport,
    true_E_rus: float,
    true_E_msd: float,
):
    """Print tab-separated CSV suitable for plotting Figure 4."""
    print()
    print("=" * 68)
    print("  Figure 4 Data (tab-separated, suitable for gnuplot / matplotlib)")
    print("=" * 68)
    print()
    print("# n_shots\tRUS_mean_k\tRUS_ci95\tMSD_mean_k\tMSD_ci95")
    print(f"# true E[k]: RUS={true_E_rus:.3f}  MSD={true_E_msd:.3f}")

    # Align on n_shots (both should have same checkpoints)
    rus_by_n = {cp.n_runs: cp for cp in report_rus.convergence}
    msd_by_n = {cp.n_runs: cp for cp in report_msd.convergence}
    all_ns = sorted(set(rus_by_n) | set(msd_by_n))
    for n in all_ns:
        rc = rus_by_n.get(n)
        mc = msd_by_n.get(n)
        rus_m  = f"{rc.mean_trip:.4f}" if rc else "NA"
        rus_ci = f"{rc.ci_95_half:.4f}" if rc else "NA"
        msd_m  = f"{mc.mean_trip:.4f}" if mc else "NA"
        msd_ci = f"{mc.ci_95_half:.4f}" if mc else "NA"
        print(f"{n}\t{rus_m}\t{rus_ci}\t{msd_m}\t{msd_ci}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size",  type=int, default=10,
                    help="Shots per batch (default 10)")
    ap.add_argument("--n-batches",   type=int, default=50,
                    help="Total batches (default 50 → 500 shots)")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="Bootstrap trials (default 1000)")
    ap.add_argument("--seed",        type=int, default=7919,
                    help="RNG seed offset (default 7919)")
    args = ap.parse_args()

    n_magic = 7
    p_err   = 0.10

    true_p_rus = _rus_true_p()
    true_p_msd = _msd_success_prob(n_magic, p_err)
    true_E_rus = 1.0 / true_p_rus
    true_E_msd = 1.0 / true_p_msd

    print()
    print("=" * 68)
    print("  E4: Profile-Guided Refinement — Convergence Experiment")
    print("=" * 68)
    print(f"  RUS: true p={true_p_rus:.4f}, E[k]={true_E_rus:.3f}, rec={round(true_E_rus)}")
    print(f"  MSD: true p={true_p_msd:.4f}, E[k]={true_E_msd:.3f}, rec={round(true_E_msd)}")
    print()

    # ── RUS ──────────────────────────────────────────────────────────────────
    _hdr("Compiling", "RUS (2 qubits)")
    print("  [rus] compiling...", end=" ", flush=True)
    rus_dev = qp.device("lightning.qubit", wires=2)
    rus_circuit = _make_rus_circuit()

    with GateCounterSession(rus_circuit, rus_dev) as rus_sess:
        print("done.", flush=True)

        per_iter_rus = {
            "T_1":        1,
            "Hadamard_1": 2,
            "CNOT_2":     2,
            "Measure_1":  1,
        }
        rus_pge = ProfileGuidedEstimator(rus_sess, per_iter_costs=per_iter_rus)
        print(f"  [rus] running {args.batch_size * args.n_batches} shots "
              f"({args.n_batches} batches × {args.batch_size})...", end=" ", flush=True)
        report_rus = rus_pge.run_batches(
            batch_size=args.batch_size,
            n_batches=args.n_batches,
        )
        print("done.", flush=True)

    # ── MSD ──────────────────────────────────────────────────────────────────
    _hdr("Compiling", f"MSD (n_magic={n_magic}, p_err={p_err})")
    print("  [msd] compiling...", end=" ", flush=True)
    msd_dev = qp.device("lightning.qubit", wires=n_magic + 1)
    msd_circuit = _make_msd_circuit(n_magic=n_magic, p_err=p_err)
    init_key = jax.random.PRNGKey(0)

    with GateCounterSession(msd_circuit, msd_dev, init_key) as msd_sess:
        print("done.", flush=True)

        keyed = _KeyedSession(msd_sess, seed_offset=args.seed)
        per_iter_msd = {
            "T_1":        n_magic,
            "Hadamard_1": n_magic,
            "CNOT_2":     n_magic,
            "Measure_1":  n_magic + 1,
        }
        msd_pge = ProfileGuidedEstimator(keyed, per_iter_costs=per_iter_msd)
        print(f"  [msd] running {args.batch_size * args.n_batches} shots "
              f"({args.n_batches} batches × {args.batch_size})...", end=" ", flush=True)
        report_msd = msd_pge.run_batches(
            batch_size=args.batch_size,
            n_batches=args.n_batches,
        )
        print("done.", flush=True)

    # ── Results ───────────────────────────────────────────────────────────────
    _hdr("Convergence Table", "RUS — Geom(0.1464), true E[k]=6.83")
    print(f"  {'batch':>5}  {'n_shots':>7}  {'mean_k':>7}  {'±95%CI':>7}  "
          f"{'rec_k':>5}  {'true_k':>6}  {'err%':>6}")
    print("  " + "-" * 58)
    batch_sz = args.batch_size
    true_k_rus = max(1, round(true_E_rus))
    for i, cp in enumerate(report_rus.convergence, 1):
        err = abs(cp.mean_trip - true_E_rus) / true_E_rus * 100
        match = ("✓" if cp.recommended_iters == true_k_rus
                 else ("~" if abs(cp.recommended_iters - true_k_rus) <= 1 else "✗"))
        print(f"  {i:>5}  {cp.n_runs:>7}  {cp.mean_trip:>7.3f}"
              f"  {cp.ci_95_half:>7.3f}  {cp.recommended_iters:>5}"
              f"  {true_k_rus:>6}  {err:>6.1f}  {match}")

    _hdr("Convergence Table", f"MSD — Geom({true_p_msd:.4f}), true E[k]={true_E_msd:.3f}")
    print(f"  {'batch':>5}  {'n_shots':>7}  {'mean_k':>7}  {'±95%CI':>7}  "
          f"{'rec_k':>5}  {'true_k':>6}  {'err%':>6}")
    print("  " + "-" * 58)
    true_k_msd = max(1, round(true_E_msd))
    for i, cp in enumerate(report_msd.convergence, 1):
        err = abs(cp.mean_trip - true_E_msd) / true_E_msd * 100
        match = ("✓" if cp.recommended_iters == true_k_msd
                 else ("~" if abs(cp.recommended_iters - true_k_msd) <= 1 else "✗"))
        print(f"  {i:>5}  {cp.n_runs:>7}  {cp.mean_trip:>7.3f}"
              f"  {cp.ci_95_half:>7.3f}  {cp.recommended_iters:>5}"
              f"  {true_k_msd:>6}  {err:>6.1f}  {match}")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    n_total = args.batch_size * args.n_batches
    sample_sizes = [5, 10, 20, 30, 50, 100]
    sample_sizes = [s for s in sample_sizes if s <= n_total]

    _hdr("Bootstrap Analysis (1000 trials)", "P(|rec_k − true_k| ≤ 1)")
    print(f"  {'n_shots':>8}  {'RUS P(±1)':>10}  {'RUS ✓':>6}"
          f"  {'MSD P(±1)':>10}  {'MSD ✓':>6}")
    print("  " + "-" * 52)
    acc_rus = bootstrap_accuracy(
        report_rus.trip_counts, true_k_rus, sample_sizes, args.n_bootstrap, args.seed)
    acc_msd = bootstrap_accuracy(
        report_msd.trip_counts, true_k_msd, sample_sizes, args.n_bootstrap, args.seed + 1)
    for n in sample_sizes:
        pr = acc_rus.get(n, float("nan"))
        pm = acc_msd.get(n, float("nan"))
        cr = "✓" if pr >= 0.90 else ("·" if pr >= 0.80 else "✗")
        cm = "✓" if pm >= 0.90 else ("·" if pm >= 0.80 else "✗")
        print(f"  {n:>8}  {pr:>10.3f}  {cr:>6}  {pm:>10.3f}  {cm:>6}")

    # ── Figure-4 CSV ──────────────────────────────────────────────────────────
    print_figure4_data("", report_rus, report_msd, true_E_rus, true_E_msd)

    # ── Summary ───────────────────────────────────────────────────────────────
    _hdr("Summary", "E4 Profile-Guided Refinement Convergence")
    print(f"  RUS: {n_total} shots → mean_k={report_rus.mean_trip:.3f}"
          f"  rec={report_rus.recommended_iters}  (true: E[k]={true_E_rus:.3f}, rec={true_k_rus})")
    print(f"  MSD: {n_total} shots → mean_k={report_msd.mean_trip:.3f}"
          f"  rec={report_msd.recommended_iters}  (true: E[k]={true_E_msd:.3f}, rec={true_k_msd})")
    err_rus = abs(report_rus.mean_trip - true_E_rus) / true_E_rus * 100
    err_msd = abs(report_msd.mean_trip - true_E_msd) / true_E_msd * 100
    print(f"  RUS error: {err_rus:.1f}%    MSD error: {err_msd:.1f}%")
    # First batch where rec_k hits true_k
    def _first_hit(conv, true_k_, batch_sz_):
        for cp in conv:
            if cp.recommended_iters == true_k_:
                return cp.n_runs
        return None
    fh_rus = _first_hit(report_rus.convergence, true_k_rus, batch_sz)
    fh_msd = _first_hit(report_msd.convergence, true_k_msd, batch_sz)
    if fh_rus:
        print(f"  RUS: first rec_k={true_k_rus} at n_shots={fh_rus}")
    if fh_msd:
        print(f"  MSD: first rec_k={true_k_msd} at n_shots={fh_msd}")
    print()
    print("  DONE.")


if __name__ == "__main__":
    main()
