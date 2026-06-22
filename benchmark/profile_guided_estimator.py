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
"""Profile-guided iteration-count refinement for dynamic Catalyst circuits.

Overview
--------
The static ``resource-analysis`` pass reports per-iteration costs for dynamic
loops but cannot know how many iterations each loop will execute (the trip
count depends on mid-circuit measurement outcomes).  This module closes the
gap:

1. **Instrument** the circuit once with ``GateCounterSession`` to collect
   actual per-gate execution counts across many runs.

2. **Estimate trip count** per run: divide the observed total count of a
   reference gate by its known per-iteration count (from static analysis).

3. **Fit** a geometric distribution parameter ``p_hat = 1 / mean(trip_count)``
   (valid when each iteration succeeds independently with probability p).

4. **Report** the recommended ``estimated_iterations = round(1/p_hat)`` to
   annotate the while loop for the loop-peeling pass.

5. **Convergence** table: show how ``p_hat`` stabilises over successive
   batches so the paper can claim "X runs suffice to within ±Y%".

Usage::

    from gate_counter_estimator import GateCounterSession
    from profile_guided_estimator import ProfileGuidedEstimator

    def _rus_circuit():
        ...  # your circuit body

    dev = qp.device("lightning.qubit", wires=2)

    # per-iteration gate costs from static analysis for the RUS loop body
    per_iter = {"Hadamard_1": 2, "T_1": 1, "Measure_1": 1}

    with GateCounterSession(_rus_circuit, dev) as sess:
        pge = ProfileGuidedEstimator(sess, per_iter_costs=per_iter)
        report = pge.run(n_runs=100)

    print(report.summary())
    print(f"Recommended annotation:  estimated_iterations = {report.recommended_iters}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Result of one instrumented run."""
    gate_counts: Dict[str, int]
    estimated_trip_count: float     # inferred from gate counts


@dataclass
class ConvergencePoint:
    """Running statistics after processing `n_runs_so_far` runs."""
    n_runs: int
    mean_trip: float
    p_hat: float
    recommended_iters: int
    std_trip: float
    ci_95_half: float               # 1.96 * std / sqrt(n)


@dataclass
class ProfileReport:
    """Full report from a profile-guided estimation run."""
    per_iter_costs: Dict[str, int]
    records: List[RunRecord]
    convergence: List[ConvergencePoint]

    @property
    def trip_counts(self) -> List[float]:
        return [r.estimated_trip_count for r in self.records]

    @property
    def mean_trip(self) -> float:
        tc = self.trip_counts
        return sum(tc) / len(tc) if tc else 0.0

    @property
    def std_trip(self) -> float:
        tc = self.trip_counts
        n = len(tc)
        if n < 2:
            return 0.0
        mu = self.mean_trip
        return math.sqrt(sum((x - mu) ** 2 for x in tc) / (n - 1))

    @property
    def p_hat(self) -> float:
        m = self.mean_trip
        return 1.0 / m if m > 0 else float("nan")

    @property
    def recommended_iters(self) -> int:
        return max(1, round(self.mean_trip))

    def max_iterations(self, c: float = 3.0) -> int:
        """Return ceil(E[k] + c * sigma) — the depth bound for confidence c.

        This value should be written as ``max_iterations = N`` on the target
        ``scf.while`` op so that ``--depth-bounding`` guarantees termination
        within N iterations with probability ≥ 1 − ε(c) under the fitted
        geometric distribution.

        Parameters
        ----------
        c :
            Number of standard deviations above the mean.  Default 3.0 gives
            P(k ≤ MAX_ITER) ≈ 99.7% for a Normal approximation; the geometric
            tail is heavier, so this is a lower bound on coverage.
        """
        return math.ceil(self.mean_trip + c * self.std_trip)

    def summary(self, true_p: Optional[float] = None) -> str:
        lines = [
            "── Profile-Guided Estimation Report ────────────────────────────",
            f"  runs          : {len(self.records)}",
            f"  mean trip     : {self.mean_trip:.3f}",
            f"  std  trip     : {self.std_trip:.3f}",
            f"  p_hat (geom.) : {self.p_hat:.4f}",
            f"  recommended   : estimated_iterations = {self.recommended_iters}",
        ]
        if true_p is not None:
            true_mean = 1.0 / true_p
            err_pct = abs(self.mean_trip - true_mean) / true_mean * 100
            lines.append(f"  true p        : {true_p:.4f}  (mean={true_mean:.2f})")
            lines.append(f"  error vs true : {err_pct:.1f}%")

        if self.convergence:
            lines.append("")
            lines.append(f"  {'runs':>6}  {'mean':>7}  {'p_hat':>7}  {'±95% CI':>8}  {'rec_k':>5}")
            lines.append("  " + "-" * 44)
            for cp in self.convergence:
                lines.append(
                    f"  {cp.n_runs:>6}  {cp.mean_trip:>7.3f}  {cp.p_hat:>7.4f}"
                    f"  {cp.ci_95_half:>8.3f}  {cp.recommended_iters:>5}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProfileGuidedEstimator
# ---------------------------------------------------------------------------

class ProfileGuidedEstimator:
    """Infer loop iteration counts from runtime gate counters.

    Parameters
    ----------
    session :
        An already-entered ``GateCounterSession`` (use inside a ``with``
        block or call ``session.__enter__()`` first).
    per_iter_costs :
        Dict mapping ``gate_label`` (as it appears in the gate counter
        manifest, e.g. ``"Hadamard_1"``) to the number of times that gate
        executes per *one* iteration of the target dynamic loop.

        Obtained from the static resource analysis:  look at the
        ``dyn_while_loop_N`` entry in the ``DynamicResourceEstimator``
        output and read the per-iteration gate counts there.

        Only gates with a non-zero per-iteration count should be listed.
    """

    def __init__(self, session, per_iter_costs: Dict[str, int]):
        self._session = session
        self._per_iter = {k: v for k, v in per_iter_costs.items() if v > 0}
        if not self._per_iter:
            raise ValueError("per_iter_costs must contain at least one gate with count > 0")

    # ── public API ─────────────────────────────────────────────────────────

    def run(
        self,
        n_runs: int = 100,
        convergence_checkpoints: Optional[List[int]] = None,
    ) -> ProfileReport:
        """Run the circuit ``n_runs`` times and fit iteration-count distribution.

        Parameters
        ----------
        n_runs :
            Total number of circuit executions.
        convergence_checkpoints :
            List of run counts at which to record a ``ConvergencePoint``.
            Defaults to every 10 runs up to ``n_runs``.

        Returns
        -------
        ProfileReport
        """
        if convergence_checkpoints is None:
            step = max(1, n_runs // 10)
            convergence_checkpoints = list(range(step, n_runs + 1, step))
            if convergence_checkpoints[-1] != n_runs:
                convergence_checkpoints.append(n_runs)

        records: List[RunRecord] = []
        convergence: List[ConvergencePoint] = []
        checkpoint_set = set(convergence_checkpoints)

        for i in range(1, n_runs + 1):
            run_result = self._session.run()
            tc = self._estimate_trip(run_result.gate_counts)
            records.append(RunRecord(gate_counts=run_result.gate_counts,
                                     estimated_trip_count=tc))

            if i in checkpoint_set:
                convergence.append(self._make_checkpoint(records))

        return ProfileReport(
            per_iter_costs=dict(self._per_iter),
            records=records,
            convergence=convergence,
        )

    def run_batches(
        self,
        batch_size: int = 10,
        n_batches: int = 10,
    ) -> ProfileReport:
        """Run in batches, recording a convergence point after each batch."""
        n_runs = batch_size * n_batches
        checkpoints = [batch_size * i for i in range(1, n_batches + 1)]
        return self.run(n_runs=n_runs, convergence_checkpoints=checkpoints)

    # ── internals ──────────────────────────────────────────────────────────

    def _estimate_trip(self, gate_counts: Dict[str, int]) -> float:
        """Infer trip count from observed gate counts.

        Uses every known per-iter gate as an independent estimator and
        returns their weighted average (weight = per_iter count, so
        higher-frequency gates dominate and reduce noise).
        """
        total_weight = 0
        weighted_sum = 0.0
        for gate, per_iter in self._per_iter.items():
            observed = gate_counts.get(gate, 0)
            if per_iter > 0:
                weighted_sum += observed  # raw count
                total_weight += per_iter  # expected per-iter weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _make_checkpoint(records: List[RunRecord]) -> ConvergencePoint:
        n = len(records)
        tcs = [r.estimated_trip_count for r in records]
        mu = sum(tcs) / n
        var = sum((x - mu) ** 2 for x in tcs) / max(n - 1, 1)
        std = math.sqrt(var)
        ci_half = 1.96 * std / math.sqrt(n) if n > 1 else float("inf")
        p_hat = 1.0 / mu if mu > 0 else float("nan")
        return ConvergencePoint(
            n_runs=n,
            mean_trip=mu,
            p_hat=p_hat,
            recommended_iters=max(1, round(mu)),
            std_trip=std,
            ci_95_half=ci_half,
        )


# ---------------------------------------------------------------------------
# Convenience: derive per_iter_costs from a ResourceReport
# ---------------------------------------------------------------------------

def per_iter_from_report(report, loop_name: str) -> Dict[str, int]:
    """Extract per-iteration gate costs for ``loop_name`` from a ResourceReport.

    ``loop_name`` is e.g. ``"dyn_while_loop_1"`` as it appears in the
    ``DynamicResourceEstimator`` output.

    Returns a dict suitable for passing to ``ProfileGuidedEstimator``.

    The gate labels use the format ``"GateName_wires"`` as produced by the
    gate counter instrumentation pass (e.g. ``"Hadamard_1"``, ``"CNOT_2"``).
    """
    fn = report.entries.get(loop_name)
    if fn is None:
        raise KeyError(f"Loop body '{loop_name}' not found in report. "
                       f"Available: {list(report.entries)}")
    costs: Dict[str, int] = {}
    for gate_name, count in fn.operations.items():
        # gate_name from analysis: "Hadamard(1)" → label: "Hadamard_1"
        # gate_name from analysis: "CNOT(2)"     → label: "CNOT_2"
        label = _analysis_name_to_counter_label(gate_name)
        if label and count > 0:
            costs[label] = count
    # Also add measurements.
    for meas_name, count in fn.measurements.items():
        if "MidCircuit" in meas_name and count > 0:
            costs["Measure_1"] = costs.get("Measure_1", 0) + count
    return costs


def annotate_mlir_max_iterations(mlir_text: str, max_iter: int) -> str:
    """Add ``max_iterations = N : i64`` to the first ``scf.while`` in *mlir_text*.

    Injects ``attributes {max_iterations = N : i64}`` at the end of the first
    ``scf.while`` op (after the closing ``}`` of its ``do`` region), so the
    patched text is accepted by::

        quantum-opt --depth-bounding <patched.mlir>

    Parameters
    ----------
    mlir_text :
        The MLIR module text, e.g. from ``qjit_fn.mlir``.
    max_iter :
        The integer bound (typically ``ProfileReport.max_iterations(c)``).

    Returns
    -------
    str
        Modified MLIR text with the attribute appended after the do-region.
    """
    # Find the first "scf.while".
    while_pos = mlir_text.find("scf.while")
    if while_pos == -1:
        raise ValueError("No 'scf.while' found in the provided MLIR text.")

    # Locate the "} do {" junction between before and after regions.
    do_marker = "} do {"
    do_pos = mlir_text.find(do_marker, while_pos)
    if do_pos == -1:
        raise ValueError("No '} do {' found after 'scf.while' — already lowered?")

    # Count braces from after "} do {" to find the end of the do region.
    search_start = do_pos + len(do_marker)
    depth = 1
    i = search_start
    while i < len(mlir_text) and depth > 0:
        ch = mlir_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    # i-1 is the position of the closing '}' of the do region.
    close_pos = i - 1

    attr_str = f" attributes {{max_iterations = {max_iter} : i64}}"
    return mlir_text[: close_pos + 1] + attr_str + mlir_text[close_pos + 1 :]


def _analysis_name_to_counter_label(analysis_name: str) -> Optional[str]:
    """Convert analysis gate name to gate counter label.

    Analysis: "Hadamard(1)" → counter: "Hadamard_1"
    Analysis: "CNOT(2)"     → counter: "CNOT_2"
    Analysis: "Adjoint(T)(1)" → counter: "Adjoint_T__1"  (sanitised)
    """
    import re
    # Pattern: "Name(wires)" — the last parenthesised group is the wire count.
    m = re.match(r"^(.+)\((\d+)\)$", analysis_name.strip())
    if not m:
        return None
    name_part = m.group(1)
    wires = m.group(2)
    # Sanitise name: keep alnum and underscore only (mirrors GateCounterPass).
    sanitised = re.sub(r"[^A-Za-z0-9_]", "_", name_part)
    return f"{sanitised}_{wires}"
