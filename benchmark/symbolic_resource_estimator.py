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
"""Symbolic resource analysis for parameterised Catalyst quantum circuits.

Problem
-------
The static ``resource-analysis`` pass reports concrete gate counts for a
fixed circuit size n.  For circuits that are parameterised by n (QFT,
iterative QPE, BBHT), the resource analyst wants a *symbolic* formula such as

    CNOT(n)  ≈  n² / 2          (QFT, quadratic)
    CRZ(n)   =  2^n − 1         (iterative QPE, exponential)
    H(n)     ≈  (π/4)·√(2^n)·k  (BBHT, sub-exponential)

This module:

1. Runs ``DynamicResourceEstimator`` for n ∈ n_values (e.g. [2,4,6,8]).
2. Collects the flattened total gate count of a chosen gate for each n.
3. Fits candidate symbolic formulas using least-squares (log-space for
   exponentials).
4. Returns a ``SymbolicFormula`` with the formula string, coefficients, and
   goodness-of-fit (R²).

Supported formula families (tried in order; best R² wins):
  - Linear:       a·n + b
  - Quadratic:    a·n² + b·n + c
  - n·log2(n):    a·n·log2(n) + b·n + c
  - Exponential:  a·2^n + b
  - Sqrt-exp:     a·√(2^n) + b          (BBHT-style)
  - n²·log2(n):   a·n²·log2(n) + b·n²  (e.g. some QFT decompositions)

Usage::

    from dynamic_resource_estimator import DynamicResourceEstimator
    from symbolic_resource_estimator import SymbolicResourceEstimator

    def make_qft(n):
        \"\"\"Return a compiled QFT circuit for n qubits.\"\"\"
        ...

    sre = SymbolicResourceEstimator(DynamicResourceEstimator())

    formula = sre.fit(
        circuit_factory=make_qft,
        n_values=[2, 3, 4, 6, 8],
        gate="CNOT(2)",           # gate name as it appears in analysis output
        expected_iters_fn=None,   # None = circuit is fully static
    )
    print(formula)
    # → SymbolicFormula(family='quadratic', expr='0.50·n² + 0.50·n', R²=1.000)

    # Verify at n=10
    print(formula.evaluate(10))   # → 55.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SymbolicFormula:
    """Best-fit symbolic formula for a gate count as a function of n."""

    family: str          # 'linear', 'quadratic', 'n_log_n', 'exponential', ...
    expr: str            # Human-readable formula string
    coeffs: List[float]  # Raw coefficient vector (family-specific order)
    r_squared: float     # Goodness-of-fit on the fitting data
    n_values: List[int]  # The n values used for fitting
    gate_counts: List[float]  # The observed gate counts at those n values

    def evaluate(self, n: float) -> float:
        """Evaluate the formula at a given n."""
        return _FAMILY_EVAL[self.family](self.coeffs, n)

    def __str__(self) -> str:
        return (
            f"SymbolicFormula(family={self.family!r}, "
            f"expr={self.expr!r}, R²={self.r_squared:.4f})"
        )

    def summary_table(self) -> str:
        """Return a table comparing formula vs. observed values."""
        lines = [
            f"  Gate count formula: {self.expr}",
            f"  Fit quality:        R² = {self.r_squared:.4f}",
            f"",
            f"  {'n':>5}  {'observed':>10}  {'formula':>10}  {'error%':>8}",
            f"  {'─'*40}",
        ]
        for n, obs in zip(self.n_values, self.gate_counts):
            pred = self.evaluate(n)
            err = abs(pred - obs) / max(obs, 1) * 100
            lines.append(f"  {n:>5}  {obs:>10.1f}  {pred:>10.1f}  {err:>7.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formula families and fitting
# ---------------------------------------------------------------------------

def _fit_ols(X: List[List[float]], y: List[float]) -> Tuple[List[float], float]:
    """Ordinary least squares via normal equations X^T X c = X^T y.

    Returns (coefficients, R²).  X is a list of row vectors (one per data
    point); y is the target vector.  No external dependencies needed.
    """
    n = len(y)
    m = len(X[0])

    # XtX and Xty
    XtX = [[0.0] * m for _ in range(m)]
    Xty = [0.0] * m
    for xi, yi in zip(X, y):
        for i in range(m):
            Xty[i] += xi[i] * yi
            for j in range(m):
                XtX[i][j] += xi[i] * xi[j]

    # Gaussian elimination with partial pivoting
    # Augmented matrix [XtX | Xty]
    aug = [XtX[i][:] + [Xty[i]] for i in range(m)]
    for col in range(m):
        pivot = max(range(col, m), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return [0.0] * m, 0.0  # singular / underdetermined
        aug[col] = [v / aug[col][col] for v in aug[col]]
        for row in range(m):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(m + 1)]

    coeffs = [aug[i][m] for i in range(m)]

    # R²
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    preds = [sum(c * xi for c, xi in zip(coeffs, row)) for row in X]
    ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    return coeffs, r2


# Each family: (feature_fn, eval_fn, label_fn)
# feature_fn(n) -> feature vector (list of floats)
# eval_fn(coeffs, n) -> predicted count
# label_fn(coeffs) -> human-readable string

def _linear_feat(n):     return [float(n), 1.0]
def _linear_eval(c, n):  return c[0] * n + c[1]
def _linear_label(c):    return f"{c[0]:.3g}·n + {c[1]:.3g}"

def _quad_feat(n):        return [float(n ** 2), float(n), 1.0]
def _quad_eval(c, n):     return c[0] * n**2 + c[1] * n + c[2]
def _quad_label(c):       return f"{c[0]:.3g}·n² + {c[1]:.3g}·n + {c[2]:.3g}"

def _nlogn_feat(n):       return [n * math.log2(n) if n > 1 else 0, float(n), 1.0]
def _nlogn_eval(c, n):    return c[0] * (n * math.log2(n) if n > 1 else 0) + c[1]*n + c[2]
def _nlogn_label(c):      return f"{c[0]:.3g}·n·log₂n + {c[1]:.3g}·n + {c[2]:.3g}"

def _n2logn_feat(n):      return [n**2 * math.log2(n) if n > 1 else 0, float(n**2), float(n)]
def _n2logn_eval(c, n):   return c[0]*(n**2*math.log2(n) if n > 1 else 0) + c[1]*n**2 + c[2]*n
def _n2logn_label(c):     return f"{c[0]:.3g}·n²·log₂n + {c[1]:.3g}·n² + {c[2]:.3g}·n"

def _exp_feat(n):         return [float(2**n), 1.0]
def _exp_eval(c, n):      return c[0] * 2**n + c[1]
def _exp_label(c):        return f"{c[0]:.3g}·2ⁿ + {c[1]:.3g}"

def _sqrtexp_feat(n):     return [math.sqrt(2**n), 1.0]
def _sqrtexp_eval(c, n):  return c[0] * math.sqrt(2**n) + c[1]
def _sqrtexp_label(c):    return f"{c[0]:.3g}·√(2ⁿ) + {c[1]:.3g}"

_FAMILIES = [
    ("linear",    _linear_feat,  _linear_eval,  _linear_label),
    ("quadratic", _quad_feat,    _quad_eval,     _quad_label),
    ("n_log_n",   _nlogn_feat,   _nlogn_eval,    _nlogn_label),
    ("n2_log_n",  _n2logn_feat,  _n2logn_eval,   _n2logn_label),
    ("exponential", _exp_feat,   _exp_eval,      _exp_label),
    ("sqrt_exp",  _sqrtexp_feat, _sqrtexp_eval,  _sqrtexp_label),
]

_FAMILY_EVAL = {name: ev for name, _, ev, _ in _FAMILIES}


def fit_formula(n_values: List[int], counts: List[float]) -> SymbolicFormula:
    """Fit the best symbolic formula for (n_values, counts).

    Tries all families; returns the one with highest R².
    Requires at least 2 data points.
    """
    if len(n_values) < 2:
        raise ValueError("Need at least 2 data points to fit a formula.")

    best_r2 = -float("inf")
    best = None

    for name, feat_fn, eval_fn, label_fn in _FAMILIES:
        X = [feat_fn(n) for n in n_values]
        # Skip if feature matrix has too many columns for the data
        if len(X[0]) > len(n_values):
            continue
        try:
            coeffs, r2 = _fit_ols(X, counts)
        except Exception:
            continue
        if r2 > best_r2:
            best_r2 = r2
            best = (name, coeffs, label_fn(coeffs))

    if best is None:
        raise RuntimeError("All formula families failed to fit.")

    name, coeffs, expr = best
    return SymbolicFormula(
        family=name,
        expr=expr,
        coeffs=coeffs,
        r_squared=best_r2,
        n_values=list(n_values),
        gate_counts=list(counts),
    )


# ---------------------------------------------------------------------------
# SymbolicResourceEstimator
# ---------------------------------------------------------------------------

class SymbolicResourceEstimator:
    """Fit symbolic gate-count formulas for n-parameterised circuits.

    Parameters
    ----------
    estimator :
        A ``DynamicResourceEstimator`` instance.
    """

    def __init__(self, estimator):
        self._est = estimator

    def fit(
        self,
        circuit_factory: Callable[[int], object],
        n_values: List[int],
        gate: str,
        expected_iters_fn: Optional[Callable[[int], Dict[str, int]]] = None,
        verbose: bool = False,
    ) -> SymbolicFormula:
        """Fit a symbolic formula for ``gate`` count as a function of n.

        Parameters
        ----------
        circuit_factory :
            ``circuit_factory(n)`` must return a compiled ``@qjit`` function
            (already called once to populate ``.mlir``).
        n_values :
            List of problem sizes to evaluate (e.g. ``[2, 4, 6, 8]``).
        gate :
            Gate name as it appears in the analysis output
            (e.g. ``"CNOT(2)"``, ``"Hadamard(1)"``).
        expected_iters_fn :
            Optional callable ``expected_iters_fn(n) -> {loop_name: k}``
            for circuits with dynamic loops (passed to
            ``ResourceReport.with_expected_iters``).
        verbose :
            Print analysis output for each n when True.

        Returns
        -------
        SymbolicFormula
        """
        counts: List[float] = []
        for n in n_values:
            c = self._count_gate(circuit_factory, n, gate, expected_iters_fn, verbose)
            counts.append(float(c))
            if verbose:
                print(f"  n={n}: {gate} = {c}")

        return fit_formula(n_values, counts)

    def fit_all_gates(
        self,
        circuit_factory: Callable[[int], object],
        n_values: List[int],
        expected_iters_fn: Optional[Callable[[int], Dict[str, int]]] = None,
        min_count_threshold: int = 1,
    ) -> Dict[str, SymbolicFormula]:
        """Fit symbolic formulas for every gate in the circuit.

        Returns a dict mapping gate name → SymbolicFormula.
        Only gates whose total count is ≥ ``min_count_threshold`` at every n
        are included.
        """
        # Collect all gate names from the first n value.
        n0 = n_values[0]
        report0 = self._run_analysis(circuit_factory, n0, None)
        entry0 = self._get_flat_entry(report0)
        gates = list(entry0.operations.keys()) if entry0 else []

        formulas: Dict[str, SymbolicFormula] = {}
        for gate in gates:
            try:
                counts: List[float] = []
                for n in n_values:
                    c = self._count_gate(circuit_factory, n, gate, expected_iters_fn,
                                         verbose=False)
                    counts.append(float(c))
                if all(c >= min_count_threshold for c in counts):
                    formulas[gate] = fit_formula(n_values, counts)
            except Exception:
                pass  # skip gates where fitting fails

        return formulas

    def verify_against_gate_counter(
        self,
        circuit_factory: Callable[[int], object],
        n_values: List[int],
        gate: str,
        formula: SymbolicFormula,
        session_factory: Optional[Callable] = None,
    ) -> Dict[str, object]:
        """Compare symbolic formula predictions with runtime gate counter.

        Parameters
        ----------
        session_factory :
            Callable ``session_factory(n) -> GateCounterSession``.
            If None, only the static analysis comparison is made.

        Returns a dict with keys ``n``, ``formula_pred``, ``static_count``,
        and optionally ``runtime_count``.
        """
        rows = []
        for n in n_values:
            pred = formula.evaluate(n)
            static = self._count_gate(circuit_factory, n, gate, None, verbose=False)
            row = {"n": n, "formula_pred": pred, "static_count": static}
            if session_factory is not None:
                sess = session_factory(n)
                label = _analysis_name_to_counter_label(gate)
                if label:
                    result = sess.run()
                    row["runtime_count"] = result.gate_counts.get(label, -1)
            rows.append(row)
        return rows

    # ── internals ─────────────────────────────────────────────────────────

    def _run_analysis(self, circuit_factory, n, expected_iters_fn):
        qjit_fn = circuit_factory(n)
        return self._est.analyse(qjit_fn)

    def _get_flat_entry(self, report):
        """Return the flattened ResourceResult for the entry function."""
        from dynamic_resource_estimator import ResourceReport
        # The entry function's flattened view includes all loop bodies.
        # We use the DynamicResourceEstimator's getFlattenedResource via
        # the raw JSON's first qnode entry.
        for name, fn in report.entries.items():
            if fn.qnode:
                return fn
        # Fallback: return first entry.
        return next(iter(report.entries.values()), None)

    def _count_gate(self, circuit_factory, n, gate, expected_iters_fn, verbose):
        """Return the total gate count for ``gate`` in a circuit of size n.

        Uses with_expected_iters which handles both static loop flattening
        (function_calls) and dynamic loop expansion (var_function_calls).
        """
        report = self._run_analysis(circuit_factory, n, expected_iters_fn)
        if verbose:
            print(report.summary())

        iters = expected_iters_fn(n) if expected_iters_fn is not None else {}
        totals = report.with_expected_iters(iters)

        # Prefer the qnode entry function's flattened total.
        for fn in report.entry_functions():
            if fn.name in totals:
                return totals[fn.name].get(gate, 0)
        # Fallback: max across all functions (conservative).
        return max((g.get(gate, 0) for g in totals.values()), default=0)


# ---------------------------------------------------------------------------
# Convenience: compare symbolic formula to gate counter runtime values
# ---------------------------------------------------------------------------

def print_verification_table(
    formula: SymbolicFormula,
    n_values: List[int],
    runtime_counts: Optional[Dict[int, float]] = None,
):
    """Print a verification table for a symbolic formula.

    Parameters
    ----------
    runtime_counts :
        Optional dict {n: gate_count_from_hardware} for E5 experiment.
    """
    header = f"{'n':>5}  {'static':>10}  {'formula':>10}  {'err%':>7}"
    if runtime_counts:
        header += f"  {'runtime':>10}  {'rt_err%':>7}"
    print(header)
    print("─" * (len(header) + 2))

    for i, n in enumerate(formula.n_values):
        obs = formula.gate_counts[i]
        pred = formula.evaluate(n)
        err = abs(pred - obs) / max(obs, 1) * 100
        row = f"{n:>5}  {obs:>10.1f}  {pred:>10.1f}  {err:>6.1f}%"
        if runtime_counts and n in runtime_counts:
            rt = runtime_counts[n]
            rt_err = abs(pred - rt) / max(rt, 1) * 100
            row += f"  {rt:>10.1f}  {rt_err:>6.1f}%"
        print(row)

    print()
    print(f"Formula: {formula.expr}")
    print(f"R²:      {formula.r_squared:.4f}")


def _analysis_name_to_counter_label(analysis_name: str) -> Optional[str]:
    """Convert analysis gate name to gate counter label (mirrors profile_guided_estimator)."""
    import re
    m = re.match(r"^(.+)\((\d+)\)$", analysis_name.strip())
    if not m:
        return None
    name_part, wires = m.group(1), m.group(2)
    sanitised = re.sub(r"[^A-Za-z0-9_]", "_", name_part)
    return f"{sanitised}_{wires}"
