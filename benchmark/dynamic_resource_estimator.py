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
"""Dynamic resource estimator for Catalyst quantum circuits.

Strategy
--------
Static analysis via the ``resource-analysis`` MLIR pass gives exact gate
counts for static loops and per-iteration costs for dynamic loops.  For
dynamic loops (``var_function_calls``), the pass cannot know the trip
count, so this estimator lets the caller supply *expected* iteration
counts and computes estimated totals.

Two public entry points:

  ``DynamicResourceEstimator.analyse(qjit_fn, *args)``
      Compile the function (if not already compiled) and run the
      resource-analysis MLIR pass.  Returns a ``ResourceReport``.

  ``ResourceReport.with_expected_iters(mapping)``
      Given a dict ``{loop_body_name: expected_iters}``, compute
      estimated total gate counts by multiplying per-iteration costs
      through the call graph.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FunctionReport:
    """Resource data for a single function / loop-body entry."""

    name: str
    num_qubits: int
    num_alloc_qubits: int
    num_arg_qubits: int
    operations: Dict[str, int]
    measurements: Dict[str, int]
    function_calls: Dict[str, int]          # static calls: {callee: count}
    var_function_calls: Dict[str, str]      # dynamic: {callee: hex-id}
    has_branches: bool
    qnode: bool
    device_name: str

    def is_dynamic(self) -> bool:
        """True if this function contains at least one dynamic loop."""
        return bool(self.var_function_calls)

    def total_direct_gates(self) -> int:
        return sum(self.operations.values())


@dataclass
class ResourceReport:
    """Full resource report for a compiled Catalyst module."""

    entries: Dict[str, FunctionReport]      # function_name → FunctionReport
    raw_json: Dict[str, Any]               # unmodified pass output

    # ── helpers ────────────────────────────────────────────────────────────

    def entry_functions(self) -> list[FunctionReport]:
        """Functions annotated as qnodes or that have qubit arguments."""
        return [e for e in self.entries.values() if e.qnode or e.num_arg_qubits > 0]

    def dynamic_loops(self) -> Dict[str, list[str]]:
        """Return {parent_fn: [dynamic_child_names]} for all dynamic loops."""
        result = {}
        for name, fn in self.entries.items():
            if fn.var_function_calls:
                result[name] = list(fn.var_function_calls.keys())
        return result

    def with_expected_iters(
        self, expected_iters: Dict[str, int]
    ) -> Dict[str, Dict[str, int]]:
        """Compute estimated total gate counts given expected loop iterations.

        Args:
            expected_iters: mapping from *loop body name* (e.g.
                ``"dyn_for_loop_1"``, ``"dyn_while_loop_1"``) to the
                expected number of iterations.

        Returns:
            Dict mapping every function name to its estimated gate dict,
            where dynamic loop bodies are multiplied by the provided
            expected iteration count.
        """
        totals: Dict[str, Dict[str, int]] = {}

        def _compute(name: str, visited: set) -> Dict[str, int]:
            if name in totals:
                return totals[name]
            if name in visited:
                # Recursive call — can't flatten; return empty to match pass warning.
                return {}
            if name not in self.entries:
                return {}

            visited = visited | {name}
            fn = self.entries[name]
            gates: Dict[str, int] = dict(fn.operations)

            # Static callees.
            for callee, count in fn.function_calls.items():
                for gate, n in _compute(callee, visited).items():
                    gates[gate] = gates.get(gate, 0) + count * n

            # Dynamic callees.
            for callee in fn.var_function_calls:
                iters = expected_iters.get(callee, None)
                if iters is None:
                    continue  # unknown — skip
                for gate, n in _compute(callee, visited).items():
                    gates[gate] = gates.get(gate, 0) + iters * n

            totals[name] = gates
            return gates

        for name in self.entries:
            _compute(name, set())

        return totals

    def summary(self, expected_iters: Optional[Dict[str, int]] = None) -> str:
        """Human-readable summary string."""
        lines = ["── Dynamic Resource Report ──────────────────────────────"]

        for name, fn in self.entries.items():
            lines.append(f"\n  [{name}]  qubits={fn.num_qubits}  qnode={fn.qnode}")
            if fn.operations:
                lines.append("    gates (direct):")
                for g, n in sorted(fn.operations.items()):
                    lines.append(f"      {g}: {n}")
            if fn.measurements:
                lines.append("    measurements:")
                for m, n in sorted(fn.measurements.items()):
                    lines.append(f"      {m}: {n}")
            if fn.function_calls:
                lines.append("    static calls:")
                for c, n in sorted(fn.function_calls.items()):
                    lines.append(f"      {c}: ×{n}")
            if fn.var_function_calls:
                lines.append("    dynamic loops (unknown trip count):")
                for c in sorted(fn.var_function_calls):
                    ei = (expected_iters or {}).get(c)
                    tag = f"  [expected ×{ei}]" if ei is not None else "  [count unknown]"
                    lines.append(f"      {c}{tag}")

        if expected_iters:
            totals = self.with_expected_iters(expected_iters)
            lines.append("\n  ── Estimated totals (with expected iterations) ──")
            for name, gates in totals.items():
                if gates:
                    lines.append(f"  [{name}]: " + ", ".join(
                        f"{g}×{n}" for g, n in sorted(gates.items())
                    ))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class DynamicResourceEstimator:
    """Run the ``resource-analysis`` MLIR pass on a compiled ``@qjit`` function.

    Usage::

        from catalyst import qjit
        from dynamic_resource_estimator import DynamicResourceEstimator

        @qjit
        def my_circuit():
            ...

        # Trigger compilation (needed to populate mlir_module).
        my_circuit()

        est = DynamicResourceEstimator()
        report = est.analyse(my_circuit)
        print(report.summary(expected_iters={"dyn_for_loop_1": 5}))
    """

    def analyse(self, qjit_fn, *args) -> ResourceReport:
        """Compile (if needed) and run resource-analysis on *qjit_fn*.

        If *args* are provided the function is called with them first to
        ensure the MLIR module is populated.
        """
        # Ensure the MLIR is compiled.
        if args:
            qjit_fn(*args)

        mlir_text = qjit_fn.mlir
        # The MLIR from @qjit wraps the circuit in a nested module named
        # @module__circuit.  The resource-analysis pass only scans top-level
        # func.func ops, so we extract that inner module and pipe it instead.
        circuit_mlir = self._extract_circuit_module(mlir_text)
        raw = self._run_pass(circuit_mlir)
        entries = self._parse(raw)
        return ResourceReport(entries=entries, raw_json=raw)

    @staticmethod
    def _extract_circuit_module(mlir_text: str) -> str:
        """Extract the nested ``module @module_<name> { … }`` block.

        Catalyst wraps each @qjit function's quantum circuit in a nested module
        named @module_<fn_name> (e.g. @module_rus, @module_coin_flip).
        The resource-analysis pass only scans top-level func.func ops, so we
        extract that inner module.  If no such block is found the original text
        is returned unchanged.
        """
        import re
        match = re.search(r'\bmodule @module_\w+ \{', mlir_text)
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
                    return mlir_text[start:i + 1].strip()
            i += 1

        return mlir_text

    # ── internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _run_pass(mlir_text: str) -> Dict[str, Any]:
        """Pipe *mlir_text* through ``resource-analysis{output-json=true}``."""
        from catalyst.compiler import _quantum_opt

        output = _quantum_opt(
            "--pass-pipeline",
            "builtin.module(resource-analysis{output-json=true})",
            stdin=mlir_text,
        )
        return DynamicResourceEstimator._extract_json(output)

    @staticmethod
    def _extract_json(output: str) -> Dict[str, Any]:
        """Extract the JSON blob that the pass prepends to the MLIR output."""
        # The pass prints the JSON first, then the MLIR module.
        # Find the boundary: the JSON ends just before "module {" or "module @".
        match = re.search(r"\nmodule\b", output)
        if match:
            json_text = output[: match.start()]
        else:
            # Fallback: try to find the last complete JSON object.
            json_text = output

        json_text = json_text.strip()
        if not json_text:
            return {}

        return json.loads(json_text)

    @staticmethod
    def _parse(raw: Dict[str, Any]) -> Dict[str, FunctionReport]:
        entries: Dict[str, FunctionReport] = {}
        for name, data in raw.items():
            entries[name] = FunctionReport(
                name=name,
                num_qubits=data.get("num_qubits", 0),
                num_alloc_qubits=data.get("num_alloc_qubits", 0),
                num_arg_qubits=data.get("num_arg_qubits", 0),
                operations=data.get("operations", {}),
                measurements=data.get("measurements", {}),
                function_calls=data.get("function_calls", {}),
                var_function_calls=data.get("var_function_calls", {}),
                has_branches=data.get("has_branches", False),
                qnode=data.get("qnode", False),
                device_name=data.get("device_name", ""),
            )
        return entries
