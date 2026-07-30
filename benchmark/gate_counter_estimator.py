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
"""Gate counter estimator — runtime instrumentation for Catalyst circuits.

Strategy
--------
The static ``resource-analysis`` pass can report per-iteration gate costs for
dynamic loops, but trip counts for quantum-measurement-driven loops (e.g. RUS,
BBHT) are only known at execution time.

This module:

1. Recompiles a ``@qjit``-decorated function with the
   ``gate-counter-instrumentation`` MLIR pass injected into the
   ``QuantumCompilationStage`` pipeline.  The pass adds one
   ``memref.global`` counter (type ``[1 x i64]``) per unique
   (gate_name, wire_count) pair.  Each gate execution increments the
   corresponding counter.

2. Provides ``GateCounterSession``, a context manager that:
   - Compiles the instrumented circuit on first call.
   - Resets all counters before each run.
   - Reads counter values back via ctypes after execution.
   - Returns a ``GateCountDict`` mapping ``"GateName_wires"`` → count.

Usage::

    import pennylane as qp
    import jax.numpy as jnp
    from catalyst import qjit, while_loop, measure
    from gate_counter_estimator import GateCounterSession

    dev = qp.device("lightning.qubit", wires=2)

    def _circuit():
        @while_loop(lambda s: s == 0)
        def loop(s):
            qp.Hadamard(wires=1)
            m = measure(1, reset=True)
            return jnp.int64(m)
        loop(jnp.int64(0))
        return qp.probs(wires=[0])

    with GateCounterSession(_circuit, dev) as sess:
        for trial in range(10):
            result, counts = sess.run()
            print(f"  trial {trial}: {dict(counts)}")
"""

from __future__ import annotations

import ctypes
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pennylane as qp

from catalyst import qjit
from catalyst.pipelines import default_pipeline, insert_pass_after

from runtime_model import DEFAULT_DEVICE, DeviceTimingModel, RuntimeEstimate


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

GateCountDict = Dict[str, int]  # gate_label → total executions in one run


@dataclass
class RunResult:
    """Outcome of one instrumented circuit execution."""
    circuit_output: Any        # whatever the @qjit function returns
    gate_counts: GateCountDict # gate_label → count for this run
    runtime_ns: float = 0.0    # modeled device runtime for this run (ns)
    runtime: Optional[RuntimeEstimate] = None  # runtime with per-component breakdown


# ---------------------------------------------------------------------------
# GateCounterSession
# ---------------------------------------------------------------------------

class GateCounterSession:
    """Compile a quantum circuit once with gate counters, then run it N times.

    Each call to :meth:`run` resets the counters, executes the compiled
    circuit, and returns the actual per-gate execution counts alongside the
    circuit output.

    Parameters
    ----------
    circuit_fn :
        The *undecorated* circuit body (a Python function whose body
        contains PennyLane + Catalyst ops).  This is wrapped in a
        ``@qjit`` automatically.
    dev :
        PennyLane device (must be lightning.qubit for Catalyst).
    args :
        Fixed positional arguments forwarded to the circuit on every call.
    qnode_kwargs :
        Extra kwargs forwarded to ``qp.QNode``.  The reserved keyword
        ``timing_model`` (a :class:`runtime_model.DeviceTimingModel`) is
        consumed here rather than forwarded, and selects the device used to
        model per-run runtime (default: IQM Garnet).
    """

    def __init__(
        self,
        circuit_fn: Callable,
        dev: qp.Device,
        *args,
        **qnode_kwargs,
    ):
        self._circuit_fn = circuit_fn
        self._dev = dev
        self._fixed_args = args
        self._timing_model: DeviceTimingModel = qnode_kwargs.pop(
            "timing_model", DEFAULT_DEVICE
        )
        # Optional MLIR passes inserted after adjoint-lowering and *before* the
        # gate-counter pass, so their rewritten gates are the ones counted
        # (e.g. "width-guarded-mcx-decomp{qubit-budget=20}").
        self._pre_passes = list(qnode_kwargs.pop("pre_instrumentation_passes", []))
        self._qnode_kwargs = qnode_kwargs

        self._manifest_path: Optional[str] = None
        self._manifest: Dict[str, str] = {}   # label → symbol_name
        self._compiled_fn = None               # the compiled @qjit function
        self._lib = None                       # ctypes handle to the .so

    # ── context manager protocol ───────────────────────────────────────────

    def __enter__(self):
        self._compile()
        return self

    def __exit__(self, *_):
        pass

    # ── public API ─────────────────────────────────────────────────────────

    def run(self, *args) -> RunResult:
        """Reset counters, execute circuit, return output + gate counts."""
        if self._compiled_fn is None:
            self._compile()
        call_args = args or self._fixed_args
        self._reset_counters()
        result = self._compiled_fn(*call_args)
        counts = self._read_counters()
        est = self._timing_model.runtime_ns(counts)
        return RunResult(
            circuit_output=result,
            gate_counts=counts,
            runtime_ns=est.total_ns,
            runtime=est,
        )

    # ── internals ──────────────────────────────────────────────────────────

    def _compile(self):
        """Build the instrumented @qjit function and extract the .so handle."""
        # Write manifest to a temp file so the pass can record gate names.
        fd, manifest_path = tempfile.mkstemp(suffix=".json", prefix="gate_counter_manifest_")
        os.close(fd)
        self._manifest_path = manifest_path

        # Build the instrumented pipeline: inject gate-counter-instrumentation
        # after adjoint-lowering (the last substantive pass in
        # QuantumCompilationStage before symbol-dce).
        pipeline = default_pipeline()
        pass_name = (
            f"gate-counter-instrumentation{{manifest-file={manifest_path}}}"
        )
        for _name, passes in pipeline:
            if "adjoint-lowering" in passes:
                # Insert any pre-instrumentation passes first, then the counter
                # pass after the last of them, preserving requested order.
                anchor = "adjoint-lowering"
                for p in self._pre_passes:
                    insert_pass_after(passes, p, anchor)
                    anchor = p
                insert_pass_after(passes, pass_name, anchor)
                break

        circuit_fn = self._circuit_fn
        dev = self._dev
        qnode_kwargs = self._qnode_kwargs
        fixed_args = self._fixed_args

        if fixed_args:
            # Circuit takes arguments — compile with matching arg shapes.
            @qjit(pipelines=pipeline, keep_intermediate=False)
            def _instrumented(*args):
                return qp.QNode(circuit_fn, dev, **qnode_kwargs)(*args)

            _instrumented(*fixed_args)
        else:
            @qjit(pipelines=pipeline, keep_intermediate=False)
            def _instrumented():
                return qp.QNode(circuit_fn, dev, **qnode_kwargs)()

            _instrumented()

        self._compiled_fn = _instrumented

        # Load the manifest produced by the pass.
        if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
            with open(manifest_path) as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {}

        # Get a ctypes handle to the compiled shared library.
        so_file = _instrumented.compiled_function.shared_object.shared_object_file
        self._lib = ctypes.CDLL(so_file)

    def _reset_counters(self):
        """Zero all counter globals in the compiled library."""
        if self._lib is None:
            return
        for sym in self._manifest.values():
            try:
                arr = (ctypes.c_int64 * 1).in_dll(self._lib, sym)
                arr[0] = 0
            except AttributeError:
                pass

    def _read_counters(self) -> GateCountDict:
        """Read all counter globals from the compiled library."""
        counts: GateCountDict = {}
        if self._lib is None:
            return counts
        for label, sym in self._manifest.items():
            try:
                arr = (ctypes.c_int64 * 1).in_dll(self._lib, sym)
                counts[label] = int(arr[0])
            except AttributeError:
                counts[label] = -1  # symbol not found
        return counts
