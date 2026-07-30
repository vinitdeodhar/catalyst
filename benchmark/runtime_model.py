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
"""Hardware-grounded runtime (temporal-cost) model for instrumented circuits.

The gate-counter instrumentation records, for a single circuit execution, the
exact number of times each ``(gate_name, wire_count)`` op runs — including the
mid-circuit measurements that drive dynamic loops.  This module turns those
per-run counts into a *modeled device runtime* in nanoseconds, using published
per-operation durations and classical-feedforward latencies for real
superconducting hardware.

Because the counts come from a real execution, the returned runtime carries the
true (stochastic) trip count of every dynamic loop for that shot.  Running an
instrumented circuit N times therefore yields a runtime *distribution*, not a
single mean — which is what the depth-bounding and loop-peeling optimizations
act on.

Model
-----
Each logical gate is expanded to its native cost ``(n_1q, n_2q)`` on the target
device (a controlled-Z entangler plus single-qubit pulses); Z-family rotations
(Z, S, T, RZ, PhaseShift) are virtual frame updates and cost 0 ns.  A
mid-circuit measurement costs ``d_meas`` plus one classical-feedforward round
``tau_fb`` (the loop-continuation / reset decision reads the measurement
result, and cannot proceed until it is classified and fed back).

    runtime_ns = sum_g count_g * duration_g            (gates)
               + n_measure * (d_meas + tau_fb)         (measure + feedback)

This is a *serial* accumulation: it does not model cross-qubit gate
parallelism, so the gate term is an upper bound on the true critical path.  For
the measurement-driven loops targeted here the runtime is dominated by
``d_meas + tau_fb`` per iteration (see the breakdown), so the serial gate term
is a small and near-tight component; a critical-path refinement would require
per-op ordering and qubit operands, i.e. an IR-level timing pass rather than
aggregate counts.  The breakdown is returned so the three components are always
visible.

Published parameters (see ``docs`` strings on each preset for citations):

  IQM Garnet : PRX 20 ns, CZ 40 ns, readout 280 ns, feedforward ~600 ns
  IBM Heron  : 1q 32 ns, CZ 68 ns, mid-circuit readout 1288 ns, feedforward ~600 ns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

GateCountDict = Dict[str, int]

# Logical gate -> native decomposition cost (n_single_qubit_pulses, n_two_qubit_gates).
# Z-family gates are virtual frame updates: (0, 0).  Values are device-independent
# op *counts*; the device model multiplies them by its d_1q / d_2q durations.
_DEFAULT_NATIVE_COST: Dict[str, Tuple[int, int]] = {
    # ── virtual Z-family (0 ns) ────────────────────────────────────────────
    "PauliZ": (0, 0), "S": (0, 0), "Sdag": (0, 0), "Adjoint(S)": (0, 0),
    "T": (0, 0), "Tdag": (0, 0), "Adjoint(T)": (0, 0),
    "RZ": (0, 0), "PhaseShift": (0, 0), "Identity": (0, 0),
    # ── single-qubit pulses ────────────────────────────────────────────────
    "Hadamard": (1, 0), "PauliX": (1, 0), "PauliY": (1, 0),
    "RX": (1, 0), "RY": (1, 0), "SX": (1, 0), "PRX": (1, 0), "Rot": (1, 0),
    # ── two-qubit ──────────────────────────────────────────────────────────
    "CZ": (0, 1), "CNOT": (2, 1), "CX": (2, 1), "CY": (2, 1),
    "CRZ": (2, 2), "CRX": (2, 2), "CRY": (2, 2),
    "ControlledPhaseShift": (2, 2), "CPhase": (2, 2), "IsingZZ": (0, 2),
    "SWAP": (0, 3),
    # ── three-qubit (Clifford+T / Shende decomposition) ────────────────────
    "Toffoli": (8, 6), "CCX": (8, 6), "CCZ": (6, 6), "CSWAP": (8, 8),
}

# Labels treated as mid-circuit measurements (each incurs d_meas + tau_fb).
_MEASURE_PREFIXES = ("Measure", "MidCircuitMeasure", "MCM")


@dataclass(frozen=True)
class DeviceTimingModel:
    """Per-operation durations (ns) and feedforward latency for one device."""

    name: str
    d_1q: float          # single-qubit native pulse (ns)
    d_2q: float          # two-qubit native entangler, e.g. CZ (ns)
    d_meas: float        # mid-circuit measurement (ns)
    tau_fb: float        # classical feedforward latency, measure->cond gate (ns)
    doc: str = ""
    native_cost: Mapping[str, Tuple[int, int]] = field(
        default_factory=lambda: dict(_DEFAULT_NATIVE_COST)
    )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _base_name(label: str) -> str:
        """'CNOT_2' -> 'CNOT'; leaves already-bare names untouched."""
        head, _, tail = label.rpartition("_")
        return head if (head and tail.isdigit()) else label

    @staticmethod
    def _wire_count(label: str) -> int:
        _, _, tail = label.rpartition("_")
        return int(tail) if tail.isdigit() else 1

    def is_measure(self, label: str) -> bool:
        return self._base_name(label).startswith(_MEASURE_PREFIXES)

    def gate_duration(self, label: str) -> float:
        """Duration (ns) of one logical gate given by its counter label."""
        base = self._base_name(label)

        # Multi-controlled X/Y/Z that survives to the counter as a single op
        # (e.g. `quantum.custom "PauliX" ctrls(...)`) is not a 1-qubit gate: on a
        # device with no spare qubits it must be realized ancilla-free, at
        # O(N^2) cost.  N = (wire_count - 1) controls.  Closed forms fitted to
        # PennyLane's ancilla-free decomposition (exact for 2q; real-1q excludes
        # virtual Z-family): 2q = 24N^2-116N+156, real-1q ≈ 10.14N^2-48.7N+67.9.
        if base in ("PauliX", "PauliY", "PauliZ"):
            w = self._wire_count(label)
            N = w - 1  # number of controls
            if N >= 3:
                n2 = 24 * N * N - 116 * N + 156
                n1 = round(10.143 * N * N - 48.714 * N + 67.857)
                return n1 * self.d_1q + n2 * self.d_2q

        cost = self.native_cost.get(base)
        if cost is None:
            # Fallback for un-tabulated ops: infer from wire count.
            w = self._wire_count(label)
            if w <= 1:
                cost = (1, 0)
            elif w == 2:
                cost = (2, 1)
            else:  # crude multi-qubit fallback ~ (w-1) Toffoli-equivalents
                cost = (8 * (w - 2), 6 * (w - 2))
        n1, n2 = cost
        return n1 * self.d_1q + n2 * self.d_2q

    # ── main entry point ─────────────────────────────────────────────────────

    def runtime_ns(self, gate_counts: GateCountDict) -> "RuntimeEstimate":
        """Modeled device runtime (ns) for one execution's gate counts."""
        gate_ns = 0.0
        n_measure = 0
        for label, count in gate_counts.items():
            if count <= 0:
                continue
            if self.is_measure(label):
                n_measure += count
            else:
                gate_ns += count * self.gate_duration(label)
        measure_ns = n_measure * self.d_meas
        feedback_ns = n_measure * self.tau_fb
        return RuntimeEstimate(
            device=self.name,
            total_ns=gate_ns + measure_ns + feedback_ns,
            gate_ns=gate_ns,
            measure_ns=measure_ns,
            feedback_ns=feedback_ns,
            n_measure=n_measure,
        )


@dataclass(frozen=True)
class RuntimeEstimate:
    """Modeled runtime for one execution, with its three components."""

    device: str
    total_ns: float
    gate_ns: float
    measure_ns: float
    feedback_ns: float
    n_measure: int

    def __float__(self) -> float:
        return self.total_ns

    def pretty(self) -> str:
        t = self.total_ns
        unit = f"{t:.0f} ns" if t < 1e3 else (
            f"{t/1e3:.2f} us" if t < 1e6 else f"{t/1e6:.3f} ms")
        return (f"{unit} on {self.device}  "
                f"(gates {self.gate_ns:.0f} ns + measure {self.measure_ns:.0f} ns "
                f"+ feedback {self.feedback_ns:.0f} ns, {self.n_measure} MCM)")


# ---------------------------------------------------------------------------
# Device presets (real published numbers)
# ---------------------------------------------------------------------------

IQM_GARNET = DeviceTimingModel(
    name="IQM Garnet",
    d_1q=20.0, d_2q=40.0, d_meas=280.0, tau_fb=600.0,
    doc=("Superconducting 20-qubit QPU (Braket dynamic-circuit target). "
         "PRX 20 ns, CZ 40 ns [arXiv:2508.16437]; readout 280 ns; "
         "feedforward ~600 ns (IBM-class, no IQM-specific figure published)."),
)

IBM_HERON = DeviceTimingModel(
    name="IBM Heron",
    d_1q=32.0, d_2q=68.0, d_meas=1288.0, tau_fb=600.0,
    doc=("Superconducting 156-qubit QPU. 1q 32 ns, CZ ~68-88 ns, fast "
         "mid-circuit (M2) readout 1288 ns [arXiv:2402.17833]; feedforward "
         "~600 ns [IBM dynamic-circuits docs]."),
)

DEVICES = {"garnet": IQM_GARNET, "heron": IBM_HERON}
DEFAULT_DEVICE = IQM_GARNET


# ---------------------------------------------------------------------------
# CLI / self-test: reconcile with the worked RUS example in the model doc.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:  # post-process a JSON dump of {label: count}
        counts = json.loads(open(sys.argv[1]).read())
        for dev in DEVICES.values():
            print(dev.runtime_ns(counts).pretty())
        sys.exit(0)

    # One RUS iteration (Fig. 1 body): H,CNOT,T,CNOT,H,measure(reset).
    rus_iter = {"Hadamard_1": 2, "CNOT_2": 2, "T_1": 1, "PauliX_1": 1, "Measure_1": 1}
    print("One RUS iteration:")
    for dev in DEVICES.values():
        est = dev.runtime_ns(rus_iter)
        print("  " + est.pretty())
    # RUS full run at k=7 (E[k]=6.83): scale loop-body counts by 7, +1 outer H.
    rus_run = {"Hadamard_1": 2 * 7 + 1, "CNOT_2": 2 * 7, "T_1": 7,
               "PauliX_1": 7, "Measure_1": 7}
    print("\nRUS full run, k=7:")
    for dev in DEVICES.values():
        print("  " + dev.runtime_ns(rus_run).pretty())
