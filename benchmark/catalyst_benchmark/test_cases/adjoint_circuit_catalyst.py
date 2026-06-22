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
"""Adjoint gate circuit — Catalyst benchmark.

A static circuit (no dynamic loops) that applies adjoint (dagger) gates.
Purpose: exercises the isAdjoint flag tracking in ResourceAnalysis,
verifying that Adjoint(T) appears correctly in the analysis output and
matches the gate counter runtime ground truth.

Circuit (1 qubit):
  H · T · T† · H |0⟩ = H · (T†T) · H |0⟩ = H · I · H |0⟩ = |0⟩

Static analysis should report:
  Hadamard(1) × 2
  T(1)        × 1
  Adjoint(T)  × 1   ← the path exercised by getGateOpName with isAdjoint=True

Gate counter should match exactly (static circuit, no branches, no loops).
Expected result: probs = [1.0, 0.0].
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import qjit


@dataclass
class ProblemAdjoint(Problem):
    """Adjoint gate benchmark.  Uses 1 qubit."""

    def __init__(self, dev, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits >= 1, "Adjoint circuit requires at least 1 wire"
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


def qcompile(p: ProblemAdjoint, _):
    """Compile the adjoint circuit into p.qcircuit."""

    def _circuit():
        # H · T · T† · H — net effect is identity on |0⟩.
        # qp.adjoint(qp.T) compiles to a quantum.custom "T" adjoint=true op,
        # which ResourceAnalysis names "Adjoint(T)".
        qp.Hadamard(wires=0)
        qp.T(wires=0)
        qp.adjoint(qp.T)(wires=0)
        qp.Hadamard(wires=0)
        return qp.probs(wires=[0])

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemAdjoint, _):
    """Execute the adjoint circuit."""
    return p.qcircuit()


def run_catalyst(N: int = 1):
    """Stand-alone entry point."""
    p = ProblemAdjoint(qp.device("lightning.qubit", wires=max(N, 1)))

    @qjit
    def _main():
        qcompile(p, None)
        return workflow(p, None)

    result = _main()
    print(f"Adjoint circuit probs (expect [1.0, 0.0]): {result}")
    return result
