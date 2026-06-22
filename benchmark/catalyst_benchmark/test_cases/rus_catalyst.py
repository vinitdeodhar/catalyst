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
"""Repeat-Until-Success circuit — Catalyst implementation.

Circuit (2 qubits):
  target  = wire 0
  ancilla = wire 1

Each attempt applies H(a)-CNOT(t→a)-T(a)-CNOT(t→a)-H(a) then
measures the ancilla.  A |1⟩ outcome means success (loop exits);
|0⟩ means failure and the ancilla is reset via reset=True for the
next attempt.

With target prepared in |+⟩, the success probability per attempt is
(1 − 1/√2)/2 ≈ 14.6 %, giving an expected ~7 iterations.

Note: the T gate must be on the *ancilla* wire, not the target.
If T is placed on the target with CNOT(target→ancilla) the circuit is
a zero-outcome channel — the ancilla always measures |0⟩ and the loop
never terminates.

This is the canonical example of a Repeat-Until-Success (RUS) circuit:
the loop trip count is quantum-measurement-driven and genuinely unknown
at compile time.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import measure, qjit, while_loop

# Expected number of iterations: p(success) = (1 - 1/√2)/2 ≈ 14.6 % → ~7 iters.
EXPECTED_ITERATIONS = 7


@dataclass
class ProblemRUS(Problem):
    """RUS T-gate benchmark problem.  Needs exactly 2 qubits."""

    def __init__(self, dev, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits >= 2, "RUS requires at least 2 wires (target + ancilla)"
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


def qcompile(p: ProblemRUS, _):
    """Compile the RUS T-gate circuit into p.qcircuit."""
    target, ancilla = 0, 1

    def _circuit():
        # Prepare target in |+⟩ so T maps it to a distinguishable state.
        qp.Hadamard(wires=target)

        @while_loop(lambda s: s == 0)
        def rus_attempt(success):
            # ── One attempt ─────────────────────────────────────────────
            qp.Hadamard(wires=ancilla)
            qp.CNOT(wires=[target, ancilla])
            qp.T(wires=ancilla)              # T on ancilla, not target
            qp.CNOT(wires=[target, ancilla])
            qp.Hadamard(wires=ancilla)

            # Measure ancilla: 1 → success (loop exits), 0 → failure.
            # reset=True collapses and resets ancilla to |0⟩ for the next attempt.
            # On failure the target stays in |+⟩ (no explicit correction needed).
            m = measure(ancilla, reset=True)
            return jnp.int64(m)

        # Enter the loop with success=0 to force the first attempt.
        rus_attempt(jnp.int64(0))

        return qp.probs(wires=[target])

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemRUS, _):
    """Execute the compiled RUS circuit."""
    return p.qcircuit()


def run_catalyst(N: int = 2):
    """Stand-alone entry point — compile and run one shot."""
    p = ProblemRUS(qp.device("lightning.qubit", wires=N))

    @qjit
    def _main():
        qcompile(p, None)
        return workflow(p, None)

    result = _main()
    print(f"Target qubit probabilities after T-via-RUS: {result}")
    return result
