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
"""Coin-flip accumulator — Catalyst benchmark.

A while-loop that measures a qubit prepared in |+⟩ (50/50 superposition)
and exits on outcome |1⟩.  This is the simplest possible measurement-driven
loop: Geometric(p = 0.5) trip count, expected iterations = 2.

Circuit (1 qubit):
  Loop:
    H           — reset to |+⟩
    measure(0, reset=True) → m
    exit if m == 1
  Return flip count.

Per-iteration cost (static analysis):
  Hadamard(1)      × 1
  MidCircuitMeasure × 1

Expected iterations: 2   (E[k] = 1/p = 1/0.5 = 2)

Primary use in the paper:
  1. Sanity check: the simplest dynamic circuit with a known exact E[iters].
  2. Cleanest demonstration of the branch over-approximation gap: any scf.if
     inside the body would be over-counted by the max model by a factor of
     exactly 2 (p = 0.5), giving a clean "2×" number to cite.
  3. Validation (E1): static per-iteration count must exactly equal
     (gate_counter_total / observed_trip_count) across all executions.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import measure, qjit, while_loop

# E[k] = 1/p = 2; this is exact for a Geometric(0.5) distribution.
EXPECTED_ITERATIONS = 2


@dataclass
class ProblemCoinFlip(Problem):
    """Coin-flip benchmark.  Uses 1 qubit."""

    def __init__(self, dev, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits >= 1, "Coin-flip circuit requires at least 1 wire"
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


def qcompile(p: ProblemCoinFlip, _):
    """Compile the coin-flip circuit into p.qcircuit."""

    def _circuit():
        # Carry: (count: int64, last_result: int64)
        # Condition: keep flipping while last result was tails (0).
        @while_loop(lambda count, result: result == 0)
        def flip_loop(count, result):
            qp.Hadamard(wires=0)           # prepare |+⟩
            m = measure(0, reset=True)     # measure; reset for next attempt
            return count + jnp.int64(1), jnp.int64(m)

        # Start with result=0 to force the first flip.
        count, _ = flip_loop(jnp.int64(0), jnp.int64(0))
        return count

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemCoinFlip, _):
    """Execute the coin-flip circuit and return the number of flips."""
    return p.qcircuit()


def run_catalyst(N: int = 1):
    """Stand-alone entry point."""
    p = ProblemCoinFlip(qp.device("lightning.qubit", wires=max(N, 1)))

    @qjit
    def _main():
        qcompile(p, None)
        return workflow(p, None)

    result = _main()
    print(f"Coin-flip count (expected ~2): {int(result)}")
    return result
