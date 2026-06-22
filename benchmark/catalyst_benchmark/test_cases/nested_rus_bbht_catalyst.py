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
"""Nested RUS-in-BBHT — two-level dynamic loop benchmark.

Purpose: create a while→while nesting that tests ResourceAnalysis call-graph
flattening across two dynamic loop levels.  None of the four original
benchmarks (RUS, MSD, BBHT, QPE) have two nested while_loops; this circuit
fills that gap.

Structure:
  Outer while_loop (BBHT-style search):
    Exits when all data qubits measure |1⟩.

  Inner while_loop (RUS-style oracle):
    Each Grover step uses a repeat-until-success sub-circuit to apply a
    phase oracle instead of a deterministic Toffoli.  Exits when the ancilla
    measures |1⟩ (oracle successfully applied).

Circuit (n_data + 1 qubits, default n_data = 2):
  Wires 0 .. n_data-1 : data qubits (search space = 2^n_data)
  Wire  n_data         : ancilla for the RUS oracle

Outer loop body (per search attempt):
  1. H on all data qubits              → equal superposition
  2. Inner RUS oracle (while loop):
       H(anc)
       CNOT(data[i] → anc) for each i
       T(anc)
       CNOT(data[i] → anc) for each i  (uncompute)
       H(anc)
       measure(anc, reset=True) → m; exit if m == 1
  3. Grover diffuser on data qubits:
       H·X on each, CZ (n_data=2) or MCZ, X·H on each
  4. measure(data[i], reset=True) for each i; check all-ones

Expected iterations (analytical approximations):
  Inner RUS: P(oracle success per attempt) ≈ 0.5  → E[inner iters] ≈ 2
  Outer search: for n_data=2 the search space is 4; one Grover step per outer
    iteration amplifies |11⟩.  Expected outer iterations ≈ 2–4 in practice.

Static analysis output:
  _circuit:
    dynamic loops: dyn_while_loop_1 (outer)
  dyn_while_loop_1:
    gates: H(1)×n_data, PauliX(1)×2*n_data, CZ(2)×1  [for n_data=2]
    measurements: MidCircuitMeasure×n_data
    dynamic loops: dyn_while_loop_2 (inner RUS)
  dyn_while_loop_2:
    gates: H(1)×2, CNOT(2)×(2*n_data), T(1)×1
    measurements: MidCircuitMeasure×1

This is the only benchmark in the suite with while→while nesting, testing
getFlattenedResource across two dynamic call-graph levels.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import measure, qjit, while_loop

# Analytical expected iterations (approximate).
EXPECTED_OUTER_ITERATIONS = 3   # O(√(2^n_data)) for n_data=2
EXPECTED_INNER_ITERATIONS = 2   # 1 / P(RUS success) ≈ 1/0.5


@dataclass
class ProblemNestedRUSBBHT(Problem):
    """Nested RUS-in-BBHT benchmark.

    nqubits must be n_data + 1 (ancilla).  Default: n_data=2, nqubits=3.
    """

    def __init__(self, dev, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits >= 3, "Nested RUS-BBHT requires at least 3 wires (2 data + 1 ancilla)"
        self.n_data = self.nqubits - 1   # last wire is ancilla
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


# ── Oracle and diffuser ────────────────────────────────────────────────────

def _diffuser(n_data: int):
    """Grover inversion-about-mean on data qubits 0..n_data-1."""
    for i in range(n_data):
        qp.Hadamard(wires=i)
        qp.PauliX(wires=i)
    if n_data == 2:
        qp.CZ(wires=[0, 1])
    else:
        # MCZ: H · MCX · H on last qubit.
        target = n_data - 1
        qp.Hadamard(wires=target)
        qp.MultiControlledX(wires=list(range(n_data - 1)) + [target])
        qp.Hadamard(wires=target)
    for i in range(n_data):
        qp.PauliX(wires=i)
        qp.Hadamard(wires=i)


# ── Main circuit ──────────────────────────────────────────────────────────

def qcompile(p: ProblemNestedRUSBBHT, _):
    """Compile the nested RUS-in-BBHT circuit into p.qcircuit."""
    n_data = p.n_data
    ancilla = n_data   # ancilla wire index

    def _circuit():
        # ── Outer BBHT-style search loop ──────────────────────────────────
        # Carry: (found: bool, attempt_count: int64)
        @while_loop(lambda found, _cnt: ~found)
        def search_loop(found, attempt_count):

            # 1. Prepare data qubits in equal superposition.
            for i in range(n_data):
                qp.Hadamard(wires=i)

            # 2. Inner RUS oracle: repeat until ancilla measures |1⟩.
            @while_loop(lambda oracle_done: ~oracle_done)
            def rus_oracle(oracle_done):
                # H-CNOT_chain-T-CNOT_chain-H on ancilla.
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

            # 3. Grover diffuser on data qubits.
            _diffuser(n_data)

            # 4. Measure data qubits; check if all-ones solution found.
            bits = jnp.zeros(n_data, dtype=jnp.int64)
            for i in range(n_data):
                m_i = measure(i, reset=True)
                bits = bits.at[i].set(jnp.int64(m_i))

            found_now = jnp.all(bits == 1)
            return found_now, attempt_count + jnp.int64(1)

        found, n_attempts = search_loop(jnp.bool_(False), jnp.int64(0))
        return found, n_attempts

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemNestedRUSBBHT, _):
    """Execute one nested RUS-in-BBHT run."""
    return p.qcircuit()


def run_catalyst(N: int = 3):
    """Stand-alone entry point — compile and run one search."""
    p = ProblemNestedRUSBBHT(qp.device("lightning.qubit", wires=max(N, 3)))

    @qjit
    def _main():
        qcompile(p, None)
        return workflow(p, None)

    found, n_attempts = _main()
    print(f"Nested RUS-BBHT: found={bool(found)}, attempts={int(n_attempts)}")
    return found, n_attempts
