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
"""Boyer-Brassard-Høyer-Tapp (BBHT) Grover search — Catalyst implementation.

The BBHT algorithm handles the case where the number of solutions is
unknown.  Instead of a fixed-iteration Grover run it uses:

    m = 1,  λ = 6/5
    repeat:
        k ~ Uniform[1, ⌈m⌉]
        run k Grover iterations
        measure; if solution found → return
        m = min(λ·m, √N)

Two levels of non-static resource usage:
  • outer while_loop  — trip count depends on quantum measurements
  • inner for_loop    — upper bound k is randomly chosen per outer iteration

This is the canonical case where static resource analysis can only
report per-iteration costs; only runtime instrumentation can report
actual total gate counts for a given run.

Oracle: marks |11…1⟩ (all-ones bitstring).
  N=3: Toffoli-based (exact).
  N>3: uses qp.MultiControlledX with a single ancilla wire.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import for_loop, measure, qjit, while_loop

_LAMBDA = 6.0 / 5.0  # BBHT growth factor


@dataclass
class ProblemBBHT(Problem):
    """BBHT problem. nqubits includes one ancilla for the MCX gate when N>3."""

    def __init__(self, dev, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits >= 3, "BBHT needs at least 3 wires"
        # data wires: 0 .. nqubits-1
        # For N>3 the last wire doubles as MCX ancilla during the oracle.
        self.n_data = self.nqubits
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


# ── Oracle and diffuser ────────────────────────────────────────────────────

def _oracle(n_data: int):
    """Phase-flip the |11…1⟩ state.

    Uses a Toffoli for N=3, MultiControlledX otherwise.
    The ancilla for MCX is always reset after use.
    """
    if n_data == 3:
        # Toffoli implements controlled-controlled-X → MCZ via H sandwich.
        qp.Hadamard(wires=n_data - 1)
        qp.Toffoli(wires=[0, 1, 2])
        qp.Hadamard(wires=n_data - 1)
    else:
        # MCZ = H · MCX · H on the target qubit.
        target = n_data - 1
        controls = list(range(n_data - 1))
        qp.Hadamard(wires=target)
        qp.MultiControlledX(wires=controls + [target])
        qp.Hadamard(wires=target)


def _diffuser(n_data: int):
    """Grover diffusion operator (inversion about the mean)."""
    for i in range(n_data):
        qp.Hadamard(wires=i)
        qp.PauliX(wires=i)

    # MCZ on all data qubits.
    if n_data == 3:
        qp.Hadamard(wires=n_data - 1)
        qp.Toffoli(wires=[0, 1, 2])
        qp.Hadamard(wires=n_data - 1)
    else:
        target = n_data - 1
        controls = list(range(n_data - 1))
        qp.Hadamard(wires=target)
        qp.MultiControlledX(wires=controls + [target])
        qp.Hadamard(wires=target)

    for i in range(n_data):
        qp.PauliX(wires=i)
        qp.Hadamard(wires=i)


# ── Main circuit ──────────────────────────────────────────────────────────

def qcompile(p: ProblemBBHT, _):
    """Compile the BBHT circuit into p.qcircuit."""
    n_data = p.n_data
    n_space = jnp.int64(2 ** n_data)

    def _circuit(key):
        # ── BBHT outer loop ───────────────────────────────────────────
        # Carry: (found: bool, m: float64, key: PRNGKey)
        # Condition: keep going while NOT found.
        @while_loop(lambda found, _m, _k: ~found)
        def bbht_loop(found, m, rng_key):
            # Sample k ∈ [1, ⌈m⌉] uniformly at random.
            rng_key, subkey = jax.random.split(rng_key)
            k = jax.random.randint(
                subkey, shape=(), minval=jnp.int64(1), maxval=jnp.int64(m) + 1
            )

            # ── Prepare equal superposition ───────────────────────────
            for i in range(n_data):
                qp.Hadamard(wires=i)

            # ── k Grover iterations (dynamic upper bound) ─────────────
            @for_loop(0, k, 1)
            def grover_step(_):
                _oracle(n_data)
                _diffuser(n_data)

            grover_step()

            # ── Measure all data qubits; reset for next attempt ───────
            # Python-level unroll: n_data is a compile-time constant.
            bits = jnp.zeros(n_data, dtype=jnp.int64)
            for i in range(n_data):
                m_i = measure(i, reset=True)
                bits = bits.at[i].set(jnp.int64(m_i))

            # Check: did we find the all-ones solution?
            found_now = jnp.all(bits == 1)

            # Update m: grow by λ, cap at √N.
            new_m = jnp.minimum(
                jnp.float64(_LAMBDA) * m,
                jnp.sqrt(jnp.float64(n_space)),
            )

            return found_now, new_m, rng_key

        found, _, _ = bbht_loop(jnp.bool_(False), jnp.float64(1.0), key)
        return found

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemBBHT, _):
    """Execute one BBHT search run."""
    key = jax.random.PRNGKey(0)
    return p.qcircuit(key)


def run_catalyst(N: int = 3):
    """Stand-alone entry point — compile and run one BBHT search."""
    p = ProblemBBHT(qp.device("lightning.qubit", wires=N))

    @qjit
    def _main(key):
        qcompile(p, None)
        return workflow(p, None)

    key = jax.random.PRNGKey(42)
    result = _main(key)
    print(f"BBHT found all-ones solution: {result}")
    return result
