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
"""Iterative Quantum Phase Estimation (Kitaev QPE) — Catalyst implementation.

Target unitary: U = CRZ(π/2) with target in eigenstate |1⟩.
  Phase kicked back per application: e^{i·(π/2)/2} = e^{iπ/4} = e^{i·2π·(1/8)}.
  Eigenphase: φ = 1/8  (= 0.001 in 3-bit binary, 0.0010 in 4-bit binary).

Algorithm — LSB-first semiclassical QPE (n_bits rounds):
  Iterate j = 0, 1, …, n_bits-1  (bit position k = n_bits-1-j, LSB first):
    1. ancilla ← H|0⟩
    2. Apply PhaseShift(correction_accum) — feedback from previously measured bits
    3. Apply controlled-U^{2^k}: repeat CRZ(π/2) exactly 2^k times (inner for_loop)
    4. ancilla ← H
    5. bit ← measure(ancilla, reset=True)
    6. correction_accum ← (correction_accum - π · bit) / 2   [semiclassical update]
    7. estimate ← estimate + bit · 2^j

  estimated_phase = estimate / 2^n_bits   (= 1/8 for T-gate eigenphase, exactly)

Dynamic loop structure — two nested dynamic for_loops visible to the resource analysis:
  • Outer: for_loop(0, n_bits, 1)             — n_bits is a JAX int64 → dyn_for_loop_1
  • Inner: for_loop(0, 2^(n_bits-1-j), 1)    — bound depends on outer var j → dyn_for_loop_2

Inner loop sizes (j=0 to n_bits-1): 2^{n_bits-1}, 2^{n_bits-2}, …, 1  (geometric decay).
Total CRZ applications: 2^{n_bits-1} + … + 1 = 2^n_bits - 1  (exponential in n_bits).

This is the canonical example of nested for_loops where the inner bound is a function of the
outer induction variable.  Static analysis reports 1 CRZ per inner iteration but cannot
evaluate the geometric sum without knowing n_bits at compile time.  Runtime instrumentation
trivially records the true total (e.g., 15 for n_bits=4, 31 for n_bits=5).
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import for_loop, measure, qjit


@dataclass
class ProblemQPE(Problem):
    """Iterative QPE problem.  Wires: ancilla=0, target=1."""

    n_bits: int = 4

    def __init__(self, dev, n_bits: int = 4, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits >= 2, "Iterative QPE needs at least 2 wires (ancilla + target)"
        self.n_bits = n_bits
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


def qcompile(p: ProblemQPE, _):
    """Compile the iterative QPE circuit into p.qcircuit."""
    ancilla, target = 0, 1

    def _circuit(n_bits):
        # Prepare target in eigenstate |1⟩.
        # CRZ(π/2)|+⟩|1⟩ kicks back phase e^{iπ/4} = e^{i·2π·(1/8)} → φ = 1/8.
        qp.PauliX(wires=target)

        # LSB-first outer loop.
        # Carry: (correction_accum: float64, estimate: int64)
        @for_loop(jnp.int64(0), n_bits, jnp.int64(1))
        def qpe_round(j, correction_accum, estimate):
            # k is the actual bit position (n_bits-1 → 0, LSB first).
            k = n_bits - jnp.int64(1) - j

            # ── Ancilla preparation ────────────────────────────────────────
            qp.Hadamard(wires=ancilla)

            # ── Semiclassical correction from already-estimated bits ────────
            # Subtracts the contribution of known lower bits so each measurement
            # is deterministic for eigenphases with exact binary expansions.
            qp.PhaseShift(correction_accum, wires=ancilla)

            # ── Apply controlled-U^{2^k}: 2^k repetitions of CRZ(π/2) ─────
            # inner_iters = 2^k: largest (2^{n_bits-1}) at j=0, decays to 1 at j=n_bits-1.
            inner_iters = jnp.left_shift(jnp.int64(1), k)

            @for_loop(jnp.int64(0), inner_iters, jnp.int64(1))
            def apply_cu(_):
                # Phase kickback: each CRZ(π/2) adds e^{iπ/4} to the ancilla |1⟩ component.
                qp.CRZ(jnp.pi / 2, wires=[ancilla, target])

            apply_cu()

            # ── Readout ────────────────────────────────────────────────────
            qp.Hadamard(wires=ancilla)
            bit = measure(ancilla, reset=True)

            # ── Semiclassical feedback update ──────────────────────────────
            # Derived from: correction_{k-1} = (correction_k - π·bit) / 2
            new_correction = (correction_accum - jnp.pi * jnp.float64(bit)) / jnp.float64(2.0)

            # ── Accumulate bit into estimate ───────────────────────────────
            # Bit measured at j contributes bit·2^j to the MSB-first integer.
            # (j=0 → LSB of estimate, j=n_bits-1 → MSB of estimate)
            new_estimate = estimate + jnp.int64(bit) * jnp.left_shift(jnp.int64(1), j)

            return new_correction, new_estimate

        _, phase_bits = qpe_round(jnp.float64(0.0), jnp.int64(0))

        # estimated_phase = phase_bits / 2^n_bits
        estimated_phase = jnp.float64(phase_bits) / jnp.float64(
            jnp.left_shift(jnp.int64(1), n_bits)
        )
        return estimated_phase, phase_bits

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemQPE, _):
    """Execute the compiled iterative QPE circuit."""
    return p.qcircuit(jnp.int64(p.n_bits))


def run_catalyst(n_bits: int = 4):
    """Stand-alone entry point — compile and run one shot of iterative QPE."""
    p = ProblemQPE(qp.device("lightning.qubit", wires=2), n_bits=n_bits)

    @qjit
    def _main(nb):
        qcompile(p, None)
        return workflow(p, None)

    estimated_phase, phase_bits = _main(jnp.int64(n_bits))
    total_crz = 2**n_bits - 1
    print(f"Iterative QPE ({n_bits} bits)")
    print(f"  Estimated phase : {float(estimated_phase):.6f}  (true: 0.125000)")
    print(f"  Phase bits      : {int(phase_bits):0{n_bits}b} = {int(phase_bits)}")
    print(f"  Total CRZ gates : {total_crz}  (= 2^{n_bits} - 1)")
    return estimated_phase, phase_bits


def expected_total_crz(n_bits: int) -> int:
    """Total CRZ applications: Σ_{k=0}^{n_bits-1} 2^k = 2^n_bits - 1."""
    return 2**n_bits - 1


def expected_outer_iters(n_bits: int) -> int:
    """Outer for_loop trip count (equals n_bits exactly)."""
    return n_bits
