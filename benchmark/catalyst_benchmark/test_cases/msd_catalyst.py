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
"""Magic State Distillation (MSD) — Catalyst benchmark.

Protocol: simplified n_magic-to-1 MSD with parity-syndrome check.

Each attempt:
  1. Prepare n_magic ancilla qubits in |T⟩ = T|+⟩ (the ideal magic state).
  2. Inject a bit-flip error (X gate) on each ancilla independently with
     probability ``p`` (physical error rate).  Modelled via JAX PRNG:
       key → bernoulli(p) → catalyst.cond → PauliX or nothing.
  3. Parity syndrome: CNOT(magic_i → syndrome) for each i; measure syndrome.
     syndrome == 0 iff an even number of magic states have errors.
  4. If syndrome == 0 → attempt accepted (success); loop exits.
     If syndrome == 1 → error detected; reset all ancillae and retry.

Success probability per attempt:
  P(success) = P(even errors) = Σ_{k even} C(n,k) p^k (1-p)^{n-k}
             = ½ [(1-2p)^n + 1]

Dynamic loop structure:
  • Outer: while_loop(~success) — trip count geometric with p(success) above.
    Unlike RUS (where p is circuit-determined), here p is a *parameter*:
    the same circuit at p=0.01 vs p=0.1 produces different distributions.

  • No inner dynamic loops — the prep/syndrome for each magic wire is
    unrolled statically (n_magic is a Python constant visible at trace time).

Key resource-estimation challenge:
  - Static analysis counts n_magic T gates + n_magic H gates + n_magic CNOTs
    per while-loop body, PLUS n_magic PauliX (worst-case: all cond branches
    taken).  Runtime shows the mean X count is p·n_magic — static overestimates
    by 1/p for the X gate.
  - Static cannot predict trip count without p; runtime records the actual
    distribution which shifts with p.

Comparison table (n_magic=7):
  p      P(success)  E[iters]  E[T gates]  E[CNOT gates]
  0.01   0.932       1.07      7.5         7.5
  0.05   0.708       1.41      9.9         9.9
  0.10   0.536       1.87      13.1        13.1
  0.20   0.327       3.06      21.4        21.4
  0.30   0.224       4.46      31.2        31.2
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pennylane as qp
from catalyst_benchmark.types import Problem

from catalyst import cond, measure, qjit, while_loop

# Default protocol parameters.
DEFAULT_N_MAGIC = 7     # number of noisy magic state ancillae
DEFAULT_P_ERR   = 0.10  # physical bit-flip error rate per magic state


@dataclass
class ProblemMSD(Problem):
    """MSD problem.

    Wires:
      0            — syndrome qubit (parity check accumulator)
      1 .. n_magic — noisy magic state ancillae
    Total: n_magic + 1 qubits.
    """

    n_magic: int   = DEFAULT_N_MAGIC
    p_err:   float = DEFAULT_P_ERR

    def __init__(self, dev, n_magic: int = DEFAULT_N_MAGIC,
                 p_err: float = DEFAULT_P_ERR, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        assert self.nqubits == n_magic + 1, (
            f"MSD needs exactly n_magic+1={n_magic+1} wires, got {self.nqubits}"
        )
        self.n_magic = n_magic
        self.p_err   = p_err
        self.qcircuit = None

    def trial_params(self, _: int) -> Any:
        return jnp.array([], dtype=jnp.float64)


# ── Helpers ────────────────────────────────────────────────────────────────

def _prepare_noisy_T(wire: int, p_err: float, key):
    """Prepare one noisy T state on `wire`.

    Applies H·T to get |T⟩, then injects a Pauli-X error with probability p_err.
    Uses Catalyst cond so the conditional gate is traced correctly through @qjit.
    """
    qp.Hadamard(wires=wire)
    qp.T(wires=wire)
    key, subkey = jax.random.split(key)
    error = jax.random.bernoulli(subkey, jnp.float64(p_err))

    @cond(error)
    def inject():
        qp.PauliX(wires=wire)

    inject()   # no otherwise → else branch is a no-op
    return key


def _reset_magic(n_magic: int):
    """Mid-circuit reset of all magic ancillae for the next attempt."""
    for i in range(1, n_magic + 1):
        measure(i, reset=True)


# ── Circuit ────────────────────────────────────────────────────────────────

def qcompile(p: ProblemMSD, _):
    """Compile the MSD circuit into p.qcircuit."""
    n_magic = p.n_magic
    p_err   = p.p_err
    syndrome_wire = 0

    def _circuit(key):
        # while_loop carry: (success: bool, key: PRNGKey)
        @while_loop(lambda success, _k: ~success)
        def msd_attempt(success, key):
            # ── 1. Prepare n_magic noisy T states ──────────────────
            for wire in range(1, n_magic + 1):
                key = _prepare_noisy_T(wire, p_err, key)

            # ── 2. Parity syndrome via CNOT chain ───────────────────
            for wire in range(1, n_magic + 1):
                qp.CNOT(wires=[wire, syndrome_wire])

            # ── 3. Measure syndrome ─────────────────────────────────
            syn = measure(syndrome_wire, reset=True)

            # ── 4. Reset magic ancillae for next attempt ─────────────
            _reset_magic(n_magic)

            return jnp.bool_(syn == 0), key

        success, _ = msd_attempt(jnp.bool_(False), key)
        return success

    p.qcircuit = qp.QNode(_circuit, p.dev, **p.qnode_kwargs)


def workflow(p: ProblemMSD, _):
    """Execute the compiled MSD circuit."""
    key = jax.random.PRNGKey(0)
    return p.qcircuit(key)


# ── Analytics ──────────────────────────────────────────────────────────────

def success_prob(n_magic: int, p_err: float) -> float:
    """P(syndrome == 0) = P(even-weight errors) = ½[(1-2p)^n + 1]."""
    return 0.5 * ((1 - 2 * p_err) ** n_magic + 1)


def expected_iters(n_magic: int, p_err: float) -> float:
    """Expected number of while-loop iterations = 1 / P(success)."""
    return 1.0 / success_prob(n_magic, p_err)


def expected_gate_count(gate: str, n_magic: int, p_err: float) -> float:
    """Expected total gate count over the geometric-distributed trip count.

    Gates per attempt:
      Hadamard : n_magic   (one per magic wire prep)
      T        : n_magic
      PauliX   : p_err * n_magic  (on average, from the cond blocks)
      CNOT     : n_magic   (parity syndrome)
    """
    iters = expected_iters(n_magic, p_err)
    per_attempt = {
        "Hadamard(1)": n_magic,
        "T(1)":        n_magic,
        "PauliX(1)":   p_err * n_magic,    # average, runtime measurable
        "CNOT(2)":     n_magic,
        "Measure":     n_magic + 1,        # n_magic resets + 1 syndrome
    }
    if gate not in per_attempt:
        return 0.0
    return per_attempt[gate] * iters


# ── Stand-alone entry point ────────────────────────────────────────────────

def run_catalyst(n_magic: int = DEFAULT_N_MAGIC, p_err: float = DEFAULT_P_ERR):
    """Compile and run one MSD shot."""
    p_succ = success_prob(n_magic, p_err)
    e_iters = expected_iters(n_magic, p_err)
    print(f"MSD  (n_magic={n_magic}, p_err={p_err})")
    print(f"  P(success/attempt) = {p_succ:.4f}   E[iters] = {e_iters:.2f}")

    prob = ProblemMSD(
        qp.device("lightning.qubit", wires=n_magic + 1),
        n_magic=n_magic, p_err=p_err,
    )

    @qjit
    def _main(key):
        qcompile(prob, None)
        return workflow(prob, None)

    key = jax.random.PRNGKey(42)
    result = _main(key)
    print(f"  Circuit output (success flag): {bool(result)}")
    return result
