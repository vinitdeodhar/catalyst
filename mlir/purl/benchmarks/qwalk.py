"""
qwalk -- random-walk hitting-time search: a FAT-TAILED, 2q-heavy, non-Clifford
carry benchmark built to exercise the MIGRATE strategy (spec 13).

Widely-cited anchor: quantum-walk search (Aharonov-Ambainis-Kempe-Vazirani 2001;
Ambainis element distinctness 2003; Szegedy 2004; Childs et al.) runs a walk until
it HITS a marked vertex. The number of steps to hit -- a random-walk first-passage
time -- is the canonical HEAVY-TAILED distribution: for a symmetric walk the first
return to the origin has P(T = 2n) ~ n^(-3/2) (Polya recurrence; Feller Vol. 1),
with a fat power-law tail and (untruncated) infinite mean. That fat tail is exactly
what the thin geometric p=5/8 RUS benchmarks lacked: a large fraction of shots run
long enough to CROSS a cut, so migrate's per-window leakage clearing acts on most
shots instead of a ~2% sliver.

Structure (fits Purl's carry shape):
  * Held data `d`: a generic NON-Clifford reference |psi> = Rz(0.7) Ry(0.4)|0>,
    held across the walk, never measured.
  * Body per step: a 2q-gate-HEAVY, NET-IDENTITY coupling to d -- the pump sandwich
    CNOT(a,d), T(a), CNOT(a,d) repeated 3x (n2q = 6). The two CNOTs cancel on d for
    each ancilla branch, so d is left invariant; the middle T is NON-Clifford, so the
    tableau CANNOT prove it -> known_state = unknown -> refresh unsound, migrate/knit
    valid. The 6 CNOTs charge per-2q leakage on d (the reset-clearable error migrate
    removes).
  * Herald: a classical symmetric random walk started at 1; the step succeeds (loop
    exits) when the walk first hits 0. Decoupled from d, so cutting d never changes
    the trip count (unlike ipe_project's posterior coupling). Fat-tailed, truncated
    at max_trips.

Because the per-step action on d is net-identity, the ideal delivered state is the
FIXED input |psi> (trip-count independent) -> clean 3-basis tomography against one
target, no per-shot reference (as rus_data).

Keep in lockstep with mlir/test/Quantum/Purl/qwalk_migrate.mlir.
"""

import math

import numpy as np

from sim.qsim import QSim, RY, RZ

N_WIRES = 2
DATA = 0                 # carried, never measured in the loop
ANCILLA = 1             # walk/touch ancilla, reset + reused each step
N_BLOCKS = 3            # sandwich repeats -> N2Q_PER_ITER = 2 * N_BLOCKS
N2Q_PER_ITER = 2 * N_BLOCKS   # 6 two-qubit gates on d per step
PREP_RY, PREP_RZ = 0.4, 0.7   # non-stabilizer held reference

# effective per-step "success" rate of the first-passage walk (mean is dominated by
# the ~50% that hit 0 on step 1; used only as the pass's `p` profile input).
P_ANALYTIC = 0.5


def _psi():
    return RZ(PREP_RZ) @ RY(PREP_RY) @ np.array([1, 0], dtype=complex)


def ideal_bloch():
    v = _psi()
    rho = np.outer(v, v.conj())
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    return np.array([np.trace(rho @ M).real for M in (X, Y, Z)])


IDEAL_BLOCH = ideal_bloch()
Z_IDEAL = float(IDEAL_BLOCH[2])


def prepare_input(sim):
    """Held reference |psi> = Rz(0.7) Ry(0.4)|0> on the carried data wire."""
    sim.ry(DATA, PREP_RY)
    sim.rz(DATA, PREP_RZ)


def touch(sim):
    """The 2q-heavy NET-IDENTITY non-Clifford coupling on d (n2q = 6). Leaves d
    invariant (the CNOT pair cancels) but charges six 2q gates of leakage on it; the
    non-Clifford T defeats the known-state proof -> unknown."""
    a, d = ANCILLA, DATA
    sim.force_zero(a)
    sim.h(a)
    for _ in range(N_BLOCKS):
        sim.cnot(a, d)
        sim.t(a)             # non-Clifford middle gate (on the ancilla only)
        sim.cnot(a, d)
    sim.measure(a)           # collapse the ancilla (herald readout window on d)


def walk_step(rng, pos):
    """One symmetric random-walk step; returns the new position."""
    return pos + (1 if rng.random() < 0.5 else -1)


def run_unbounded(rng, calib=None, lam=1.0, max_trips=500):
    """Held reference under the walk until first passage to 0. Returns (steps, z)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    pos, k = 1, 0
    while pos != 0 and k < max_trips:
        touch(sim)
        pos = walk_step(rng, pos)
        k += 1
    z = 1 - 2 * sim.measure(DATA)
    return k, z


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    N = 8000
    trips, zs = [], []
    for _ in range(N):
        k, z = run_unbounded(rng, lam=0.0)
        trips.append(k)
        zs.append(z)
    t = np.array(trips)
    mu = float(np.mean(zs))
    print(f"qwalk (lam=0, N={N}): mean_steps={t.mean():.2f} median={np.median(t):.0f} "
          f"max={t.max()} P(T>10)={np.mean(t > 10):.3f} P(T>1)={np.mean(t > 1):.3f}")
    print(f"  <Z> = {mu:+.4f}  (ideal held |psi> <Z> = {Z_IDEAL:+.4f})  "
          f"n2q/iter={N2Q_PER_ITER}")
