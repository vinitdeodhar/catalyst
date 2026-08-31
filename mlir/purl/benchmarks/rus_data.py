"""
rus_data -- RUS gate synthesis applied to program DATA (spec 6.5).

Paetznick-Svore repeat-until-success synthesis of the non-Clifford axial rotation
V3 = (I + 2iZ)/sqrt(5), applied to an ARBITRARY program state mid-computation. The
carried qubit is the algorithm's own data (prepared through non-Clifford gates), so
the known-state analysis returns `unknown` and refresh is unsound within Purl's
framework -- a knit-only benchmark whose unknown-ness is the cited protocol's own
premise (not a constructed obstruction).

  *** FLAGGED BEST-EFFORT RECONSTRUCTION (spec 6.5 option 2) ***
  The spec mandates TRANSCRIBING the exact Paetznick-Svore V3 circuit from the
  published paper. This module was NOT verified against the paper. Instead it
  realizes the V3 RUS *channel* EXACTLY -- the Kraus operators are
      M0 = sqrt(5/8) * V3   (success, ancilla=0),   M1 = sqrt(3/8) * I  (failure),
  so success probability is exactly 5/8 (any input), failure is exactly the identity
  on the data, and success delivers V3|psi> exactly. The GATE-LEVEL circuit here (a
  single data-multiplexed ancilla rotation, qsim.ctrl_branch) is a stand-in; the
  published circuit may use a different but equivalent gate sequence (expected 2-4
  two-qubit gates). REPLACE with the transcribed circuit and record the figure/eq
  number before this counts as the spec's rus_data. n2q is recorded below.

Derivation of the gadget: with data |psi> and a fresh ancilla |0>, apply the
data-controlled ancilla unitary U_d (d = data comp-basis value):
    U_d |0>_a = a0(d)|0>_a + a1|1>_a,   a0(d) = (1 + 2i(-1)^d)/(2 sqrt2),  a1 = sqrt(3/8)
measure the ancilla. Outcome 0 (prob |a0|^2 = 5/8) projects data -> M0|psi> with
M0 = diag(a0(0), a0(1)) = sqrt(5/8) diag(e^{i atan2}, e^{-i atan2}) = sqrt(5/8) V3.
Outcome 1 (prob 3/8) projects data -> a1|psi> = M1|psi>, i.e. the identity. No
correction is needed (failure is exactly identity), so the carried state stays |psi>
at every iteration boundary.

Because failure is identity, the ideal final state is the FIXED V3|psi> (trip-count
independent) -- the cleanest delivered-fidelity metric of any knit benchmark: plain
3-basis tomography against one target, no per-shot reference (unlike ipe_project).

Keep in lockstep with mlir/test/Quantum/Purl/rus_data_{unknown,knit}.mlir.
"""

import math

import numpy as np

from sim.qsim import QSim, RY, RZ

N_WIRES = 2
DATA = 0                 # carried, never measured in the loop
ANCILLA = 1             # RUS ancilla, reset + reused each iteration
P_ANALYTIC = 5.0 / 8.0  # published per-attempt success probability
N2Q_PER_ITER = 1        # 2q gates on the data wire per attempt (this reconstruction)

PREP_RY = 0.4           # |psi> = Rz(0.7) Ry(0.4)|0> -- generic non-stabilizer state
PREP_RZ = 0.7
_ALPHA = math.atan(2.0)  # V3 = diag(e^{i alpha}, e^{-i alpha}), alpha = atan(2)

# V3 = (I + 2iZ)/sqrt(5)
V3 = np.array([[(1 + 2j) / math.sqrt(5), 0], [0, (1 - 2j) / math.sqrt(5)]],
              dtype=complex)

# data-multiplexed ancilla unitaries U_0, U_1 (see module doc)
_A1 = math.sqrt(3.0 / 8.0)
def _Ud(d):
    a0 = (1 + 2j * (1 if d == 0 else -1)) / (2 * math.sqrt(2))
    return np.array([[a0, -_A1], [_A1, np.conj(a0)]], dtype=complex)
U0, U1 = _Ud(0), _Ud(1)


def _psi():
    """State vector of |psi> = Rz(0.7) Ry(0.4)|0> (2-vector)."""
    return RZ(PREP_RZ) @ RY(PREP_RY) @ np.array([1, 0], dtype=complex)


def ideal_bloch():
    """Bloch vector of the delivered ideal V3|psi> (tomography reference)."""
    v = V3 @ _psi()
    rho = np.outer(v, v.conj())
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    return np.array([np.trace(rho @ M).real for M in (X, Y, Z)])


IDEAL_BLOCH = ideal_bloch()
Z_IDEAL = float(IDEAL_BLOCH[2])


def prepare_input(sim):
    """Program input on the carried data wire: |psi> = Rz(0.7) Ry(0.4)|0>."""
    sim.ry(DATA, PREP_RY)
    sim.rz(DATA, PREP_RZ)


def attempt(sim):
    """One RUS attempt. The data-controlled ancilla gadget applies the V3 channel;
    the ancilla is measured. Outcome 0 = SUCCESS (data now V3|psi>, exit); outcome 1
    = FAILURE (data unchanged = identity, repeat). Returns the fail flag (bool)."""
    a, d = ANCILLA, DATA
    sim.force_zero(a)                 # fresh ancilla |0>
    sim.ctrl_branch(d, a, U0, U1)     # data-multiplexed rotation (the V3 RUS gadget)
    m = sim.measure(a)                # 0 = success, 1 = failure
    sim.feedback(active=[d])          # data idles through the feedback window
    return bool(m == 1)


def prep_fast(sim):
    """1-qubit prep hook (unused: rus_data drives the full 2-wire sim)."""
    sim.ry(0, PREP_RY)
    sim.rz(0, PREP_RZ)


def run_unbounded(rng, calib=None, lam=1.0, max_trips=500):
    """Unbounded RUS: retry until success. Returns (trip_count, z_sample)."""
    sim = QSim(N_WIRES, calib=calib, lam=lam, rng=rng)
    prepare_input(sim)
    k, fail = 0, True
    while fail and k < max_trips:
        fail = attempt(sim)
        k += 1
    z = 1 - 2 * sim.measure(DATA)
    return k, z


if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    N = 6000
    trips, zs = [], []
    for _ in range(N):
        k, z = run_unbounded(rng, lam=0.0)
        trips.append(k)
        zs.append(z)
    p_hat = 1.0 / (sum(trips) / len(trips))
    mu = float(np.mean(zs))
    se = float(np.std(zs, ddof=1) / np.sqrt(N))
    print(f"rus_data (lam=0, N={N}): p_hat={p_hat:.4f} (published {P_ANALYTIC})")
    print(f"  <Z> = {mu:+.4f} +- {se:.4f}  (ideal V3|psi> <Z> = {Z_IDEAL:+.4f})")
    print(f"  ideal Bloch V3|psi> = {IDEAL_BLOCH.round(4)}  n2q/iter={N2Q_PER_ITER}")
