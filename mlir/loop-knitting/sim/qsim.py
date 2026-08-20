"""
qsim.py -- pure-NumPy Monte-Carlo trajectory simulator for loop-knitting eval.

One pure statevector per shot; noise via stochastic channel sampling (quantum
trajectories / Monte-Carlo wavefunction). No Catalyst dependency: benchmarks are
mirrored as Python programs that call this op API one-to-one with the MLIR.

Implements Track B of the loop-knitting implementation spec (Section 4):
  * Op set (4.1): alloc, h, x, z, s, sdg, t, cnot, toffoli, measure, reset.
  * Noise model (4.2): per-gate depolarizing, measurement readout flip +
    pre-measurement depolarizing, and -- the term the transform targets --
    idle amplitude damping + pure dephasing accrued as wall-clock advances on
    qubits that are NOT being operated on (other qubits' gates, the readout
    window, and the feedback window tau).

All noise is scaled by a global scalar `lam` in [0, 4]:  lam = 0 is exactly
noiseless (must reproduce the noiseless benchmark numbers); lam = 1 is the
calibrated backend.

Calibration schema (JSON, shared with the pass's 3.2 option; extended with the
error probabilities the noise model needs):

  {
    # durations (seconds) -- drive idle decay and the pass depth analysis
    "gate_1q": 30e-9, "gate_2q": 60e-9, "readout": 700e-9, "tau": 500e-9,
    "T1": 150e-6, "T2": 200e-6,
    # error probabilities -- drive the discrete channels (per application)
    "p1": 1e-3,      # 1q depolarizing prob
    "p2": 1e-2,      # 2q/3q depolarizing prob
    "p_ro": 1e-2,    # classical readout bit-flip prob
    "p_meas": 5e-3   # pre-measurement depolarizing prob
  }

`calib="unit"` (handled by load_calib) sets every duration/tau/readout to 1 for
layer counting; it is meant for the pass, not for physically meaningful noise.
"""

import json
import math
import cmath

import numpy as np

# ---------------------------------------------------------------------------
# Single-qubit operators (2x2). Convention: qubit q occupies bit q of the
# statevector index (qubit 0 = least significant bit).
# ---------------------------------------------------------------------------
_ISQRT2 = 1.0 / math.sqrt(2.0)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = _ISQRT2 * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)

_PAULIS_1Q = [X, Y, Z]  # non-identity single-qubit Paulis (depolarizing draws)


DEFAULT_CALIB = {
    "gate_1q": 30e-9, "gate_2q": 60e-9, "readout": 700e-9, "tau": 500e-9,
    "T1": 150e-6, "T2": 200e-6,
    "p1": 1e-3, "p2": 1e-2, "p_ro": 1e-2, "p_meas": 5e-3,
    # Leakage: population leaving the computational subspace during idle. A
    # *fresh* qubit (the cut's "continue on a fresh qubit") is leakage-free, so
    # reset / cut-reprep clears it -- this is the non-Markovian, reset-clearable
    # error the transform actually reduces (see NOTES.md). Default OFF
    # (T_leak = inf) so it never touches the Markovian validation gates; the
    # evaluation opts in with a finite T_leak.
    "T_leak": float("inf"),
}


def load_calib(spec):
    """spec: dict, path to JSON, 'unit', or None -> merged calibration dict."""
    if spec is None:
        return dict(DEFAULT_CALIB)
    if spec == "unit":
        c = dict(DEFAULT_CALIB)
        for k in ("gate_1q", "gate_2q", "readout", "tau"):
            c[k] = 1.0
        c["T1"] = c["T2"] = float("inf")
        return c
    if isinstance(spec, dict):
        c = dict(DEFAULT_CALIB)
        c.update(spec)
        return c
    with open(spec) as fh:
        c = dict(DEFAULT_CALIB)
        c.update(json.load(fh))
    return c


class QSim:
    """A single-shot trajectory. Construct one per shot (or call .reset_all)."""

    def __init__(self, n, calib=None, lam=1.0, rng=None):
        self.n = n
        self.calib = load_calib(calib)  # merges defaults; idempotent on dicts
        self.lam = float(lam)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.psi = np.zeros(2 ** n, dtype=complex)
        self.psi[0] = 1.0
        # per-qubit "live" flag: idle decay only accrues on live qubits.
        self.live = [True] * n
        # per-qubit leakage flag: True once the qubit has left the computational
        # subspace. A leaked qubit reads out as garbage; reset / cut-reprep
        # (a fresh physical qubit) clears it.
        self.leaked = [False] * n
        Tl = self.calib.get("T_leak", float("inf"))
        self._inv_Tleak = 0.0 if math.isinf(Tl) else 1.0 / Tl
        # pure-dephasing time constant Tphi from T1,T2: 1/T2 = 1/(2T1)+1/Tphi
        T1, T2 = self.calib["T1"], self.calib["T2"]
        if math.isinf(T1) or math.isinf(T2):
            self._inv_Tphi = 0.0
            self._inv_T1 = 0.0
        else:
            self._inv_T1 = 1.0 / T1
            inv_tphi = 1.0 / T2 - 1.0 / (2.0 * T1)
            self._inv_Tphi = max(0.0, inv_tphi)

    # -- statevector helpers ------------------------------------------------
    def _apply_1q(self, U, q):
        psi = self.psi.reshape([2] * self.n)  # axis order: qubit n-1 ... 0? no
        # index bit q -> axis (n-1-q) if we treat reshape MSB-first. Keep it
        # simple & explicit with a move: bring qubit q to axis 0.
        ax = self.n - 1 - q
        psi = np.moveaxis(psi, ax, 0)
        psi = np.tensordot(U, psi, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, ax)
        self.psi = psi.reshape(-1)

    def _apply_ctrl(self, U, controls, target):
        """Apply U on target conditioned on all controls being |1>."""
        dims = [2] * self.n
        psi = self.psi.reshape(dims)
        # build slice selecting control bits = 1
        idx = [slice(None)] * self.n
        for c in controls:
            idx[self.n - 1 - c] = 1
        sub = psi[tuple(idx)]  # view over remaining axes incl. target
        # target axis within full tensor:
        tax = self.n - 1 - target
        # Since we've fixed control axes via integer index, sub has fewer dims;
        # recompute target axis position among the surviving axes.
        surviving = [a for a in range(self.n) if a not in
                     [self.n - 1 - c for c in controls]]
        tpos = surviving.index(tax)
        sub2 = np.moveaxis(sub, tpos, 0)
        sub2 = np.tensordot(U, sub2, axes=([1], [0]))
        sub2 = np.moveaxis(sub2, 0, tpos)
        psi[tuple(idx)] = sub2
        self.psi = psi.reshape(-1)

    def _prob_one(self, q):
        dims = [2] * self.n
        psi = self.psi.reshape(dims)
        idx = [slice(None)] * self.n
        idx[self.n - 1 - q] = 1
        amp = psi[tuple(idx)]
        return float(np.vdot(amp, amp).real)

    def _collapse(self, q, outcome):
        dims = [2] * self.n
        psi = self.psi.reshape(dims)
        idx = [slice(None)] * self.n
        idx[self.n - 1 - q] = 1 - outcome
        psi[tuple(idx)] = 0.0
        self.psi = psi.reshape(-1)
        nrm = np.linalg.norm(self.psi)
        if nrm > 0:
            self.psi /= nrm

    # -- idle decay ---------------------------------------------------------
    def _idle_qubit(self, q, dt):
        """Trajectory amplitude damping + pure dephasing on qubit q for dt."""
        if dt <= 0 or self.lam == 0.0:
            return
        eff = self.lam * dt
        # amplitude damping
        gamma = 1.0 - math.exp(-eff * self._inv_T1) if self._inv_T1 > 0 else 0.0
        if gamma > 0:
            p1 = self._prob_one(q)
            pjump = gamma * p1
            if self.rng.random() < pjump:
                # jump K1 = |0><1|: move population from 1 to 0, kill 1-branch
                dims = [2] * self.n
                psi = self.psi.reshape(dims).copy()
                idx0 = [slice(None)] * self.n
                idx1 = [slice(None)] * self.n
                idx0[self.n - 1 - q] = 0
                idx1[self.n - 1 - q] = 1
                psi[tuple(idx0)] = psi[tuple(idx1)]
                psi[tuple(idx1)] = 0.0
                self.psi = psi.reshape(-1)
            else:
                # no-jump K0 = diag(1, sqrt(1-gamma)); damp the 1-branch
                dims = [2] * self.n
                psi = self.psi.reshape(dims)
                idx1 = [slice(None)] * self.n
                idx1[self.n - 1 - q] = 1
                psi[tuple(idx1)] *= math.sqrt(1.0 - gamma)
                self.psi = psi.reshape(-1)
            nrm = np.linalg.norm(self.psi)
            if nrm > 0:
                self.psi /= nrm
        # pure dephasing: apply Z with prob p_pd/2 so coherence x exp(-eff/Tphi)
        if self._inv_Tphi > 0:
            p_pd = 1.0 - math.exp(-eff * self._inv_Tphi)
            if self.rng.random() < p_pd / 2.0:
                self._apply_1q(Z, q)
        # leakage: irreversible loss to a non-computational level during idle.
        # Reset / cut-reprep (a fresh qubit) clears it; that is why bounding the
        # hold length (knit) caps leakage while unbounded lets it accumulate.
        if self._inv_Tleak > 0 and not self.leaked[q]:
            p_leak = 1.0 - math.exp(-eff * self._inv_Tleak)
            if self.rng.random() < p_leak:
                self.leaked[q] = True

    def _idle_others(self, active, dt):
        """Advance idle decay on all live qubits except `active` for time dt."""
        if dt <= 0 or self.lam == 0.0:
            return
        act = set(active)
        for q in range(self.n):
            if q in act or not self.live[q]:
                continue
            self._idle_qubit(q, dt)

    def feedback(self, active=()):
        """Classical-feedback window: idle every live qubit by tau.

        `active` qubits (those being conditionally operated in the same window)
        are excluded. Call once per measurement-conditioned branch.
        """
        self._idle_others(active, self.calib["tau"])

    # -- gate noise ---------------------------------------------------------
    def _depol_1q(self, q):
        if self.lam == 0.0:
            return
        if self.rng.random() < self.lam * self.calib["p1"]:
            P = _PAULIS_1Q[self.rng.integers(3)]
            self._apply_1q(P, q)

    def _depol_nq(self, qubits):
        if self.lam == 0.0:
            return
        if self.rng.random() < self.lam * self.calib["p2"]:
            # sample a non-identity Pauli string over `qubits` (uniform over the
            # 4^k - 1 non-identity strings)
            k = len(qubits)
            while True:
                picks = [self.rng.integers(4) for _ in range(k)]  # 0=I
                if any(p != 0 for p in picks):
                    break
            for qb, p in zip(qubits, picks):
                if p == 1:
                    self._apply_1q(X, qb)
                elif p == 2:
                    self._apply_1q(Y, qb)
                elif p == 3:
                    self._apply_1q(Z, qb)

    # -- public op API (mirrors the MLIR ops) -------------------------------
    def alloc(self, n):
        assert n == self.n, "QSim register size is fixed at construction"

    def _gate1(self, U, q):
        self._idle_others([q], self.calib["gate_1q"])
        self._apply_1q(U, q)
        self._depol_1q(q)

    def h(self, q):
        self._gate1(H, q)

    def x(self, q):
        self._gate1(X, q)

    def z(self, q):
        self._gate1(Z, q)

    def s(self, q):
        self._gate1(S, q)

    def sdg(self, q):
        self._gate1(SDG, q)

    def t(self, q):
        self._gate1(T, q)

    def cnot(self, c, tq):
        self._idle_others([c, tq], self.calib["gate_2q"])
        self._apply_ctrl(X, [c], tq)
        self._depol_nq([c, tq])

    def cz(self, c, tq):
        self._idle_others([c, tq], self.calib["gate_2q"])
        self._apply_ctrl(Z, [c], tq)
        self._depol_nq([c, tq])

    def toffoli(self, a, b, tq):
        self._idle_others([a, b, tq], self.calib["gate_2q"])
        self._apply_ctrl(X, [a, b], tq)
        self._depol_nq([a, b, tq])

    def measure(self, q):
        """Born-sample + collapse. Returns {0,1} (post readout flip).

        A leaked qubit reads out as garbage (uniform random bit) -- it is no
        longer a computational state.
        """
        self._idle_others([q], self.calib["readout"])
        if self.leaked[q]:
            return int(self.rng.integers(2))
        # pre-measurement depolarizing
        if self.lam > 0 and self.rng.random() < self.lam * self.calib["p_meas"]:
            self._apply_1q(_PAULIS_1Q[self.rng.integers(3)], q)
        p1 = self._prob_one(q)
        outcome = 1 if self.rng.random() < p1 else 0
        self._collapse(q, outcome)
        # classical readout flip
        reported = outcome
        if self.lam > 0 and self.rng.random() < self.lam * self.calib["p_ro"]:
            reported = 1 - outcome
        return reported

    def force_zero(self, q):
        """Continue on a fresh qubit: clear leakage and (re)initialise q to |0>,
        preserving the other wires' state.

        Models allocating/replacing the physical qubit -- leakage-free. Assumes q
        is unentangled from the rest (always true here: force_zero is only called
        on freshly-measured/collapsed qubits, or on a leaked qubit being reset).
        No readout; the caller accounts for any timing separately.
        """
        self.leaked[q] = False
        dims = [2] * self.n
        ax = self.n - 1 - q
        psi = self.psi.reshape(dims)
        s0 = np.take(psi, 0, axis=ax)
        s1 = np.take(psi, 1, axis=ax)
        R = s0 if np.linalg.norm(s0) >= np.linalg.norm(s1) else s1  # rest state
        nR = np.linalg.norm(R)
        newpsi = np.zeros(dims, dtype=complex)
        slicer = [slice(None)] * self.n
        slicer[ax] = 0
        newpsi[tuple(slicer)] = R / nR if nR > 1e-12 else 0.0
        if nR <= 1e-12:                              # degenerate: set wire to |0>
            flat = newpsi.reshape(-1)
            flat[0] = 1.0
            newpsi = flat.reshape(dims)
        self.psi = newpsi.reshape(-1)

    def reset(self, q):
        """Reset q to |0> via measure-and-conditional-X (consumes readout).

        Uses a fresh qubit, so it clears any leakage on q.
        """
        out = self.measure(q)
        self.force_zero(q)
        _ = out

    # -- observables (for validation only; not a physical op) ---------------
    def prob_one(self, q):
        return self._prob_one(q)
