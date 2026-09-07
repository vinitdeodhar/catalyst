"""
eval/programs.py -- the benchmarks AS Catalyst @qjit programs.

Each builder returns a @qjit callable compiled with BOTH Purl passes in the pipeline
(@purl selects the strategy, @purl_lower_qcut lowers the cut) running on the noisy
QSimDevice (the qsim runtime device fed the shared calibration). The eval driver
(run_all.py) executes these uniformly -- one path for every benchmark: compile, then
average the returned expval over shots/seeds. No per-benchmark execution logic; the
passes do the cutting, the device does the noise.

A program is a plain algorithm (prepare a held state, run a measurement-conditioned
loop, return an expval); the ideal delivered value is computed classically for the
fidelity score. This mirrors how a real Catalyst user writes the circuit.
"""

import math

import numpy as np
import pennylane as qml
from catalyst import qjit, while_loop, measure, cond
from catalyst.passes import purl, purl_lower_qcut

from eval.qsim_device import QSimDevice

CALIB = "benchmarks/ibm_eagle_r3.json"


# Trajectory averaging (how Catalyst runs noisy measurement-conditioned circuits):
# mcm_method="one-shot" re-runs the whole circuit `shots` times, each an INDEPENDENT
# noise trajectory (the device PRNG advancing), and averages the terminal expval;
# qjit(seed=...) gives independent ensembles for seed error bars.
def _dev(wires, lam, shots, carry_qubit=0, calib=CALIB):
    return QSimDevice(wires=wires, calib=calib, carry_qubit=carry_qubit, lam=lam,
                      shots=shots)


# --- rus (= rus_lowp): held magic state |psi0>=H T H T H|0>, low-p CNOT herald.
# Provable identity on the held wire -> the pass selects REFRESH. Ideal <Z> = 0.5.
def rus(lam, seed, shots=1500, p=0.1, calib=CALIB):
    dev = _dev(2, lam, shots, 0, calib)

    @qjit(seed=seed)
    @purl_lower_qcut
    @purl(calib=calib, p=p, shots=6000)
    @qml.qnode(dev, mcm_method="one-shot")
    def f():
        qml.Hadamard(0); qml.T(0); qml.Hadamard(0); qml.T(0); qml.Hadamard(0)

        @while_loop(lambda c: c)
        def loop(c):
            qml.CNOT(wires=[0, 1])            # net-identity touch on the held wire
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(1)
            m = measure(1)                    # low-p herald (target-independent)
            return m

        loop(True)
        return qml.expval(qml.PauliZ(0))

    return f, 0.5   # (program, ideal <Z>)


# --- ipe: held eigenstate |+>, adaptive herald loop. Provable identity -> REFRESH.
# Ideal <X> = 1 (read via the returned PauliX expval).
def ipe(lam, seed, shots=1500, p=0.12, calib=CALIB):
    dev = _dev(2, lam, shots, 0, calib)

    @qjit(seed=seed)
    @purl_lower_qcut
    @purl(calib=calib, p=p, shots=6000)
    @qml.qnode(dev, mcm_method="one-shot")
    def f():
        qml.Hadamard(0)                       # held eigenstate |+>

        @while_loop(lambda c: c)
        def loop(c):
            qml.CNOT(wires=[0, 1])            # net-identity phase-kickback touch
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(1)
            m = measure(1)
            return m

        loop(True)
        return qml.expval(qml.PauliX(0))

    return f, 1.0   # (program, ideal <X>)


# --- qwalk: fat-tailed random-walk herald, 2q-heavy net-identity non-Clifford body.
# Held reference Rz(0.7)Ry(0.4)|0> (ideal <Z>=cos(0.4)=0.9211). Unknown state (the
# non-Clifford T in the sandwich) -> the pass selects migrate/knit/none. The walk
# position (classical) is threaded through the loop and stepped by a measured |+> bit.
_QW_RY, _QW_RZ = 0.4, 0.7


def qwalk(lam, seed, shots=1500, p=0.5, calib=CALIB, max_trips=60):
    dev = _dev(3, lam, shots, 0, calib)  # 0=data, 1=sandwich ancilla, 2=walk coin

    @qjit(seed=seed)
    @purl_lower_qcut
    @purl(calib=calib, p=p, shots=6000)
    @qml.qnode(dev, mcm_method="one-shot")
    def f():
        qml.RY(_QW_RY, wires=0); qml.RZ(_QW_RZ, wires=0)     # held reference

        @while_loop(lambda pos, k: (pos != 0) & (k < max_trips))
        def loop(pos, k):
            # 2q-heavy NET-IDENTITY non-Clifford touch on the data (n2q = 6)
            qml.Hadamard(1)
            for _ in range(3):
                qml.CNOT(wires=[1, 0]); qml.T(1); qml.CNOT(wires=[1, 0])
            measure(1, reset=True)
            # random-walk step from a fresh |+> coin: +1 if 1 else -1
            qml.Hadamard(2)
            b = measure(2, reset=True)
            return pos + (2 * b - 1), k + 1

        loop(1, 0)
        return qml.expval(qml.PauliZ(0))

    return f, math.cos(_QW_RY)   # ideal <Z> = cos(0.4) = 0.9211


# --- ipe_project: phase-estimation-as-projection. Held SUPERPOSITION of the
# eigenstates of U=Rz(theta); each round's ancilla measurement partially projects it.
# Unknown state -> knit/migrate. A classical Bayesian posterior (threaded through the
# loop) drives the adaptive stop; at exit the delivered wire is aligned to the shot's
# posterior WINNER (an X-frame flip) so <Z> directly scores the per-shot-reference
# fidelity (ideal <Z> = +1).
_THETA = 2.0 * math.pi / 7.0
_ALPHA = math.pi / 8.0
# fixed round likelihoods L(b|i) (see benchmarks/ipe_project.py); precomputed:
_L00 = math.cos(-_THETA / 4 + math.pi / 4) ** 2   # P(b=0 | eigenstate 0)
_L01 = math.cos(_THETA / 4 + math.pi / 4) ** 2    # P(b=0 | eigenstate 1)


def ipe_project(lam, seed, shots=1500, p=0.45, calib=CALIB, thresh=0.87,
                max_trips=10):
    dev = _dev(2, lam, shots, 0, calib)          # 0=data, 1=ancilla
    prior0 = math.cos(_ALPHA) ** 2

    @qjit(seed=seed)
    @purl_lower_qcut
    @purl(calib=calib, p=p, shots=6000)
    @qml.qnode(dev, mcm_method="one-shot")
    def f():
        qml.RY(2.0 * _ALPHA, wires=0)            # cos(a)|0> + sin(a)|1> (unknown)

        @while_loop(lambda p0, k: ((p0 < thresh) & ((1.0 - p0) < thresh))
                    & (k < max_trips))
        def loop(p0, k):
            qml.Hadamard(1)
            qml.ctrl(qml.RZ(_THETA, wires=0), control=1)   # controlled-U (non-Clifford)
            qml.S(1)                                        # feedback
            qml.Hadamard(1)
            b = measure(1, reset=True)
            # Bayesian update from the measured bit (b in {0,1}); L for b=1 = 1-L(b=0)
            L0 = _L00 * (1 - b) + (1.0 - _L00) * b
            L1 = _L01 * (1 - b) + (1.0 - _L01) * b
            num = p0 * L0
            p0n = num / (num + (1.0 - p0) * L1)
            return p0n, k + 1

        p0f, _ = loop(prior0, 0)
        # align the delivered wire to the shot's posterior winner: if winner==1
        # (p0<0.5) flip the Z-frame, so <Z> scores fidelity to |winner> (ideal +1)
        cond(p0f < 0.5, lambda: qml.PauliX(0))()
        return qml.expval(qml.PauliZ(0))

    return f, 1.0   # ideal <Z> = +1 (winner-aligned per-shot reference)


# The active suite is rus + ipe (single carried slot -> refresh, fully supported
# through the real Catalyst pipeline). qwalk and ipe_project are written above as
# faithful @qjit programs, but their persistent ancilla wires present as MULTIPLE
# carried register slots, which the pass rejects ("multi-wire cut unsupported", a
# deliberate single-carry-slot limitation pinned by two_carry.mlir). Enabling them
# needs either restructuring so only the data wire is carried, or extending the pass
# to multi-slot carries. Kept here (not active) pending that.
PROGRAMS = {"rus": rus, "ipe": ipe}
BLOCKED = {"ipe_project": ipe_project, "qwalk": qwalk}  # pass: multi-slot unsupported
