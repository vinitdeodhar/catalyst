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
from catalyst import qjit, while_loop, measure
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


PROGRAMS = {"rus": rus, "ipe": ipe}
