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


# Two arms through the identical compiled path (§14.7): purl_on=True applies both
# passes (the pass selects and lowers the cut); purl_on=False omits them (the
# unbounded baseline -- same program, no cut). Only the passes differ.
def _build(qnode_fn, ideal, seed, keep, purl_on, calib, p):
    fn = qnode_fn
    if purl_on:
        fn = purl(calib=calib, p=p, shots=6000)(fn)
        fn = purl_lower_qcut(fn)
    return qjit(fn, seed=seed, keep_intermediate=keep), ideal


# --- rus: held magic state |psi0>=H T H T H|0>, low-p herald. The held wire idles
# (untouched) through each attempt, so it is an untouched register slot -> provable
# identity -> REFRESH, which re-prepares the ideal state and clears the idle T1/T2
# decoherence accumulated over the hold (escapes the §11 no-go; no leakage needed).
# Ideal <Z> = 0.5.
def rus(lam, seed, shots=1500, p=0.1, calib=CALIB, keep=False, purl_on=True):
    dev = _dev(2, lam, shots, 0, calib)
    # biased herald coin so the loop STOPS with prob p (mean trips = 1/p): RY(theta)|0>
    # has P(measure 0) = cos^2(theta/2) = p, so theta = 2*acos(sqrt(p)). Matching the
    # runtime p to the p the pass is told makes mean depth (1/p) exceed the cap C.
    theta = 2.0 * math.acos(math.sqrt(p))

    @qml.qnode(dev, mcm_method="one-shot")
    def f():
        qml.Hadamard(0); qml.T(0); qml.Hadamard(0); qml.T(0); qml.Hadamard(0)

        @while_loop(lambda c, k: c)
        def loop(c, k):
            qml.RY(theta, wires=1)            # low-p herald on a FRESH |0> coin
            m = measure(1, reset=True)        # reset -> next RY sees |0> (stable p)
            return m, k + 1                   # k = trip counter (runtime depth)

        _, k = loop(True, 0)
        return qml.expval(qml.PauliZ(0)), k

    return _build(f, 0.5, seed, keep, purl_on, calib, p)   # (program, ideal <Z>)


# --- ipe: held eigenstate |+>, adaptive herald loop. The held wire idles (untouched)
# through each round -> untouched register slot -> provable identity -> REFRESH clears
# the idle decoherence accumulated over the hold. Ideal <X> = 1 (returned PauliX).
def ipe(lam, seed, shots=1500, p=0.12, calib=CALIB, keep=False, purl_on=True):
    dev = _dev(2, lam, shots, 0, calib)
    # biased herald coin: STOP prob p (mean trips = 1/p), theta = 2*acos(sqrt(p)) so
    # mean depth exceeds the cap C and refresh fires on most shots (see rus).
    theta = 2.0 * math.acos(math.sqrt(p))

    @qml.qnode(dev, mcm_method="one-shot")
    def f():
        qml.Hadamard(0)                       # held eigenstate |+>

        @while_loop(lambda c, k: c)
        def loop(c, k):
            qml.RY(theta, wires=1)            # low-p herald on a FRESH |0> coin
            m = measure(1, reset=True)        # reset -> next RY sees |0> (stable p)
            return m, k + 1                   # k = trip counter (runtime depth)

        _, k = loop(True, 0)
        return qml.expval(qml.PauliX(0)), k

    return _build(f, 1.0, seed, keep, purl_on, calib, p)   # (program, ideal <X>)


# --- qwalk: fat-tailed random-walk herald, 2q-heavy net-identity non-Clifford body.
# Held reference Rz(0.7)Ry(0.4)|0> (ideal <Z>=cos(0.4)=0.9211). Unknown state (the
# non-Clifford T in the sandwich) -> the pass selects migrate/knit/none. The walk
# position (classical) is threaded through the loop and stepped by a measured |+> bit.
_QW_RY, _QW_RZ = 0.4, 0.7


def qwalk(lam, seed, shots=1500, p=0.5, calib=CALIB, max_trips=60, keep=False,
          purl_on=True):
    dev = _dev(3, lam, shots, 0, calib)  # 0=data, 1=sandwich ancilla, 2=walk coin

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
            return pos + (2 * b - 1), k + 1     # k = trip counter (runtime depth)

        _, k = loop(1, 0)
        return qml.expval(qml.PauliZ(0)), k

    # ideal <Z> = cos(0.4) = 0.9211
    return _build(f, math.cos(_QW_RY), seed, keep, purl_on, calib, p)


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
                max_trips=10, keep=False, purl_on=True):
    dev = _dev(2, lam, shots, 0, calib)          # 0=data, 1=ancilla
    prior0 = math.cos(_ALPHA) ** 2

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

        p0f, k = loop(prior0, 0)             # k = trip counter (runtime depth)

        # align the delivered wire to the shot's posterior winner: if winner==1
        # (p0<0.5) flip the Z-frame, so <Z> scores fidelity to |winner> (ideal +1)
        @cond(p0f < 0.5)
        def _align():
            qml.PauliX(0)         # side effect only (both branches return None)

        _align()
        return qml.expval(qml.PauliZ(0)), k

    # ideal <Z> = +1 (winner-aligned per-shot reference)
    return _build(f, 1.0, seed, keep, purl_on, calib, p)


# All four benchmarks are Python @qjit programs compiled through the ENTIRE Catalyst
# pipeline with both purl passes active (spec §14 hard requirement). The pass decides
# per program: rus/ipe (provable identity) -> refresh; qwalk/ipe_project (unknown) ->
# the cost model selects (none where migrate is not cost-positive on the calibration;
# migrate fires where a cheap partner edge makes it profitable). No benchmark names a
# strategy or invokes a pass out-of-band.
PROGRAMS = {"rus": rus, "ipe": ipe, "qwalk": qwalk, "ipe_project": ipe_project}


def probe_strategy(builder, calib):
    """Extract the PASS's decision from the real compiled pipeline (spec §8.2: the
    eval must report the pass-selected strategy). Compiles the benchmark once with
    keep_intermediate and reads purl.strategy / purl.C / purl.applied /
    purl.predicted_fidelity from the post-QuantumCompilation IR (the passes' own
    output in-pipeline -- not an out-of-band quantum-opt call). The decision depends
    on p/calib, not lam/seed, so one probe per (benchmark, calib) suffices."""
    import glob
    import os
    import re
    import tempfile

    d = tempfile.mkdtemp(prefix="purl_probe_")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        f, _ = builder(lam=0.0, seed=0, shots=1, calib=os.path.join(cwd, calib),
                       keep=True)
        f()  # trigger compilation (dumps stage IR under ./<fn>/)
        files = glob.glob(os.path.join(d, "*", "*AfterQuantumCompilationStage.mlir"))
        text = open(files[0]).read() if files else ""
    finally:
        os.chdir(cwd)
    strat = re.search(r'purl\.strategy = "([a-z]+)"', text)
    C = re.search(r'purl\.C = (\d+)', text)
    pred = re.search(r'purl\.predicted_fidelity = [^}]*bounded = ([0-9.eE+-]+)', text)
    return {
        "strategy": strat.group(1) if strat else "none",
        "C": int(C.group(1)) if C else None,
        "applied": "purl.applied = true" in text,
        "predicted_bounded": float(pred.group(1)) if pred else None,
    }
