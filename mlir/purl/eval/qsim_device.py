"""
qsim_device.py -- the PennyLane/Catalyst device that runs a compiled Catalyst
program on the Purl noisy trajectory simulator (the C++ QSimQubit runtime device,
runtime/lib/backend/qsim_qubit). It flattens the SHARED hardware calibration JSON
(the one the passes read) with the SAME loader (ibm_dataset.carried_calib) and
forwards the flat rates + the global noise scale `lam` to the C++ device as kwargs.

Usage (the two Purl passes go in the pipeline; qsim replaces lightning.qubit):

    from eval.qsim_device import QSimDevice
    dev = QSimDevice(wires=2, calib="benchmarks/ibm_eagle_r3.json", carry_qubit=0, lam=1.0)

    @qjit
    @purl_lower_qcut
    @purl(calib="benchmarks/ibm_eagle_r3.json", p=0.1, shots=6000)
    @qml.qnode(dev)
    def circuit(): ...

Requires the runtime device built: `make runtime` (produces librtd_qsim_qubit.so).
"""

import os
import platform

from pennylane.devices import Device

from sim.ibm_dataset import carried_calib, JSON_PATH

# repo root = .../catalyst ; the built device lives under runtime/build/lib
_PURL = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_ROOT = os.path.abspath(os.path.join(_PURL, os.pardir, os.pardir))
_RT_LIB = os.path.join(_ROOT, "runtime", "build", "lib")
_EXT = ".dylib" if platform.system() == "Darwin" else ".so"


class QSimDevice(Device):
    """Purl noisy trajectory device (qjit-only): the C++ QSimQubit backend fed the
    flattened shared calibration + noise scale `lam`."""

    config_filepath = os.path.join(_RT_LIB, "backend", "qsim_qubit.toml")

    @staticmethod
    def get_c_interface():
        # the identifier MUST match the GENERATE_DEVICE_FACTORY id (QSimQubit)
        return "QSimQubit", os.path.join(_RT_LIB, "librtd_qsim_qubit" + _EXT)

    def __init__(self, wires, calib=JSON_PATH, carry_qubit=0, lam=1.0, shots=None):
        super().__init__(wires=wires, shots=shots)
        # flatten via the SAME loader the passes use -> forward the flat rates as
        # device kwargs (strings); the C++ device parses them (parse_kwargs).
        c = carried_calib(carry_qubit, path=calib)
        keys = ("gate_1q", "gate_2q", "readout", "tau", "T1", "T2",
                "p1", "p2", "p_ro", "p_meas", "p_leak")
        self.device_kwargs = {k: repr(float(c[k])) for k in keys}
        self.device_kwargs["lam"] = repr(float(lam))
        self.device_kwargs["calib"] = str(calib)  # provenance (on record)

    def execute(self, circuits, execution_config=None):
        # qjit path executes via the C++ QuantumDevice; the Python path is unused.
        raise NotImplementedError("QSimDevice is qjit-only (compiled runtime device)")
