"""
ibm_dataset.py -- an IBM Eagle r3 (127-qubit) hardware dataset for the
loop-knitting pass and the Track-B evaluation.

Real published *median* device metrics for the Eagle r3 processor family
(ibm_sherbrooke / ibm_brisbane class), with a realistic deterministic per-qubit
spread, plus the heavy-hexagon coupling map with per-edge 2q (ECR) errors.

Sources: IBM Quantum device pages / calibration snapshots and the Eagle
processor papers (representative medians as of 2024 snapshots):
  * T1  median ~ 250 us         * T2 (Hahn-echo) median ~ 150 us
  * sqrt(X) 1q error median ~ 2.5e-4,  duration ~ 32 ns
  * ECR 2q error median ~ 8e-3,        duration ~ 560 ns
  * readout error median ~ 1.3e-2,     duration ~ 1.2 us
  * feedback/reset latency (dynamic circuits) ~ 1 us
NOTE: IBM's published properties do NOT include leakage; leakage is supplied
SEPARATELY as a knob by the evaluation (see IBM_LEAK_PER_2Q), not stored here.

`build_json()` writes ibm_eagle_r3.json (per-qubit + coupling, the schema the
pass reads). `carried_calib(qubit)` extracts a flat calibration dict for the
carried qubit (T1/T2/gate times/errors) usable by the pass and sim.qsim.
"""

import json
import math
import os

N_QUBITS = 127

# --- published Eagle r3 medians (SI units: seconds; probabilities) ---
MED = {
    "T1": 250e-6, "T2": 150e-6,
    "gate_1q_time": 32e-9, "gate_1q_err": 2.5e-4,
    "gate_2q_time": 560e-9, "gate_2q_err": 8e-3,
    "readout_time": 1.2e-6, "readout_err": 1.3e-2,
    "tau": 1.0e-6,          # classical feedback / reset latency
    "p_prep": 1e-3,         # state-preparation error (reset+init)
}

# Published IBM leakage estimate (per 2q gate) -- NOT a standard property; used
# by the evaluation as a separate knob (Eagle leakage characterizations report
# ~0.1-0.3% per two-qubit gate).
IBM_LEAK_PER_2Q = 1e-3

_HERE = os.path.dirname(__file__)
JSON_PATH = os.path.join(_HERE, os.pardir, "benchmarks", "ibm_eagle_r3.json")


def _spread(median, idx, rel=0.35):
    """Deterministic log-normal-ish per-qubit variation around a median."""
    # smooth pseudo-random factor in [1/(1+rel), 1+rel], seeded by qubit index
    u = (math.sin(idx * 12.9898) * 43758.5453)
    u = u - math.floor(u)                      # in [0,1)
    factor = math.exp(rel * (2.0 * u - 1.0))   # log-normal-ish, median 1
    return median * factor


def _heavy_hex_edges(n=N_QUBITS):
    """A heavy-hexagon-style coupling map for 127 qubits (Eagle layout).

    Rows of 15 'line' qubits joined by bridge qubits every other column, the IBM
    Eagle brick pattern. Approximates the real lattice degree/structure (used
    only for per-edge 2q errors; the fidelity model is per-qubit dominated)."""
    edges = []
    row_len = 15
    # 7 main rows of 15 = 105 line qubits; bridges fill to 127.
    rows = []
    q = 0
    for r in range(7):
        rows.append(list(range(q, q + row_len)))
        q += row_len
    # intra-row (nearest-neighbour) links
    for row in rows:
        for a, b in zip(row, row[1:]):
            edges.append([a, b])
    # inter-row bridges: connect every 4th column via a bridge qubit
    for r in range(6):
        top, bot = rows[r], rows[r + 1]
        for c in range(0, row_len, 4):
            if q >= n:
                break
            bridge = q
            q += 1
            edges.append([top[c], bridge])
            edges.append([bridge, bot[c]])
    return edges


def build(seed_median=MED):
    qubits = []
    for i in range(N_QUBITS):
        qubits.append({
            "id": i,
            "T1": round(_spread(seed_median["T1"], i), 12),
            "T2": round(_spread(seed_median["T2"], i + 1), 12),
            "gate_1q_err": round(_spread(seed_median["gate_1q_err"], i + 2), 9),
            "readout_err": round(_spread(seed_median["readout_err"], i + 3), 6),
        })
    edges = _heavy_hex_edges()
    edge_list = []
    for j, (a, b) in enumerate(edges):
        edge_list.append({
            "qubits": [a, b],
            "gate_2q_err": round(_spread(seed_median["gate_2q_err"], j + 100), 6),
        })
    return {
        "device": "ibm_eagle_r3 (representative published medians)",
        "n_qubits": N_QUBITS,
        "gate_1q_time": seed_median["gate_1q_time"],
        "gate_2q_time": seed_median["gate_2q_time"],
        "readout_time": seed_median["readout_time"],
        "tau": seed_median["tau"],
        "p_prep": seed_median["p_prep"],
        "qubits": qubits,
        "edges": edge_list,
    }


def build_json(path=JSON_PATH):
    data = build()
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return path


def load(path=JSON_PATH):
    if not os.path.exists(path):
        build_json(path)
    with open(path) as fh:
        return json.load(fh)


def _median_2q_err(data):
    errs = sorted(e["gate_2q_err"] for e in data["edges"])
    return errs[len(errs) // 2]


def carried_calib(qubit=0, path=JSON_PATH, p_leak=IBM_LEAK_PER_2Q):
    """Flat calibration for the CARRIED qubit (schema sim.qsim/the pass read).

    T1/T2/readout/1q from qubit `qubit`; 2q error = device median; leakage from
    the separate `p_leak` knob (default the published IBM estimate)."""
    d = load(path)
    q = d["qubits"][qubit]
    return {
        "gate_1q": d["gate_1q_time"], "gate_2q": d["gate_2q_time"],
        "readout": d["readout_time"], "tau": d["tau"],
        "T1": q["T1"], "T2": q["T2"],
        "p1": q["gate_1q_err"], "p2": _median_2q_err(d),
        "p_ro": q["readout_err"], "p_meas": q["gate_1q_err"],
        # leakage is the separate per-2q-gate knob (spec 4.1); the sim charges it
        # on each 2q gate the carried wire participates in (qsim._leak_2q).
        "p_leak": p_leak, "p_prep": d["p_prep"],
    }


if __name__ == "__main__":
    p = build_json()
    d = load(p)
    print(f"wrote {p}: {d['n_qubits']} qubits, {len(d['edges'])} edges")
    print(f"median 2q err = {_median_2q_err(d):.4f}")
    c = carried_calib(0)
    print("carried qubit 0 calib:")
    for k in ("T1", "T2", "p1", "p2", "p_ro", "gate_2q", "readout", "tau"):
        print(f"  {k} = {c[k]}")
