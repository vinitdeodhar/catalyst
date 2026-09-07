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
Leakage IS stored here (spec 4.1): a REQUIRED top-level `leak_2q_default`
(per-two-qubit-gate probability) with OPTIONAL per-edge `leak_2q` overrides, plus
a `leak_source` provenance string. IBM does not publish leakage, so files record
where the number came from. The old `--leak` / `p-leak` knob is GONE -- leakage is
now single-source calibration data.

`build_json()` writes ibm_eagle_r3.json (per-qubit + coupling, the schema the
pass reads). `carried_calib(qubit)` extracts a flat calibration dict for the
carried qubit (T1/T2/gate times/errors/leakage) usable by the pass and sim.qsim.
"""

import json
import math
import os
import warnings

N_QUBITS = 127

# --- published Eagle r3 medians (SI units: seconds; probabilities). ECR 2q gate. ---
MED = {
    "T1": 250e-6, "T2": 150e-6,
    "gate_1q_time": 32e-9, "gate_1q_err": 2.5e-4,
    "gate_2q_time": 560e-9, "gate_2q_err": 8e-3,
    "readout_time": 1.2e-6, "readout_err": 1.3e-2,
    "tau": 1.0e-6,          # classical feedback / reset latency
    "p_prep": 1e-3,         # state-preparation error (reset+init)
}

# --- Heron r2 representative medians (SI units). Tunable-coupler CZ 2q gate. These
# gate/coherence numbers track IBM's PUBLISHED Heron r2 calibration (ibm_fez /
# ibm_marrakesh class, 2024-2025): faster + far cleaner 2q gates than Eagle. Only
# leakage is NOT from these -- see IBM_LEAK_PER_2Q below (an estimate). ---
MED_HERON = {
    "T1": 250e-6, "T2": 180e-6,
    "gate_1q_time": 32e-9, "gate_1q_err": 2.0e-4,
    "gate_2q_time": 68e-9, "gate_2q_err": 3.0e-3,   # CZ median ~3e-3 (best edges ~1e-3)
    "readout_time": 1.2e-6, "readout_err": 1.0e-2,
    "tau": 1.0e-6,
    "p_prep": 1e-3,
}

# Per-2q-gate leakage ESTIMATE. IBM does NOT publish leakage for ANY device (Eagle or
# Heron); transmon CZ/CR leakage is characterized only in research papers, ~1e-3 order.
# So this is a literature-order estimate, NOT vendor calibration, and it is identical
# across Eagle/Heron JSONs (we have no device-specific leakage source). Recorded with
# provenance in `leak_source`; the gate/coherence numbers above are the real ones.
IBM_LEAK_PER_2Q = 1e-3

_HERE = os.path.dirname(__file__)
JSON_PATH = os.path.join(_HERE, os.pardir, "benchmarks", "ibm_eagle_r3.json")
HERON_JSON_PATH = os.path.join(_HERE, os.pardir, "benchmarks", "ibm_heron_r2.json")
# hardware name -> (medians, device label, json path) for the eval's --hardware switch
HARDWARE = {
    "eagle": (MED, "ibm_eagle_r3 (representative published medians; ECR 2q)", JSON_PATH),
    "heron": (MED_HERON, "ibm_heron_r2 (representative published medians; CZ 2q)",
              HERON_JSON_PATH),
}


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


def build(seed_median=MED, leak_2q_default=IBM_LEAK_PER_2Q, leak_spread=0.0,
          device="ibm_eagle_r3 (representative published medians)",
          partner_edge_leak=None, carry_qubit=0):
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
        e = {
            "qubits": [a, b],
            "gate_2q_err": round(_spread(seed_median["gate_2q_err"], j + 100), 6),
        }
        if leak_spread > 0.0:              # optional per-edge leakage spread (spec 4.1)
            e["leak_2q"] = round(_spread(leak_2q_default, j + 200, rel=leak_spread), 6)
        # optional hand-set CHEAP partner edge on the carry qubit (a what-if lever,
        # NOT vendor data): edges incident to `carry_qubit` get `partner_edge_leak`,
        # so the pass's pair selection can pick a low-leakage SWAP edge.
        if partner_edge_leak is not None and (a == carry_qubit or b == carry_qubit):
            e["leak_2q"] = partner_edge_leak
        edge_list.append(e)
    return {
        "device": device,
        "n_qubits": N_QUBITS,
        "gate_1q_time": seed_median["gate_1q_time"],
        "gate_2q_time": seed_median["gate_2q_time"],
        "readout_time": seed_median["readout_time"],
        "tau": seed_median["tau"],
        "p_prep": seed_median["p_prep"],
        # leakage is first-class calibration data (spec 4.1): a REQUIRED default plus
        # OPTIONAL per-edge overrides. IBM does not publish it -> record provenance.
        "leak_2q_default": leak_2q_default,
        "leak_source": "estimate, not vendor calibration (IBM does not publish leakage)",
        "qubits": qubits,
        "edges": edge_list,
    }


def build_json(path=JSON_PATH, leak_2q_default=IBM_LEAK_PER_2Q, leak_spread=0.0,
               seed_median=MED,
               device="ibm_eagle_r3 (representative published medians)",
               partner_edge_leak=None, carry_qubit=0, leak_source=None):
    data = build(seed_median=seed_median, leak_2q_default=leak_2q_default,
                 leak_spread=leak_spread, device=device,
                 partner_edge_leak=partner_edge_leak, carry_qubit=carry_qubit)
    if leak_source is not None:
        data["leak_source"] = leak_source
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return path


# Heron leakage model (spec-directed what-if): body per-2q leakage 3e-3 with a hand-set
# CHEAP 1e-4 partner edge on the carry qubit. Both are ESTIMATES/levers, not vendor data
# (IBM publishes no leakage); the 1e-4 edge is deliberately optimistic (30x below body).
HERON_LEAK_DEFAULT = 3e-3
HERON_PARTNER_EDGE_LEAK = 1e-4
_HERON_LEAK_SOURCE = (
    "estimate + what-if: body p_leak=3e-3 (literature-order), carry-qubit partner edge "
    "leak_2q=1e-4 (hand-set clean SWAP edge, NOT vendor data)")


def build_hardware(name, carry_qubit=0):
    """Write the JSON for a named hardware profile ('eagle' or 'heron')."""
    medians, device, path = HARDWARE[name]
    if name == "heron":
        return build_json(path=path, seed_median=medians, device=device,
                          leak_2q_default=HERON_LEAK_DEFAULT,
                          partner_edge_leak=HERON_PARTNER_EDGE_LEAK,
                          carry_qubit=carry_qubit, leak_source=_HERON_LEAK_SOURCE)
    return build_json(path=path, seed_median=medians, device=device)


def load(path=JSON_PATH):
    if not os.path.exists(path):
        build_json(path)
    with open(path) as fh:
        return json.load(fh)


def _median_2q_err(data):
    errs = sorted(e["gate_2q_err"] for e in data["edges"])
    return errs[len(errs) // 2]


def _median_leak(data):
    """Global median per-two-qubit-gate leakage over all edges (leak_2q_default
    fills edges without an explicit `leak_2q`), mirroring _median_2q_err. An
    edgeless map -> the default. (spec 4.1/4.2)"""
    dflt = data["leak_2q_default"]
    vals = sorted(e.get("leak_2q", dflt) for e in data["edges"])
    return vals[len(vals) // 2] if vals else dflt


def _validate_leak(data):
    """spec 4.2: `leak_2q_default` present and in [0,1] (hard error otherwise);
    every per-edge `leak_2q` in [0,1]; warn on >10x deviation (typo guard)."""
    if "leak_2q_default" not in data:
        raise ValueError(
            "calibration JSON missing required 'leak_2q_default': leakage moved "
            "from the --leak / p-leak knob into the JSON (PURL_SPEC.md 4.1); "
            "regenerate with ibm_dataset.build_json().")
    dflt = data["leak_2q_default"]
    if not 0.0 <= dflt <= 1.0:
        raise ValueError(f"leak_2q_default {dflt} out of range [0,1]")
    for e in data["edges"]:
        if "leak_2q" not in e:
            continue
        v = e["leak_2q"]
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"edge leak_2q {v} out of range [0,1]")
        if dflt > 0 and (v > 10 * dflt or v < 0.1 * dflt):
            warnings.warn(
                f"edge leak_2q {v} deviates >10x from leak_2q_default {dflt} "
                "(possible typo)")


def carried_calib(qubit=0, path=JSON_PATH):
    """Flat calibration for the CARRIED qubit (schema sim.qsim/the pass read).

    T1/T2/readout/1q from qubit `qubit`; 2q error and leakage = device (global)
    medians over the coupling map. Leakage is read from the JSON (spec 4.1), not a
    knob."""
    d = load(path)
    _validate_leak(d)
    q = d["qubits"][qubit]
    return {
        "gate_1q": d["gate_1q_time"], "gate_2q": d["gate_2q_time"],
        "readout": d["readout_time"], "tau": d["tau"],
        "T1": q["T1"], "T2": q["T2"],
        "p1": q["gate_1q_err"], "p2": _median_2q_err(d),
        "p_ro": q["readout_err"], "p_meas": q["gate_1q_err"],
        # per-2q-gate leakage from the JSON (spec 4.1), global median over edges;
        # the sim charges it on each 2q gate the carried wire touches (qsim._leak_2q).
        "p_leak": _median_leak(d), "p_prep": d["p_prep"],
    }


if __name__ == "__main__":
    for name in ("eagle", "heron"):
        p = build_hardware(name)
        d = load(p)
        print(f"[{name}] wrote {os.path.basename(p)}: {d['n_qubits']} qubits, "
              f"{len(d['edges'])} edges")
        print(f"  median 2q err = {_median_2q_err(d):.4f}, "
              f"leak_2q_default = {d['leak_2q_default']} (estimate), "
              f"median leak = {_median_leak(d)}")
        c = carried_calib(0, path=p)
        print("  carried qubit 0:", {k: round(c[k], 9) for k in
              ("T1", "T2", "p2", "gate_2q", "p_leak")})
