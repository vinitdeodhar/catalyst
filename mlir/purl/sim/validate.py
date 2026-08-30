"""
validate.py -- Milestone 1 validation gates for the trajectory simulator (spec 4.2).

Run:  PYTHONPATH=. python3 sim/validate.py
Exits non-zero if any gate fails.
"""

import math
import sys

import numpy as np

from sim.qsim import QSim, DEFAULT_CALIB
from benchmarks.rus_rx_ibm import run_unbounded, Z_IDEAL


def gate_i(rng, N=6000):
    """lam=0: rus_rx_ibm p_hat = 0.625 +- 0.01; <Z> = Z_IDEAL (0.5) within stat."""
    trips, zs = [], []
    for _ in range(N):
        k, z = run_unbounded(rng, lam=0.0)
        trips.append(k)
        zs.append(z)
    p_hat = 1.0 / (sum(trips) / len(trips))
    mu = float(np.mean(zs))
    se = np.std(zs, ddof=1) / math.sqrt(N)
    ok_p = abs(p_hat - 0.625) <= 0.01
    ok_z = abs(mu - Z_IDEAL) <= 3 * se
    print(f"[gate i]  p_hat={p_hat:.4f} (0.625+-0.01)  ok={ok_p}")
    print(f"[gate i]  <Z>={mu:+.4f} (ideal {Z_IDEAL:+.3f}, 3se={3*se:.3f})  ok={ok_z}")
    return ok_p and ok_z


def gate_ii(rng, N=8000):
    """Idle-only: |+> idling time T2 shows <X> ~= e^-1 +- 0.05."""
    T2 = DEFAULT_CALIB["T2"]
    xs = []
    for _ in range(N):
        s = QSim(1, lam=1.0, rng=rng)
        s.h(0)                # |+>
        s._idle_qubit(0, T2)  # idle exactly T2
        s.h(0)                # rotate X -> Z
        m = s.measure(0)
        xs.append(1 - 2 * m)
    mu = float(np.mean(xs))
    target = math.exp(-1.0)
    ok = abs(mu - target) <= 0.05
    print(f"[gate ii] idle <X>={mu:.4f} (target {target:.4f}+-0.05)  ok={ok}")
    return ok


def gate_m2(C=3, S=12000, seed=9):
    """Cross-agreement at lam=0: direct ~ discard ~ knit within 3 sigma."""
    import benchmarks.rus_rx_ibm as bench
    from sim.knit_runtime import cross_validate
    r = cross_validate(bench, C=C, lam=0.0, S=S, seed=seed, verbose=True)
    d, se_d = r["direct"]
    t, se_t, _ = r["discard"]
    k, se_k, _ = r["knit"]
    ag = lambda a, sa, b, sb: abs(a - b) <= 3 * math.sqrt(sa ** 2 + sb ** 2)
    ok = ag(t, se_t, d, se_d) and ag(k, se_k, d, se_d)
    print(f"[gate m2] discard~direct & knit~direct within 3sigma: {ok}")
    return ok


def gate_leak():
    """Leakage-as-calibration (spec 4.1/4.2): loader validation + global-median
    flatten + a leak_2q_default=0 regression (must reproduce zero-leak numbers)."""
    import os
    import tempfile
    import ibm_dataset as ds  # sim/ on the path

    ok = True
    with tempfile.TemporaryDirectory() as td:
        # leak_2q_default=0 -> p_leak flattens to 0 (zero-leak regression)
        p0 = ds.build_json(os.path.join(td, "leak0.json"), leak_2q_default=0.0)
        c0 = ds.carried_calib(0, path=p0)
        ok_zero = c0["p_leak"] == 0.0
        print(f"[gate leak] leak_2q_default=0 -> p_leak={c0['p_leak']}  ok={ok_zero}")

        # uniform default flattens to the default (global median)
        pd = ds.build_json(os.path.join(td, "leakd.json"), leak_2q_default=1.3e-3)
        cd = ds.carried_calib(0, path=pd)
        ok_dflt = abs(cd["p_leak"] - 1.3e-3) < 1e-12
        print(f"[gate leak] uniform default -> median p_leak={cd['p_leak']}  ok={ok_dflt}")

        # per-edge override respected in the global median
        d = ds.load(pd)
        for e in d["edges"]:
            e["leak_2q"] = 5e-3
        d["edges"][0].pop("leak_2q", None)  # one edge falls back to the default
        pm = os.path.join(td, "leakm.json")
        with open(pm, "w") as fh:
            import json as _json
            _json.dump(d, fh)
        cm = ds.carried_calib(0, path=pm)
        ok_over = abs(cm["p_leak"] - 5e-3) < 1e-12  # median of {default, 5e-3...} = 5e-3
        print(f"[gate leak] per-edge override -> median p_leak={cm['p_leak']}  ok={ok_over}")

        # missing leak_2q_default -> hard error
        d2 = ds.load(pd)
        d2.pop("leak_2q_default")
        try:
            ds._validate_leak(d2)
            ok_missing = False
        except ValueError:
            ok_missing = True
        print(f"[gate leak] missing leak_2q_default rejected  ok={ok_missing}")

        # out-of-range default -> hard error
        try:
            ds._validate_leak({"leak_2q_default": 1.5, "edges": []})
            ok_range = False
        except ValueError:
            ok_range = True
        print(f"[gate leak] out-of-range leak_2q_default rejected  ok={ok_range}")

    ok = ok_zero and ok_dflt and ok_over and ok_missing and ok_range
    return ok


def main():
    rng = np.random.default_rng(20260818)
    ok1 = gate_i(rng) & gate_ii(rng)
    print("MILESTONE 1:", "PASS" if ok1 else "FAIL")
    ok2 = gate_m2()
    print("MILESTONE 2:", "PASS" if ok2 else "FAIL")
    ok3 = gate_leak()
    print("LEAKAGE SCHEMA:", "PASS" if ok3 else "FAIL")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
