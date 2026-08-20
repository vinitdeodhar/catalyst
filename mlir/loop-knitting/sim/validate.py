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


def main():
    rng = np.random.default_rng(20260818)
    ok1 = gate_i(rng) & gate_ii(rng)
    print("MILESTONE 1:", "PASS" if ok1 else "FAIL")
    ok2 = gate_m2()
    print("MILESTONE 2:", "PASS" if ok2 else "FAIL")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
