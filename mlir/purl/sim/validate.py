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


def gate_pump():
    """pump benchmark (spec 6.1): at lam=0 the mirror delivers <Z>=0.5 (the
    CNOT-S-CNOT sandwich is net-identity on the held wire); the per-iteration 2q
    count matches n2q=6 (parity with the pass); fixed seeds are byte-reproducible."""
    import benchmarks.pump as pump

    # lam=0: identity body -> <Z> = Z_IDEAL within statistics
    zs = []
    rng = np.random.default_rng(4242)
    N = 6000
    for _ in range(N):
        _, z = pump.run_unbounded(rng, lam=0.0)
        zs.append(z)
    mu = float(np.mean(zs))
    se = float(np.std(zs, ddof=1) / math.sqrt(N))
    ok_z = abs(mu - pump.Z_IDEAL) <= 3 * se
    print(f"[gate pump] lam=0 <Z>={mu:+.4f} (ideal {pump.Z_IDEAL}, 3se={3*se:.3f})  "
          f"ok={ok_z}")

    # n2q parity: the mirror charges exactly 6 two-qubit gates per iteration.
    calls = {"n": 0}
    sim = QSim(pump.N_WIRES, lam=1.0, rng=np.random.default_rng(1))
    orig = sim.cnot
    sim.cnot = lambda c, t: (calls.__setitem__("n", calls["n"] + 1), orig(c, t))[1]
    pump.prepare_input(sim)
    pump.attempt(sim)
    ok_n2q = calls["n"] == pump.N2Q_PER_ITER == 6
    print(f"[gate pump] per-iteration 2q gates={calls['n']} (n2q={pump.N2Q_PER_ITER})"
          f"  ok={ok_n2q}")

    # determinism: same seed -> identical trajectory
    a = [pump.run_unbounded(np.random.default_rng(7), lam=1.0) for _ in range(3)]
    b = [pump.run_unbounded(np.random.default_rng(7), lam=1.0) for _ in range(3)]
    ok_det = a == b
    print(f"[gate pump] fixed-seed determinism  ok={ok_det}")

    return ok_z and ok_n2q and ok_det


def gate_ipe_project():
    """ipe_project (spec 6.3): the knit-only projection benchmark. Checks the
    per-shot-reference machinery and the pass-relevant window math -- (a) lam=0
    delivered fidelity -> 1 for the confident config, (b) the posterior winner
    matches the collapsed data eigenstate on every noiseless shot, (c) the knit
    estimator is unbiased at lam=0, (d) an invalid forced refresh collapses the
    fidelity, and (e) the knit window is empty (faithful) / non-empty (fast)."""
    import benchmarks.ipe_project as ip
    from eval.run_eval import window
    from sim.ibm_dataset import carried_calib
    from sim.qsim import load_calib

    # (a) lam=0 confident projection -> F near 1 (F = posterior confidence -> 1).
    rng = np.random.default_rng(31)
    shots = [ip.run(rng, ip.FAITHFUL, lam=0.0)[1:] for _ in range(3000)]
    F0 = ip.delivered_fidelity([(w, z, 1.0) for w, z, _ in shots])
    ok_f0 = F0 >= 0.99
    print(f"[gate ipe_project] lam=0 faithful delivered F={F0:.4f} (>=0.99)  ok={ok_f0}")

    # (b) posterior winner == collapsed data eigenstate on every noiseless shot.
    rng = np.random.default_rng(32)
    mism = 0
    N = 2000
    for _ in range(N):
        wn, p1 = ip.run_debug(rng, ip.FAITHFUL, lam=0.0)
        if (1 if p1 > 0.5 else 0) != wn:
            mism += 1
    ok_match = mism == 0
    print(f"[gate ipe_project] winner==collapsed eigenstate mismatches={mism}/{N}  "
          f"ok={ok_match}")

    # (c) knit estimator unbiased at lam=0 (fast, C=6 in-window): F_knit ~ F_unb.
    rng = np.random.default_rng(33)
    ub = [ip.run(rng, ip.FAST, lam=0.0, max_rounds=10)[1:] for _ in range(6000)]
    kn = [ip.run(rng, ip.FAST, lam=0.0, C=6, max_rounds=10) for _ in range(6000)]
    Fu = ip.delivered_fidelity([(w, z, 1.0) for w, z, _ in ub])
    Fk = ip.delivered_fidelity([(w, z, wt) for _, w, z, wt in kn])
    ok_knit = abs(Fu - Fk) < 0.08
    print(f"[gate ipe_project] knit unbiased lam=0 F_unb={Fu:.3f} F_knit={Fk:.3f} "
          f"|diff|={abs(Fu-Fk):.3f} (<0.08)  ok={ok_knit}")

    # (d) forced refresh (INVALID here) collapses the delivered fidelity.
    rng = np.random.default_rng(34)
    val = [ip.run(rng, ip.FAITHFUL, lam=1.0)[1:] for _ in range(3000)]
    fr = [ip.run_forced_refresh(rng, ip.FAITHFUL, lam=1.0, C=2)[1:] for _ in range(3000)]
    Fv = ip.delivered_fidelity([(w, z, 1.0) for w, z, _ in val])
    Ff = ip.delivered_fidelity([(w, z, 1.0) for w, z, _ in fr])
    ok_fals = Ff < Fv - 0.08
    print(f"[gate ipe_project] falsification F_valid={Fv:.3f} -> F_forced_refresh="
          f"{Ff:.3f} (collapse)  ok={ok_fals}")

    # (e) window math on the IBM calib: faithful empty, fast non-empty.
    cal = load_calib(carried_calib(0))
    wf = window(ip.FAITHFUL.p_nominal, f=ip.FAITHFUL.f, calib=cal)
    ws = window(ip.FAST.p_nominal, f=ip.FAST.f, calib=cal)
    ok_win = wf[0] > wf[1] and ws[0] <= ws[1]
    print(f"[gate ipe_project] window faithful={wf} (empty) fast={ws} (non-empty)  "
          f"ok={ok_win}")

    return ok_f0 and ok_match and ok_knit and ok_fals and ok_win


def gate_rus_data():
    """rus_data (spec 6.5): Paetznick-Svore V3 RUS on program data. FLAGGED best-
    effort reconstruction -- these gates pin the V3 CHANNEL (not the paper's exact
    gate circuit): (a) transcription -- the Kraus operators are M0=sqrt(5/8)V3 and
    M1=sqrt(3/8)I (success = V3, failure = identity); (b) probability -- measured
    success rate = 5/8; (c) fidelity -- lam=0 delivered fidelity 1.0 vs the fixed
    ideal V3|psi>; (d) parity -- the pass n2q matches the mirror's per-iteration 2q
    charging."""
    import benchmarks.rus_data as rd

    # (a) transcription: Kraus M0 = sqrt(5/8) V3, M1 = sqrt(3/8) I (channel-exact)
    M0 = np.diag([rd.U0[0, 0], rd.U1[0, 0]])
    M1 = np.diag([rd.U0[1, 0], rd.U1[1, 0]])
    ok_m0 = np.allclose(M0, math.sqrt(5 / 8) * rd.V3)
    ok_m1 = np.allclose(M1, math.sqrt(3 / 8) * np.eye(2))
    print(f"[gate rus_data] Kraus M0==sqrt(5/8)V3 & M1==sqrt(3/8)I  "
          f"ok={ok_m0 and ok_m1}")

    # (b) probability: measured lam=0 success rate == 5/8
    rng = np.random.default_rng(55)
    trips = [rd.run_unbounded(rng, lam=0.0)[0] for _ in range(8000)]
    p_hat = 1.0 / (sum(trips) / len(trips))
    ok_p = abs(p_hat - rd.P_ANALYTIC) < 0.02
    print(f"[gate rus_data] success p_hat={p_hat:.4f} (published {rd.P_ANALYTIC})  "
          f"ok={ok_p}")

    # (c) fidelity: lam=0 delivered 3-basis fidelity == 1.0 vs the FIXED ideal V3|psi>
    def read(sim, q, basis):
        if basis == "X":
            sim.h(q)
        elif basis == "Y":
            sim.sdg(q); sim.h(q)
        return 1 - 2 * sim.measure(q)
    comp = {}
    for basis in "XYZ":
        vals = []
        for i in range(12000):
            s = QSim(rd.N_WIRES, lam=0.0, rng=np.random.default_rng(20000 + i))
            rd.prepare_input(s)
            k, fail = 0, True
            while fail and k < 200:
                fail = rd.attempt(s)
                k += 1
            vals.append(read(s, rd.DATA, basis))
        comp[basis] = float(np.mean(vals))
    a = np.array([comp["X"], comp["Y"], comp["Z"]])
    F = 0.5 * (1.0 + float(a @ rd.IDEAL_BLOCH))
    ok_fid = F >= 0.99
    print(f"[gate rus_data] lam=0 delivered fidelity vs V3|psi> = {F:.4f} (>=0.99)  "
          f"ok={ok_fid}")

    # (d) parity: the mirror charges exactly N2Q_PER_ITER 2q gates per attempt
    calls = {"n": 0}
    s = QSim(rd.N_WIRES, lam=1.0, rng=np.random.default_rng(1))
    orig = s.ctrl_branch
    s.ctrl_branch = lambda c, t, U0, U1: (calls.__setitem__("n", calls["n"] + 1),
                                          orig(c, t, U0, U1))[1]
    rd.prepare_input(s)
    rd.attempt(s)
    ok_n2q = calls["n"] == rd.N2Q_PER_ITER == 1
    print(f"[gate rus_data] per-iteration 2q gates={calls['n']} "
          f"(n2q={rd.N2Q_PER_ITER})  ok={ok_n2q}")

    return ok_m0 and ok_m1 and ok_p and ok_fid and ok_n2q


def gate_migrate():
    """Migrate strategy simulator semantics (spec 13.5): (a) ping-pong -- forced
    migration at C=1 alternates the physical carrier k,k',k,k'; (b) leaked-transfer
    (decision 4) -- migrating a LEAKED carrier yields a live wire in |0>, unleaked,
    with the state NOT transferred; (c) state transfer -- migrating an UNleaked
    carrier moves the state intact; (d) cost-positivity threshold matches spec 13.2."""
    LIVE, PARTNER = 0, 1

    # (a) ping-pong: migrate at C=1 alternates the state-bearing physical wire.
    s = QSim(2, lam=0.0, rng=np.random.default_rng(1))
    s.h(LIVE)                              # some state on the live wire
    trace, live = [], LIVE
    for _ in range(4):
        partner = 1 - live                # the other physical qubit is the fresh one
        live = s.migrate(live, partner)   # state moves onto `partner`
        trace.append(live)
    ok_pp = trace == [1, 0, 1, 0]
    print(f"[gate migrate] ping-pong carrier trace = {trace} (k',k,k',k)  ok={ok_pp}")

    # (b) leaked-transfer (decision 4): SWAP from a leaked carrier is a no-op, so the
    # new live wire is a clean |0> and the state does not transfer.
    s = QSim(2, lam=1.0, rng=np.random.default_rng(2))
    s.h(LIVE)
    s.leaked[LIVE] = True                  # force the carrier leaked
    live = s.migrate(LIVE, PARTNER)
    p1 = s._prob_one(live)
    ok_leaked = (live == PARTNER and not s.leaked[live] and not s.leaked[LIVE]
                 and abs(p1) < 1e-9)       # new live is |0> (state lost, as intended)
    print(f"[gate migrate] leaked-carrier migrate -> live |0> P(1)={p1:.3g}, "
          f"unleaked  ok={ok_leaked}")

    # (c) state transfer: migrating an UNleaked carrier moves the state intact.
    s = QSim(2, lam=0.0, rng=np.random.default_rng(3))
    s.h(LIVE); s.t(LIVE)                   # |+> then T -> a generic 1q state
    before = s._prob_one(LIVE)
    live = s.migrate(LIVE, PARTNER)
    after = s._prob_one(live)
    ok_xfer = abs(before - after) < 1e-9 and live == PARTNER
    print(f"[gate migrate] unleaked migrate transfers state P(1) {before:.4f}->"
          f"{after:.4f}  ok={ok_xfer}")

    # (d) cost-positivity threshold (spec 13.2): C*n2q*p_leak > 3*(g_e + l_e).
    def positive(C, n2q, p_leak, g_e, l_e):
        eps_mig = 1 - (1 - g_e) ** 3 * (1 - l_e) ** 3
        return C * n2q * p_leak > eps_mig
    # cheap pair edge + high body leakage -> positive; expensive edge -> negative
    ok_pos = positive(4, 6, 1e-2, 1e-4, 1e-4) and not positive(1, 1, 1e-3, 8e-3, 2e-2)
    print(f"[gate migrate] cost-positive on the cheap side, negative on the "
          f"expensive side  ok={ok_pos}")

    return ok_pp and ok_leaked and ok_xfer and ok_pos


def gate_qwalk():
    """qwalk (spec 13 migrate benchmark): fat-tailed, 2q-heavy, non-Clifford,
    net-identity held reference. (a) FAT TAIL -- median steps ~1 but P(T>1)~0.5 and
    a heavy upper tail (so >~50% of shots cross a C=1 cut); (b) net-identity -- lam=0
    delivered fidelity 1.0 vs the FIXED held |psi>; (c) parity -- n2q = 6 per step."""
    import benchmarks.qwalk as qw

    # (a) fat tail: median ~1, but a large fraction runs long (unlike geometric 5/8)
    rng = np.random.default_rng(7)
    T = np.array([qw.run_unbounded(rng, lam=0.0)[0] for _ in range(8000)])
    # fat tail: a small median but a heavy upper tail (P(T>1)~0.5, deep max) -- the
    # signature that separates it from the thin geometric p=5/8 benchmarks.
    ok_fat = np.median(T) <= 5 and np.mean(T > 1) > 0.4 and T.max() > 30
    print(f"[gate qwalk] fat tail: median={np.median(T):.0f} P(T>1)={np.mean(T>1):.3f}"
          f" max={T.max()}  ok={ok_fat}")

    # (b) net-identity: lam=0 delivered 3-basis fidelity vs the fixed held |psi>
    def read(sim, q, basis):
        if basis == "X":
            sim.h(q)
        elif basis == "Y":
            sim.sdg(q); sim.h(q)
        return 1 - 2 * sim.measure(q)
    comp = {}
    for basis in "XYZ":
        vals = []
        for i in range(9000):
            s = QSim(qw.N_WIRES, lam=0.0, rng=np.random.default_rng(30000 + i))
            qw.prepare_input(s)
            pos, k = 1, 0
            while pos != 0 and k < 500:
                qw.touch(s)
                pos = qw.walk_step(np.random.default_rng(77000 + i + k), pos)
                k += 1
            vals.append(read(s, qw.DATA, basis))
        comp[basis] = float(np.mean(vals))
    a = np.array([comp["X"], comp["Y"], comp["Z"]])
    F = 0.5 * (1.0 + float(a @ qw.IDEAL_BLOCH))
    ok_fid = F >= 0.99
    print(f"[gate qwalk] lam=0 delivered fidelity vs held |psi> = {F:.4f} (>=0.99)  "
          f"ok={ok_fid}")

    # (c) parity: exactly N2Q_PER_ITER two-qubit gates per step
    calls = {"n": 0}
    s = QSim(qw.N_WIRES, lam=1.0, rng=np.random.default_rng(1))
    orig = s.cnot
    s.cnot = lambda c, t: (calls.__setitem__("n", calls["n"] + 1), orig(c, t))[1]
    qw.prepare_input(s)
    qw.touch(s)
    ok_n2q = calls["n"] == qw.N2Q_PER_ITER == 6
    print(f"[gate qwalk] per-step 2q gates={calls['n']} (n2q={qw.N2Q_PER_ITER})  "
          f"ok={ok_n2q}")

    return ok_fat and ok_fid and ok_n2q


def main():
    rng = np.random.default_rng(20260818)
    ok1 = gate_i(rng) & gate_ii(rng)
    print("MILESTONE 1:", "PASS" if ok1 else "FAIL")
    ok2 = gate_m2()
    print("MILESTONE 2:", "PASS" if ok2 else "FAIL")
    ok3 = gate_leak()
    print("LEAKAGE SCHEMA:", "PASS" if ok3 else "FAIL")
    ok4 = gate_pump()
    print("PUMP BENCHMARK:", "PASS" if ok4 else "FAIL")
    ok5 = gate_ipe_project()
    print("IPE_PROJECT BENCHMARK:", "PASS" if ok5 else "FAIL")
    ok6 = gate_rus_data()
    print("RUS_DATA BENCHMARK:", "PASS" if ok6 else "FAIL")
    ok7 = gate_migrate()
    print("MIGRATE STRATEGY:", "PASS" if ok7 else "FAIL")
    ok8 = gate_qwalk()
    print("QWALK BENCHMARK:", "PASS" if ok8 else "FAIL")
    sys.exit(0 if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8)
             else 1)


if __name__ == "__main__":
    main()
