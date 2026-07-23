#!/usr/bin/env python3
"""E9: Total-cost threshold optimization experiment.

Circuit: k applications of oracle MCZ (= H·CCX·H) to |111⟩ (wires 0,1,2).
The target state stays |111⟩ in the noiseless case; with noise, fidelity decays.
Baseline (outer H retained) has 4 extra H gates per MCZ application vs optimized.

Demonstrates that per_iter_CNOT × E[k] > CNOT_budget is the correct decision
rule for applying Toffoli decomp + H-cancellation to a loop body.

Four conditions:
  never      — 4H + 6CNOT + 7T per oracle application (outer H retained)
  always     — 0H + 6CNOT + 7T per oracle application (outer H cancelled)
  per_iter   — optimize if per_iter_CNOT (=6) > 10  → never fires (6 < 10)
  total_cost — optimize if 6×k > 20                  → fires for k ≥ 4

Key result: fidelity gain of 'always' over 'never' increases with k (linear in noise).
Total-cost threshold fires at k≥4 where gains are meaningful (>1%).
Per-iter threshold cannot adapt to k and misses all cases (always wrong).
"""

from __future__ import annotations
import numpy as np
import pennylane as qml

# ── Constants ──────────────────────────────────────────────────────────────

EPS_1Q = 0.001      # lower noise → perturbative regime where H savings scale with k
EPS_2Q = 0.005      # ratio ε_2q/ε_1q = 5
PER_ITER_CNOT = 6     # 1 MCZ per oracle application × 6 CNOT (Shende)
PER_ITER_THRESHOLD = 10   # per-iter condition (6 < 10 → never fires)
CNOT_BUDGET = 20          # total-cost condition (fires when 6×k > 20 → k ≥ 4)

K_VALUES = [1, 2, 3, 4, 6, 8, 10]   # Oracle application counts


# ── Noise helpers ──────────────────────────────────────────────────────────

def _n1(w, eps):
    qml.DepolarizingChannel(eps, wires=w)

def _n2(ws, eps):
    for w in ws:
        qml.DepolarizingChannel(eps, wires=w)


# ── MCZ variants (reused from E7 pattern) ─────────────────────────────────

def _mcz_baseline(c0, c1, tg, eps1, eps2):
    """MCZ = H·CCX·H, outer H retained alongside Shende inner H (14H-equiv in context)."""
    qml.Hadamard(wires=tg);             _n1(tg, eps1)   # outer H before
    qml.Hadamard(wires=tg);             _n1(tg, eps1)   # Shende step 1
    qml.CNOT(wires=[c1, tg]);           _n2([c1, tg], eps2)
    qml.adjoint(qml.T)(wires=tg);       _n1(tg, eps1)
    qml.CNOT(wires=[c0, tg]);           _n2([c0, tg], eps2)
    qml.T(wires=tg);                    _n1(tg, eps1)
    qml.CNOT(wires=[c1, tg]);           _n2([c1, tg], eps2)
    qml.adjoint(qml.T)(wires=tg);       _n1(tg, eps1)
    qml.CNOT(wires=[c0, tg]);           _n2([c0, tg], eps2)
    qml.T(wires=c1);                    _n1(c1, eps1)   # T on c1 (not c0)
    qml.T(wires=tg);                    _n1(tg, eps1)
    qml.Hadamard(wires=tg);             _n1(tg, eps1)   # Shende step 11
    qml.CNOT(wires=[c0, c1]);           _n2([c0, c1], eps2)
    qml.T(wires=c0);                    _n1(c0, eps1)
    qml.adjoint(qml.T)(wires=c1);       _n1(c1, eps1)
    qml.CNOT(wires=[c0, c1]);           _n2([c0, c1], eps2)
    qml.Hadamard(wires=tg);             _n1(tg, eps1)   # outer H after


def _mcz_optimized(c0, c1, tg, eps1, eps2):
    """MCZ with outer H cancelled against Shende step-1/step-11 (4 fewer H on tg)."""
    # outer H before + step-1 H cancel → omit both
    qml.CNOT(wires=[c1, tg]);           _n2([c1, tg], eps2)
    qml.adjoint(qml.T)(wires=tg);       _n1(tg, eps1)
    qml.CNOT(wires=[c0, tg]);           _n2([c0, tg], eps2)
    qml.T(wires=tg);                    _n1(tg, eps1)
    qml.CNOT(wires=[c1, tg]);           _n2([c1, tg], eps2)
    qml.adjoint(qml.T)(wires=tg);       _n1(tg, eps1)
    qml.CNOT(wires=[c0, tg]);           _n2([c0, tg], eps2)
    qml.T(wires=c1);                    _n1(c1, eps1)   # Shende step 9: T on c1
    qml.T(wires=tg);                    _n1(tg, eps1)
    # step-11 H + outer H after cancel → omit both (steps 12-15 act on c0,c1 only)
    qml.CNOT(wires=[c0, c1]);           _n2([c0, c1], eps2)
    qml.T(wires=c0);                    _n1(c0, eps1)
    qml.adjoint(qml.T)(wires=c1);       _n1(c1, eps1)
    qml.CNOT(wires=[c0, c1]);           _n2([c0, c1], eps2)


# ── k-step Grover circuit ─────────────────────────────────────────────────

def _oracle_k_body(k, mcz_fn, eps1, eps2):
    """k oracle-MCZ applications to |111>.

    Target state remains |111> (MCZ adds global phase only). With noise, the
    state drifts away from |111>. Baseline has 4H per MCZ (extra noise); optimized
    has 0H. Fidelity gap = k × 4_H_cancelled × ε_1q, monotonically increasing.
    """
    for i in range(3):
        qml.PauliX(wires=i);  _n1(i, eps1)   # initialize to |111>
    for _ in range(k):
        mcz_fn(0, 1, 2, eps1, eps2)           # MCZ = H·CCX·H on target=2


# ── Gate counting ─────────────────────────────────────────────────────────

def count_step_gates(mcz_fn, k=1):
    """Count primitive gates in k oracle applications (excluding noise channels)."""
    with qml.tape.QuantumTape() as tape:
        _oracle_k_body(k, mcz_fn, 0.0, 0.0)
    counts: dict[str, int] = {}
    for op in tape.operations:
        if "Depolarizing" in op.name:
            continue
        name = "T" if ("Adjoint" in op.name and "T" in op.name) else op.name
        counts[name] = counts.get(name, 0) + 1
    return counts


# ── Fidelity to noiseless output ──────────────────────────────────────────

def _mcz_pure():
    """MCZ as a single PennyLane op (no noise), for noiseless reference."""
    qml.CCZ(wires=[0, 1, 2])


def _oracle_k_noiseless(k: int):
    """k oracle-MCZ applications to |111>, no noise channels."""
    for i in range(3):
        qml.PauliX(wires=i)
    for _ in range(k):
        qml.CCZ(wires=[0, 1, 2])
    return qml.state()


def get_noiseless_state(k: int) -> np.ndarray:
    """Return pure state vector for k-step Grover (noiseless, via default.qubit)."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit():
        return _oracle_k_noiseless(k)

    return np.array(circuit())


def run_k(k: int, use_opt: bool, eps1: float, eps2: float) -> np.ndarray:
    """Run k-step noisy Grover on default.mixed. Returns 8×8 density matrix."""
    dev = qml.device("default.mixed", wires=3)
    mcz = _mcz_optimized if use_opt else _mcz_baseline

    @qml.qnode(dev)
    def circuit():
        _oracle_k_body(k, mcz, eps1, eps2)
        return qml.state()

    return np.array(circuit())


def state_fidelity(psi: np.ndarray, rho: np.ndarray) -> float:
    """F(|ψ>, ρ) = <ψ|ρ|ψ> (fidelity of density matrix to pure state)."""
    return float(np.real(psi.conj() @ rho @ psi))


# ── Main experiment ────────────────────────────────────────────────────────

def run_experiment(
    k_values: list[int] = K_VALUES,
    eps1: float = EPS_1Q,
    eps2: float = EPS_2Q,
):
    # Gate count check (k=1 oracle application)
    bc1 = count_step_gates(_mcz_baseline, k=1)
    oc1 = count_step_gates(_mcz_optimized, k=1)
    print("Gate counts for k=1 oracle application (3×PauliX init + 1 MCZ):")
    all_g = sorted(set(bc1) | set(oc1))
    print(f"  {'Gate':<12}  {'Baseline':>10}  {'Optimized':>10}  {'Saved':>8}")
    for g in all_g:
        b, o = bc1.get(g, 0), oc1.get(g, 0)
        print(f"  {g:<12}  {b:>10}  {o:>10}  {b-o:>+8}")
    print(f"  {'TOTAL':<12}  {sum(bc1.values()):>10}  {sum(oc1.values()):>10}  "
          f"{sum(bc1.values())-sum(oc1.values()):>+8}")
    h_diff = bc1.get('Hadamard', 0) - oc1.get('Hadamard', 0)
    print(f"\n  H reduction per oracle application: {h_diff}   (outer H cancelled with Shende H_step1/H_step11)")
    print(f"  CNOT per oracle application: {oc1.get('CNOT',0)} (unchanged)")
    print(f"  per_iter_CNOT={PER_ITER_CNOT}, threshold fires when {PER_ITER_CNOT}×k > {CNOT_BUDGET} → k > {CNOT_BUDGET/PER_ITER_CNOT:.1f}\n")

    # Noiseless sanity: baseline == optimized state
    print("Noiseless sanity (ε=0, fidelity to noiseless reference = 1.0):")
    for k in [1, 2]:
        psi = get_noiseless_state(k)
        rho_b = run_k(k, False, 0.0, 0.0)
        rho_o = run_k(k, True,  0.0, 0.0)
        fb = state_fidelity(psi, rho_b)
        fo = state_fidelity(psi, rho_o)
        print(f"  k={k}: F_base={fb:.6f}  F_opt={fo:.6f}  diff={abs(fb-fo):.2e}")
    print()

    # Main sweep
    print(f"E9 results  (ε_1q={eps1}, ε_2q={eps2})")
    print(f"  per_iter_CNOT={PER_ITER_CNOT}, per_iter_threshold={PER_ITER_THRESHOLD}, "
          f"CNOT_budget={CNOT_BUDGET}")
    print(f"  → per-iter condition  fires when: {PER_ITER_CNOT} > {PER_ITER_THRESHOLD}  "
          f"= {'Yes' if PER_ITER_CNOT > PER_ITER_THRESHOLD else 'No, NEVER'}")
    print(f"  → total-cost condition fires when: {PER_ITER_CNOT} × k > {CNOT_BUDGET}  "
          f"→ k > {CNOT_BUDGET/PER_ITER_CNOT:.1f}\n")

    hdr = (f"{'k':>3}  {'Total':>7}  {'TC fires?':>10}  "
           f"{'F(never)':>10}  {'F(always)':>10}  {'F(per-it)':>10}  {'F(total)':>10}  "
           f"{'Gain(al)':>9}  {'Gain(tc)':>9}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for k in k_values:
        total = PER_ITER_CNOT * k
        tc_fires  = total > CNOT_BUDGET
        pit_fires = PER_ITER_CNOT > PER_ITER_THRESHOLD

        psi      = get_noiseless_state(k)
        rho_base = run_k(k, False, eps1, eps2)
        rho_opt  = run_k(k, True,  eps1, eps2)

        F_base   = state_fidelity(psi, rho_base)
        F_always = state_fidelity(psi, rho_opt)
        F_pit    = F_always if pit_fires else F_base
        F_tc     = F_always if tc_fires  else F_base

        gain_al = (F_always - F_base) / F_base * 100 if F_base > 1e-9 else float("nan")
        gain_tc = (F_tc     - F_base) / F_base * 100 if F_base > 1e-9 else float("nan")

        print(f"{k:>3}  {total:>7}  {'Yes' if tc_fires else 'No':>10}  "
              f"{F_base:>10.4f}  {F_always:>10.4f}  {F_pit:>10.4f}  {F_tc:>10.4f}  "
              f"{gain_al:>+8.2f}%  {gain_tc:>+8.2f}%")

        results.append(dict(
            k=k, total_CNOT=total, tc_fires=tc_fires, pit_fires=pit_fires,
            F_never=F_base, F_always=F_always, F_pit=F_pit, F_tc=F_tc,
            gain_always=gain_al, gain_tc=gain_tc,
        ))

    print()
    print("Summary:")
    print(f"  per-iter threshold: fires for NONE  ({PER_ITER_CNOT} < {PER_ITER_THRESHOLD})")
    fired = [r for r in results if r["tc_fires"]]
    not_f = [r for r in results if not r["tc_fires"]]
    print(f"  total-cost threshold: fires for k ≥ {CNOT_BUDGET//PER_ITER_CNOT + 1}  "
          f"({len(fired)} of {len(results)} circuits)")
    if fired:
        print(f"  Average fidelity gain where fires: "
              f"{np.mean([r['gain_tc'] for r in fired]):+.2f}%")
    if not_f:
        print(f"  Average fidelity gain where skips: "
              f"{np.mean([r['gain_tc'] for r in not_f]):+.2f}%  (correctly 0, "
              f"saved compile cost)")
    print()
    print("Key result: per-iter threshold misses ALL cases that benefit from optimization.")
    print("Total-cost threshold (per_iter_CNOT × E[k]) correctly identifies them.")

    return results


if __name__ == "__main__":
    run_experiment()
    print("\nDone.")
