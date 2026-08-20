"""
knit_runtime.py -- the inline cut protocol, weights, and estimators (spec 4.3).

Three per-shot executors over a carry-type benchmark, all against sim/qsim.py:

  run_unbounded(bench, rng, ...)        -> (z_sample, k)                arm 1
  run_truncated(bench, C, rng, ...)     -> (z_sample|None, k, trunc)   arm 2
  run_knit(bench, C, rng, ...)          -> (estimator, k, ncut, trunc) arm 3

A `bench` is any object exposing:
    N_WIRES : int
    TARGET  : int                       # the carried (never-measured) wire
    prepare_input(sim)                  # program input state on TARGET
    attempt(sim) -> bool                # one loop body; returns fail flag

The knit executor implements steps 3.4a-f INLINE against a single simulator:
every C failing iterations the carried wire is cut (measure in a sampled Pauli
basis, reset, re-prepare a sampled eigenstate) and the shot weight is multiplied
by the signed term weight 4*sigma*s. The final estimator sample is w * (-1)^z.
Because the cut re-initializes the wire, the carried qubit's *coherent depth*
never exceeds C bodies -- the physical point of the transform.
"""

import numpy as np

from sim.qsim import QSim

# term index encoding for the single-wire identity decomposition:
#   O in {0=I, 1=X, 2=Y, 3=Z};  t in {0,1} eigenstate index.  Uniform over 8.


def sample_term(rng):
    """Uniformly sample one of the 8 (O, t) quasi-probability terms."""
    O = int(rng.integers(4))
    t = int(rng.integers(2))
    return O, t


def cut_and_reprepare(sim, q, rng):
    """Inline cut on wire q (spec 3.4a-f). Returns the signed term weight.

    Steps: sample (O,t); rotate q into the O basis; measure (eigenvalue s);
    reset q to |0>; prepare the sampled eigenstate |O,t>; return 4*sigma*s_eff.
    The carried wire's coherent history ends at the measurement and a fresh
    error-free segment begins at the eigenstate preparation.
    """
    O, t = sample_term(rng)

    # (b) basis change guarded by O: I/Z none; X -> H; Y -> Sdg then H
    if O == 1:            # X
        sim.h(q)
    elif O == 2:          # Y
        sim.sdg(q)
        sim.h(q)

    # (c) measure -> outcome s
    s_bit = sim.measure(q)
    sim.feedback(active=[q])          # feedback latency before reset/prep
    s_val = 1.0 - 2.0 * s_bit         # +1 / -1

    # (d) continue on a FRESH qubit: |0>, leakage-free (this is the physical
    #     mechanism by which cutting reduces error -- see NOTES.md)
    sim.force_zero(q)

    # (e) prepare eigenstate |O,t>
    if t:
        sim.x(q)
    if O in (1, 2):                   # X or Y basis
        sim.h(q)
    if O == 2:                        # Y
        sim.s(q)

    # (f) signed term weight
    s_eff = 1.0 if O == 0 else s_val          # forced +1 for O = I
    sigma = -1.0 if (t and O != 0) else 1.0   # eigenvalue sign of |O,t>
    wterm = 4.0 * sigma * s_eff
    return wterm


# The three arms are N-stage chains (N=1 = the single RUS loop). A "stage" is one
# repeat-until-success loop holding the SAME carried qubit; N sequential stages
# thread that qubit through N gates (the rus_chain benchmark). Each measurement
# below optionally reads the carried qubit in a rotated basis for the secondary
# Bloch-fidelity metric: basis in {"Z","X","Y"}.

_BASIS_ROT = {"Z": None, "X": "h", "Y": ("sdg", "h")}


def _read(sim, q, basis):
    """Rotate q into `basis` and Z-measure; return +-1 eigenvalue sample."""
    rot = _BASIS_ROT[basis]
    if rot == "h":
        sim.h(q)
    elif rot is not None:
        sim.sdg(q)
        sim.h(q)
    return 1 - 2 * sim.measure(q)


# ---------------------------------------------------------------------------
# Arm 1: UNBOUNDED (input program as-is)
# ---------------------------------------------------------------------------
def run_unbounded(bench, rng, calib=None, lam=1.0, N=1, basis="Z", max_trips=500):
    sim = QSim(bench.N_WIRES, calib=calib, lam=lam, rng=rng)
    bench.prepare_input(sim)
    ktot = 0
    for _ in range(N):
        k, fail = 0, True
        while fail and k < max_trips:
            fail = bench.attempt(sim)
            k += 1
        ktot += k
    return _read(sim, bench.TARGET, basis), ktot


# ---------------------------------------------------------------------------
# Arm 2: TRUNCATE + DISCARD (bound C per stage, drop shots that truncate)
# ---------------------------------------------------------------------------
def run_truncated(bench, C, rng, calib=None, lam=1.0, N=1, basis="Z"):
    sim = QSim(bench.N_WIRES, calib=calib, lam=lam, rng=rng)
    bench.prepare_input(sim)
    ktot = 0
    for _ in range(N):
        k, fail = 0, True
        while fail and k < C:
            fail = bench.attempt(sim)
            k += 1
        ktot += k
        if fail:
            return None, ktot, True   # any stage truncates -> discard the shot
    return _read(sim, bench.TARGET, basis), ktot, False


# ---------------------------------------------------------------------------
# Arm 3: KNIT (cut every C failing iterations, inline; per stage)
# ---------------------------------------------------------------------------
def run_knit(bench, C, rng, calib=None, lam=1.0, N=1, basis="Z", max_trips=500):
    sim = QSim(bench.N_WIRES, calib=calib, lam=lam, rng=rng)
    bench.prepare_input(sim)
    w = 1.0
    ktot, ncut = 0, 0
    for _ in range(N):
        k, fail = 0, True
        while fail and k < max_trips:
            fail = bench.attempt(sim)
            k += 1
            if fail and (k % C == 0):
                w *= cut_and_reprepare(sim, bench.TARGET, rng)
                ncut += 1
        ktot += k
    est = w * _read(sim, bench.TARGET, basis)
    return est, ktot, ncut, w


# ---------------------------------------------------------------------------
# Cross-validation harness (Milestone 2): three estimators must agree at lam=0
# ---------------------------------------------------------------------------
def _mean_se(xs):
    n = len(xs)
    mu = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mu, se


def cross_validate(bench, C=2, lam=0.0, S=8000, seed=1, verbose=True):
    """Return dict of the three estimators' <Z> +- se and diagnostics."""
    rng = np.random.default_rng(seed)

    # arm 1: direct
    d = [run_unbounded(bench, rng, lam=lam)[0] for _ in range(S)]
    mu_d, se_d = _mean_se(d)

    # arm 2: truncate + discard
    kept, ntr = [], 0
    for _ in range(S):
        z, _, trunc = run_truncated(bench, C, rng, lam=lam)
        if trunc:
            ntr += 1
        else:
            kept.append(z)
    mu_t, se_t = _mean_se(kept)

    # arm 3: knit
    ests, ncut = [], 0
    for _ in range(S):
        e, _, nc, _ = run_knit(bench, C, rng, lam=lam)
        ncut += (nc > 0)
        ests.append(e)
    mu_k, se_k = _mean_se(ests)

    var_inflation = (se_k ** 2 * len(ests)) / (se_d ** 2 * len(d))
    res = {
        "direct": (mu_d, se_d),
        "discard": (mu_t, se_t, ntr / S),
        "knit": (mu_k, se_k, ncut / S),
        "var_inflation": var_inflation,
        "C": C, "lam": lam, "S": S,
    }
    if verbose:
        print(f"[{bench.__name__ if hasattr(bench,'__name__') else bench} "
              f"C={C} lam={lam} S={S}]")
        print(f"  direct  <Z> = {mu_d:+.4f} +- {se_d:.4f}")
        print(f"  discard <Z> = {mu_t:+.4f} +- {se_t:.4f}  "
              f"discard_rate={ntr/S:.4f}")
        print(f"  knit    <Z> = {mu_k:+.4f} +- {se_k:.4f}  "
              f"cut_rate={ncut/S:.4f}")
        print(f"  var_inflation knit/direct = {var_inflation:.2f}")
    return res
