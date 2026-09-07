"""
eval/run_all.py -- the eval suite. Compiles each benchmark @qjit program (both Purl
passes in the pipeline) and runs it on the noisy QSimDevice, uniformly: ONE path for
every benchmark -- compile, then average the returned expval over trajectories. The
passes select and lower the cut; the device applies the calibrated noise. No
per-benchmark execution logic.

  PYTHONPATH=. python3 eval/run_all.py                     # prompts for hardware
  PYTHONPATH=. python3 eval/run_all.py --hardware heron    # or eagle
  PYTHONPATH=. python3 eval/run_all.py --hardware eagle --fast

Requires the qsim runtime device built (make runtime) and the Catalyst frontend, and
the chosen hardware JSON generated (sim/ibm_dataset.py writes eagle + heron).
"""

import math
import os
import sys

import numpy as np

from eval.programs import PROGRAMS, probe_strategy
from sim.ibm_dataset import HARDWARE, build_hardware

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
LAMS = (0.0, 1.0, 4.0)
ARMS = ("purl", "unbounded")


def pick_hardware(argv):
    """Resolve the target hardware: --hardware {eagle,heron}; else prompt (or default
    eagle when non-interactive). Returns (name, calib_json_path)."""
    name = None
    for i, a in enumerate(argv):
        if a == "--hardware" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--hardware="):
            name = a.split("=", 1)[1]
    if name is None:
        if sys.stdin.isatty():
            name = input(f"Which hardware? {list(HARDWARE)} [eagle]: ").strip() or "eagle"
        else:
            name = "eagle"
    if name not in HARDWARE:
        raise SystemExit(f"unknown hardware {name!r}; choose from {list(HARDWARE)}")
    _, _, path = HARDWARE[name]
    if not os.path.exists(path):
        build_hardware(name)          # generate the JSON on first use
    return name, path


def run(name, builder, lams, seeds, shots, calib):
    """Two arms (purl / unbounded) across noise scales (§14.7). Each (arm, lam, seed)
    is a separate compile; the expval is averaged over `shots` one-shot trajectories
    and seeds give the error bar, while the per-shot trip counter (returned alongside
    the expval) gives the runtime iteration depth. `unbounded` omits both passes.
    Returns rows: (lam, ideal, {arm: (mean, std)}, (iters_min, iters_mean, iters_max))."""
    rows = []
    for lam in lams:
        vals = {a: [] for a in ARMS}
        iters, ideal = [], None
        for arm, on in (("purl", True), ("unbounded", False)):
            for sd in range(seeds):
                f, ideal = builder(lam, seed=sd, shots=shots, calib=calib, purl_on=on)
                ev, k = f()
                vals[arm].append(float(ev))
                if arm == "unbounded":            # runtime depth = unbounded arm (§8.2)
                    iters.append(np.asarray(k).ravel())
        it = np.concatenate(iters) if iters else np.array([0])
        stats = {a: (float(np.mean(v)),
                     float(np.std(v, ddof=1)) if seeds > 1 else 0.0)
                 for a, v in vals.items()}
        rows.append((lam, ideal, stats,
                     (int(it.min()), float(it.mean()), int(it.max()))))
    return rows


def main(argv=()):
    fast = "--fast" in argv
    hw, calib = pick_hardware(argv)
    seeds, shots = (3, 400) if fast else (6, 1500)
    lines = []
    csv = ["benchmark,lam,arm,mean,std,ideal,infidelity,rmse,"
           "iters_min,iters_mean,iters_max,strategy,applied,C,bounded_cap,"
           "pass_predicted_F,best_arm,regret"]

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    emit(f"=== Purl eval suite ({hw}): compiled @qjit, two arms (purl vs unbounded) "
         f"on qsim device ===")
    emit(f"calib={os.path.basename(calib)}; one-shot trajectory averaging, "
         f"shots={shots}, seeds={seeds}; noise lam in {LAMS}")
    emit("legend (§14.7): infid=|mean-ideal|; RMSE=sqrt(infid^2+std^2); "
         "iters=unbounded runtime depth (min/mean/max);")
    emit("  bounded_cap=C=coherent-depth cap the cut guarantees (purl arm, vs the "
         "unbounded mean); best_arm=min-RMSE; regret=RMSE(purl)-RMSE(best_arm).")

    for name, builder in PROGRAMS.items():
        # §14.7: report the PASS-selected strategy (its own in-pipeline decision)
        dec = probe_strategy(builder, calib)
        C = dec["C"]
        cap = str(C) if C is not None else "n/a"
        predf = (f"{dec['predicted_bounded']:.4f}"
                 if dec["predicted_bounded"] is not None else "n/a")
        emit(f"\n# {name}  ->  strategy: {dec['strategy']}"
             f"{' (applied)' if dec['applied'] else ' (not applied)'}"
             f", C={cap}, bounded_cap={cap} iter, pass-predicted F(bounded)={predf}")
        emit(f"  {'lam':>4}  {'arm':<9} {'<O>(mean±std)':>17}  {'ideal':>6} "
             f"{'infid':>7} {'RMSE':>7}  {'iters(min/mean/max)':>19}  "
             f"{'best':>9} {'regret':>7}")
        for lam, ideal, stats, itr in run(name, builder, LAMS, seeds, shots, calib):
            m = {}
            for a in ARMS:
                mean, std = stats[a]
                infid = abs(mean - ideal)
                m[a] = (mean, std, infid, math.hypot(infid, std))   # (.., .., infid, rmse)
            best = min(ARMS, key=lambda a: m[a][3])
            regret = m["purl"][3] - m[best][3]
            it_str = f"{itr[0]}/{itr[1]:.1f}/{itr[2]}"
            for a in ARMS:
                mean, std, infid, rmse = m[a]
                tag = best if a == "purl" else ""
                reg = f"{regret:.4f}" if a == "purl" else ""
                emit(f"  {lam:4.2f}  {a:<9} {mean:8.4f} ± {std:.4f}  {ideal:6.3f} "
                     f"{infid:7.4f} {rmse:7.4f}  {it_str:>19}  {tag:>9} {reg:>7}")
                csv.append(f"{name},{lam},{a},{mean:.6f},{std:.6f},{ideal:.6f},"
                           f"{infid:.6f},{rmse:.6f},{itr[0]},{itr[1]:.3f},{itr[2]},"
                           f"{dec['strategy']},{dec['applied']},{cap},{cap},{predf},"
                           f"{best},{regret:.6f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"suite_{hw}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS, f"suite_{hw}.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
