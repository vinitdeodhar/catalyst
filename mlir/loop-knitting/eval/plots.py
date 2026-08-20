"""
plots.py -- figures from results/eval.csv (spec 6.5).

  crossover.png  (sweep 1)  E vs lam, unbounded vs knit
  c_window.png   (sweep 2)  E vs C, with pass-computed [C_min,C_max] marked
  chain.png      (sweep 3)  E vs N, unbounded vs knit
  tail.png       (sweep 4)  per-shot error vs realized trip count k

Every figure: mean line + shaded +-1 std over seeds, arms labelled.

Usage:  PYTHONPATH=. python3 eval/plots.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, os.pardir, "results")
ARMS = ["unbounded", "knit"]
COLORS = {"unbounded": "#d62728", "knit": "#2ca02c"}


def load():
    rows, meta = [], {}
    with open(os.path.join(RESULTS, "eval.csv")) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if not row:
                continue
            if row[0] == "_meta":
                meta["C_min"], meta["C_max"], meta["C"] = int(row[3]), int(row[4]), int(row[7])
                continue
            b, arm, lam, C, N, seed, metric, val = row
            rows.append(dict(bench=b, arm=arm, lam=float(lam), C=int(C),
                             N=int(N), seed=int(seed), metric=metric,
                             value=float(val)))
    return rows, meta


def _agg(rows, key, metric, filt):
    """group value by key(row) over seeds -> {k: (mean,std)}."""
    d = defaultdict(list)
    for r in rows:
        if r["metric"] == metric and filt(r):
            d[key(r)].append(r["value"])
    return {k: (np.mean(v), np.std(v)) for k, v in d.items() if v}


def crossover(rows, meta):
    C = meta["C"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        g = _agg(rows, lambda r: r["lam"], "E",
                 lambda r: r["bench"] == "rus_rx_ibm" and r["arm"] == arm
                 and r["N"] == 1 and r["C"] == C and r["seed"] >= 0)
        xs = sorted(g)
        m = np.array([g[x][0] for x in xs])
        s = np.array([g[x][1] for x in xs])
        ax.plot(xs, m, "-o", color=COLORS[arm], label=arm)
        ax.fill_between(xs, m - s, m + s, color=COLORS[arm], alpha=0.2)
    ax.set_xlabel("noise scale  λ")
    ax.set_ylabel(r"expectation error  $E=|\langle Z\rangle_{est}-\langle Z\rangle_{ideal}|$")
    ax.set_title(f"Sweep 1: crossover (rus_rx_ibm, C={C}, leakage-dominated)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "crossover.png")


def c_window(rows, meta):
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        g = _agg(rows, lambda r: r["C"], "E",
                 lambda r: r["bench"] == "rus_rx_ibm" and r["arm"] == arm
                 and r["N"] == 1 and abs(r["lam"] - 1.0) < 1e-9 and r["seed"] >= 0)
        xs = sorted(g)
        if not xs:
            continue
        m = np.array([g[x][0] for x in xs])
        s = np.array([g[x][1] for x in xs])
        ax.plot(xs, m, "-o", color=COLORS[arm], label=arm)
        ax.fill_between(xs, m - s, m + s, color=COLORS[arm], alpha=0.2)
    ax.axvspan(meta["C_min"], meta["C_max"], color="gray", alpha=0.15,
               label=f"window [{meta['C_min']},{meta['C_max']}]")
    ax.axvline(meta["C_min"], color="k", ls="--", lw=0.8)
    ax.axvline(meta["C_max"], color="k", ls="--", lw=0.8)
    ax.set_xlabel("cut period  C")
    ax.set_ylabel(r"expectation error $E$")
    ax.set_title("Sweep 2: cut-period window (λ=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "c_window.png")


def chain(rows, meta):
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ("unbounded", "knit"):
        g = _agg(rows, lambda r: r["N"], "E",
                 lambda r: r["bench"] == "rus_chain" and r["arm"] == arm
                 and r["seed"] >= 0)
        xs = sorted(g)
        m = np.array([g[x][0] for x in xs])
        s = np.array([g[x][1] for x in xs])
        ax.plot(xs, m, "-o", color=COLORS[arm], label=arm)
        ax.fill_between(xs, m - s, m + s, color=COLORS[arm], alpha=0.2)
    ax.set_xlabel("chain length  N")
    ax.set_ylabel(r"expectation error $E$")
    ax.set_title("Sweep 3: chain (λ=1, unbounded vs knit)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "chain.png")


def tail(rows, meta):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lam in sorted({r["lam"] for r in rows if r["metric"] == "err_at_k"}):
        pts = [(r["N"], r["value"]) for r in rows
               if r["metric"] == "err_at_k" and abs(r["lam"] - lam) < 1e-9]
        pts.sort()
        ks = [k for k, _ in pts]
        es = [e for _, e in pts]
        ax.plot(ks, es, "-o", label=f"λ={lam:g}")
    ax.set_xlabel("realized trip count  k")
    ax.set_ylabel(r"per-shot delivered error $|\langle Z\rangle_k-$ideal$|$")
    ax.set_title("Sweep 4: tail shots are the corrupted ones (unbounded)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tail.png")


def _save(fig, name):
    fig.tight_layout()
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    rows, meta = load()
    crossover(rows, meta)
    c_window(rows, meta)
    chain(rows, meta)
    tail(rows, meta)


if __name__ == "__main__":
    main()
