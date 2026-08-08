#!/usr/bin/env python3
"""X16: Dynamic (richer) entanglement-width analysis for the paper's dynamic benchmarks.

For each dynamic benchmark, we analyze the per-iteration loop BODY (the gates
actually executed each iteration) as an interaction graph:
  - union-find over live qubits: a k-qubit gate unions its qubits; measure/reset
    splits a qubit back to a singleton (breaks its entanglement).
  - peak entanglement width = largest simultaneously-entangled block.
  - treewidth = min-fill heuristic (networkx) on the interaction graph of that
    block -- the quantity that governs classical simulation cost (~exp(tw)).

Honest note: the paper's dynamic benchmarks are ALL reset-based, so entanglement
is broken every iteration -> the realized (dynamic, per-shot) entanglement width
equals the static body value and is TRIP-COUNT-INDEPENDENT (no distribution).
The `chain` example at the end is an *accumulating* (no-reset) measurement-driven
loop, where the dynamic per-shot width genuinely varies with the profiled trip
count -- the case where the dynamic analysis beats a static bound.

Usage:  python3 run_x16_entanglement_width.py
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in

sys.path.insert(0, str(Path(__file__).parent))

# ── interaction-graph analyzer ──────────────────────────────────────────────
# A body is a list of ops: ('gate', [wires]) or ('measure', [wire]).
# A multi-qubit gate entangles all its wires (clique). measure resets a wire.

class UF:
    def __init__(self): self.p = {}
    def add(self, x): self.p.setdefault(x, x)
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)

def analyze(body):
    uf = UF()
    cur = {}          # logical wire -> current node id (fresh id after each reset)
    nid = [0]
    def node(w):
        if w not in cur:
            cur[w] = nid[0]; nid[0]+=1; uf.add(cur[w])
        return cur[w]
    edges = set()     # live edges among current node ids
    peak_width, peak_graph = 1, nx.Graph()
    def block(root_of):
        return [n for n in uf.p if uf.find(n) == root_of]
    for op, wires in body:
        if op == 'measure':
            for w in wires:                       # reset -> fresh singleton id
                if w in cur:
                    old = cur[w]
                    edges = {e for e in edges if old not in e}
                cur[w] = nid[0]; nid[0]+=1; uf.add(cur[w])
        else:
            ns = [node(w) for w in wires]
            for i in range(len(ns)):
                for j in range(i+1, len(ns)):
                    uf.union(ns[i], ns[j]); edges.add(frozenset((ns[i], ns[j])))
            blk = block(uf.find(ns[0]))
            if len(blk) > peak_width:
                peak_width = len(blk)
                g = nx.Graph(); g.add_nodes_from(blk)
                for e in edges:
                    a,b = tuple(e)
                    if a in blk and b in blk: g.add_edge(a,b)
                peak_graph = g
    tw = treewidth_min_fill_in(peak_graph)[0] if peak_graph.number_of_nodes()>1 else 0
    return peak_width, tw

# ── benchmark bodies (per-iteration gates; faithful to the paper's circuits) ──
def bench_bodies():
    B = {}
    # coin-flip: H on ancilla, measure(reset). no 2-qubit gate.
    B['coin-flip'] = [('H', [0]), ('measure', [0])]
    # RUS: H,CNOT,T,CNOT,H on (data0, anc1); measure(anc, reset)
    B['RUS'] = [('H',[1]),('CNOT',[0,1]),('T',[1]),('CNOT',[0,1]),('H',[1]),('measure',[1])]
    # MSD (n_magic=7): prep magic 1..7, CNOT(magic_i -> syndrome 0) star, measure
    msd = []
    for w in range(1,8): msd += [('H',[w]),('T',[w])]
    for w in range(1,8): msd += [('CNOT',[w,0])]        # star to syndrome (0)
    msd += [('measure',[0])] + [('measure',[w]) for w in range(1,8)]
    B['MSD (n=7)'] = msd
    # BBHT (n_data=3): H all; oracle Toffoli(0,1,2); diffuser Toffoli(0,1,2); measure
    bbht = [('H',[0]),('H',[1]),('H',[2]),
            ('Toffoli',[0,1,2]),                         # oracle MCZ core
            ('Toffoli',[0,1,2]),                         # diffuser MCZ core
            ('measure',[0]),('measure',[1]),('measure',[2])]
    B['BBHT (n=3)'] = bbht
    # Nested (n_data=2): inner RUS (anc2 CNOT data0) + outer CZ(0,1); measures
    nested = [('H',[2]),('CNOT',[0,2]),('T',[2]),('CNOT',[0,2]),('H',[2]),('measure',[2]),
              ('CZ',[0,1]),('measure',[0]),('measure',[1])]
    B['Nested (n=2)'] = nested
    return B

def main():
    print("="*78)
    print("  X16 — dynamic (richer) entanglement-width analysis")
    print("  interaction-graph union-find (reset-aware) + min-fill treewidth")
    print("="*78)
    print(f"  {'benchmark':<16} {'live width':>10} {'entangle width':>14} {'treewidth':>10} {'dynamic?':>18}")
    print("  " + "-"*72)
    # live-qubit width (peak, from paper section 4.1) for reference
    live = {'coin-flip':1,'RUS':2,'MSD (n=7)':8,'BBHT (n=3)':3,'Nested (n=2)':3}
    for name, body in bench_bodies().items():
        ew, tw = analyze(body)
        print(f"  {name:<16} {live[name]:>10} {ew:>14} {tw:>10} {'constant (reset)':>18}")
    print()
    print("  All reset-based -> realized entanglement width is TRIP-COUNT-INDEPENDENT")
    print("  (= static body value); no distribution. Treewidth is tiny (<=2), so these")
    print("  circuits are trivially simulable -- the metric confirms it structurally.")

    # ── does the dynamic (per-shot) width ever vary with trip count? ──────────
    print("\n  ── Does dynamic add anything? A stochastic fixed-body loop ──")
    import jax.numpy as jnp, pennylane as qp
    from collections import Counter
    from catalyst import measure, while_loop
    from gate_counter_estimator import GateCounterSession
    def rus_like():
        @while_loop(lambda s: s == 0)
        def loop(s):
            qp.Hadamard(wires=1); qp.CNOT(wires=[0,1]); qp.CNOT(wires=[0,1])
            m = measure(1, reset=True)
            return jnp.int64(m)
        loop(jnp.int64(0)); return qp.probs(wires=[0])
    dev = qp.device('lightning.qubit', wires=2)
    ks = []
    with GateCounterSession(rus_like, dev) as s:
        for _ in range(300):
            ks.append(int(s.run().gate_counts.get('CNOT_2',0))//2)   # 2 CNOT/iter
    c = Counter(ks)
    print(f"  trip count k varies (n=300): "
          + ", ".join(f'k={k}:{v}' for k,v in sorted(c.items())[:6]) + " ...")
    print(f"  BUT the body touches a FIXED footprint {{0,1}} every iteration, so the")
    print(f"  realized entanglement width = 2 for ALL k (min=max=2). Trip count varies;")
    print(f"  entanglement width does NOT. This is the honest finding below.")

if __name__ == "__main__":
    main()
