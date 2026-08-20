"""
rus_lowp -- low-p carry-type benchmark: a memory qubit held while a heralded
process retries (the quantum-repeater / heralded-entanglement regime, writeup
Section 5). Same carried magic state |psi0> = H T H T H |0> and the same
per-iteration idle as rus_rx_ibm, but the herald succeeds with a LOW probability
p (default 0.1, mean trip count 10) -- so the carried qubit idles through many
failed attempts and its coherent depth grows large. This is the regime the
transform is *meant* for.

It is literally rus_rx_ibm's carried-qubit model with the trip-count probability
set low: the same target/attempt are re-exported below, and the target-only fast
executors (sim/fast_target.py) draw the trip count as Geom(p) with p passed
explicitly. Unlike rus_rx_ibm (p = 5/8, mean k = 1.6), here mean k = 1/p is
large, so the trip-count tail is heavy -- the case where knitting can beat
unbounded holding.
"""

from benchmarks.rus_rx_ibm import (  # noqa: F401  (shared carried-qubit model)
    N_WIRES, TARGET, ANCILLAS, Z_IDEAL, prepare_input, attempt,
)

# heralding success probability per attempt (low: repeater / heralded regime)
P_LOWP = 0.1
P_ANALYTIC = P_LOWP
