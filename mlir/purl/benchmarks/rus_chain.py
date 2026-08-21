"""
rus_chain -- N sequential RUS gates on one carried data qubit (headline for the
knit-beats-discard regime, spec sweep 3).

Same carried magic state and same p=5/8 coin as rus_rx_ibm, but the data qubit
is threaded through N sequential RUS stages (run via the N argument of the
knit_runtime runners). Truncate+discard keeps a shot only if EVERY stage
succeeds within C -- keep prob (1-(1-p)^C)^N collapses multiplicatively in N --
whereas knit cuts at each stage boundary and never discards. The two estimators
cross at some N* <= 8.

There is no separate executor here: the chain semantics live in the N-stage loop
of run_unbounded/run_truncated/run_knit. This module just re-exports the coin so
`rus_chain` reads as its own benchmark and can carry chain-specific metadata.
"""

from benchmarks.rus_rx_ibm import (  # noqa: F401  (re-exported as the bench API)
    N_WIRES, TARGET, ANCILLAS, P_ANALYTIC, Z_IDEAL,
    prepare_input, attempt,
)

# default chain lengths for sweep 3
CHAIN_LENGTHS = (1, 2, 4, 8)
