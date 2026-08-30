"""
variants.py -- leakage sensitivity sweep via variant calibration JSONs.

Leakage is calibration data now (spec 4.1), the `leak_2q_default` key in the JSON,
NOT a CLI knob. To study sensitivity to the per-two-qubit-gate leakage rate we
generate one `ibm_eagle_r3_leak<value>.json` per rate with the dataset generator,
then run `eval/experiment.py --ibm --ibm-json <file>` against each. This restores
the old `--leak` sweep while keeping a single source of truth (the JSON).

Usage:
  PYTHONPATH=. python3 eval/variants.py --bench rus_lowp --leaks 0 5e-4 1e-3 2e-3
"""

import argparse
import os
import subprocess
import sys

from sim.ibm_dataset import build_json, JSON_PATH

VARIANT_DIR = os.path.dirname(JSON_PATH)


def variant_path(leak):
    return os.path.join(VARIANT_DIR, f"ibm_eagle_r3_leak{leak:g}.json")


def make_variant(leak, leak_spread=0.0):
    """Write a calibration JSON identical to the base but with leak_2q_default=leak
    (and optional per-edge spread). Returns the file path."""
    path = variant_path(leak)
    build_json(path=path, leak_2q_default=leak, leak_spread=leak_spread)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="rus_lowp",
                    choices=["rus_lowp", "rus_rx_ibm", "ipe"])
    ap.add_argument("--leaks", type=float, nargs="+",
                    default=[0.0, 5e-4, 1e-3, 2e-3],
                    help="per-2q-gate leakage rates (leak_2q_default) to sweep")
    ap.add_argument("--carry-qubit", type=int, default=0)
    ap.add_argument("--leak-spread", type=float, default=0.0,
                    help="optional per-edge leak_2q spread fraction (0 = uniform)")
    args, passthrough = ap.parse_known_args()

    for leak in args.leaks:
        path = make_variant(leak, args.leak_spread)
        print(f"\n### leak_2q_default = {leak:g}   ({os.path.basename(path)})")
        subprocess.run(
            [sys.executable, "eval/experiment.py", "--bench", args.bench, "--ibm",
             "--ibm-json", path, "--carry-qubit", str(args.carry_qubit)] + passthrough,
            check=True)


if __name__ == "__main__":
    main()
