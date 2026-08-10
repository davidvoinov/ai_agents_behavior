#!/usr/bin/env python3
"""Emit deterministic Promise Switch binary-outcome power planning tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.power import ALPHAS, CANDIDATE_N, MDES, REFERENCE_RATES, build_power_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", nargs="+", type=float, default=REFERENCE_RATES)
    parser.add_argument("--mde", nargs="+", type=float, default=MDES)
    parser.add_argument("--n", nargs="+", type=int, default=CANDIDATE_N)
    parser.add_argument("--alpha", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--target-power", nargs="+", type=float, default=(0.80, 0.90))
    args = parser.parse_args()
    report = build_power_report(
        reference_rates=args.p0,
        mdes=args.mde,
        candidate_n=args.n,
        alphas=args.alpha,
        target_powers=args.target_power,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
