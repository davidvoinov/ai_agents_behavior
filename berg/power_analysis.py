#!/usr/bin/env python3
"""
power_analysis.py
=================
Turn pilot variances into a required sample size per cell (PI directive
2026-07-22, point 1). No scipy dependency.

Two-sample comparison of means, equal n per group, normal approximation:

    n_per_group = 2 * (z_{1-alpha/2} + z_{1-beta})^2 / d^2 ,   d = (m1 - m2) / s_pooled

Usage:
    python power_analysis.py PILOT.jsonl                 # descriptives + required-n grid
    python power_analysis.py PILOT.jsonl --metric sent   # sent (default) or returned
    python power_analysis.py PILOT.jsonl --contrast canonical reskin   # observed d + n

Reads a JSONL of dyad records. Recognized fields (first present wins):
    metric   : sent | amount_sent | transfer  /  returned | amount_returned
    surface  : surface | variant | reskin | skin
    condition: condition | treatment | phase
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}     # z_{1-beta}
Z_ALPHA2 = {0.05: 1.9600, 0.10: 1.6449, 0.01: 2.5758}


def required_n(d: float, power: float = 0.80, alpha: float = 0.05) -> float:
    if d == 0:
        return float("inf")
    za, zb = Z_ALPHA2[alpha], Z[power]
    return 2 * (za + zb) ** 2 / (d * d)


def _first(rec: dict, keys) -> object:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def _metric_val(rec: dict, metric: str):
    if metric == "sent":
        return _first(rec, ["sent", "amount_sent", "transfer", "x"])
    return _first(rec, ["returned", "amount_returned", "y"])


def _cell(rec: dict) -> str:
    surf = _first(rec, ["surface", "variant", "reskin", "skin"]) or "?"
    cond = _first(rec, ["condition", "treatment", "phase"]) or "?"
    return f"{surf}/{cond}"


def load_cells(path: str, metric: str) -> dict[str, list[float]]:
    cells: dict[str, list[float]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        v = _metric_val(rec, metric)
        if isinstance(v, (int, float)):
            cells[_cell(rec)].append(float(v))
    return cells


def describe(cells: dict[str, list[float]]) -> None:
    print(f"{'cell':32} {'n':>4} {'mean':>7} {'sd':>7}")
    print("-" * 54)
    for cell in sorted(cells):
        xs = cells[cell]
        n = len(xs)
        m = st.mean(xs) if n else 0.0
        s = st.pstdev(xs) if n > 1 else 0.0
        print(f"{cell:32} {n:>4} {m:>7.2f} {s:>7.2f}")


def grid(cells: dict[str, list[float]]) -> None:
    all_x = [x for xs in cells.values() for x in xs]
    s = st.pstdev(all_x) if len(all_x) > 1 else 0.0
    print(f"\nPooled SD across pilot = {s:.2f} (metric spread).")
    print("Required n PER CELL to detect a raw mean difference Δ at 80% power, α=.05:")
    print(f"{'Δ (in metric units)':22} {'d = Δ/SD':>9} {'n/cell':>8}")
    print("-" * 42)
    for delta in (0.5, 1.0, 1.5, 2.0, 2.5):
        d = delta / s if s else float("inf")
        n = required_n(d)
        print(f"{delta:>10.1f}{'':12} {d:>9.2f} {math.ceil(n) if math.isfinite(n) else '∞':>8}")


def contrast(cells, a: str, b: str) -> None:
    ca = next((c for c in cells if c.startswith(a)), None)
    cb = next((c for c in cells if c.startswith(b)), None)
    if not ca or not cb:
        print(f"\n[contrast] could not find cells starting '{a}' and '{b}'.")
        return
    xa, xb = cells[ca], cells[cb]
    ma, mb = st.mean(xa), st.mean(xb)
    va = st.pvariance(xa) if len(xa) > 1 else 0.0
    vb = st.pvariance(xb) if len(xb) > 1 else 0.0
    sp = math.sqrt((va + vb) / 2) or float("nan")
    d = (ma - mb) / sp if sp else float("nan")
    print(f"\nObserved contrast {ca} vs {cb}:")
    print(f"  means {ma:.2f} vs {mb:.2f}   pooled SD {sp:.2f}   d = {d:.2f}")
    for pw in (0.80, 0.90):
        n = required_n(abs(d), power=pw)
        print(f"  n/cell for {int(pw*100)}% power: "
              f"{math.ceil(n) if math.isfinite(n) else '∞'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot")
    ap.add_argument("--metric", default="sent", choices=["sent", "returned"])
    ap.add_argument("--contrast", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    cells = load_cells(args.pilot, args.metric)
    if not cells:
        print("No usable records. Check field names in the JSONL.")
        return
    print(f"Metric: {args.metric}\n")
    describe(cells)
    grid(cells)
    if args.contrast:
        contrast(cells, *args.contrast)
    else:
        # default: the H2 recognition contrast, if present
        contrast(cells, "canonical", "reskin")


if __name__ == "__main__":
    main()
