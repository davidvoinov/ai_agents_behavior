#!/usr/bin/env python3
"""
purge_truncated.py — drop dyads the token budget cut off, so a rerun redoes them.

Two reasoning arms of the incentive control were launched without MAX_TOKENS and
fell back to the module default of 1500. On a reasoning model the hidden
reasoning consumes that budget before the JSON is finished, so the completion
arrives either empty or cut mid-string. DeepSeek-V4-Pro reasoning lost 37% of
its dyads this way and GLM-5.2 reasoning lost 56%, both concentrated at the two
higher values of k where the decision takes more thought.

The runner resumes by dyad id, so a rerun alone skips everything: all 300 ids
are in the file whether or not the row is usable. This removes the unusable rows
and leaves the clean ones, after which a rerun with a sane budget fills the gaps.

A backup is written next to the file before anything is removed.

    python3 purge_truncated.py --check          # report, change nothing
    python3 purge_truncated.py
    python3 purge_truncated.py --files data/incentive/incentive_glm-5.2-think.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

DEFAULT = ["data/incentive/incentive_deepseek-v4-pro-think.jsonl",
           "data/incentive/incentive_glm-5.2-think.jsonl"]


def usable(r: dict) -> bool:
    return bool(r.get("investor_parse_clean") and r.get("trustee_parse_clean"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=",".join(DEFAULT))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    for name in [x.strip() for x in a.files.split(",") if x.strip()]:
        p = Path(name)
        if not p.exists():
            print(f"{name}: missing")
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        good = [r for r in rows if usable(r)]
        bad = [r for r in rows if not usable(r)]
        byk = collections.Counter(round(r["k_level"], 2) for r in bad)
        print(f"\n{p.name}")
        print(f"  {len(rows)} rows, {len(good)} usable, {len(bad)} truncated")
        print(f"  truncated by k: {dict(sorted(byk.items()))}")
        keep = collections.Counter(round(r["k_level"], 2) for r in good)
        print(f"  kept by k:      {dict(sorted(keep.items()))}")
        if a.check:
            continue
        shutil.copy(p, p.with_suffix(".jsonl.bak_truncated"))
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in good))
        print(f"  written back with {len(good)} rows, "
              f"backup at {p.with_suffix('.jsonl.bak_truncated').name}")

    if a.check:
        print("\nnothing was changed")
    else:
        print("\nrerun with a budget that fits a reasoning completion:")
        print("    MODELS=deepseek-v4-pro-think N=100 MAX_TOKENS=4000 \\")
        print("        nohup python3 run_incentive_control.py > log_inc_dsT.log 2>&1 &")


if __name__ == "__main__":
    main()
