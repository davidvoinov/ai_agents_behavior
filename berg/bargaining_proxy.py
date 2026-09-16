#!/usr/bin/env python3
"""
bargaining_proxy.py — does "recognizable wording suppresses explicit bargaining"
survive outside the one model it was measured on?

Background
----------
The result the supervisor singled out is that under a wording the model
identifies, agents transfer MORE and negotiate LESS, while under an unfamiliar
wording they build an explicit plan and that plan is more conservative. It rests
on coded transcripts, and the coded set covers only the two Qwen configurations,
596 transcripts in the plain arm and 179 in the reasoning arm. The claim is
therefore a single family result at present, and the family whose reasoning arm
the provider withdrew.

Coding all 7,946 transcripts with the judge costs a run. This script asks first
whether the effect is even present elsewhere, using surface markers that need no
model call. The markers are deliberately crude and they are not the codebook,
so a positive reading here is a reason to spend the run and a flat reading is a
reason to look again before spending it.

Markers, per transcript
    n_amounts    distinct explicit sums named in the messages
    has_number   any explicit sum named at all
    chars        total characters exchanged
    haggle       a counter-proposal marker (but, instead, how about, what if,
                 rather, actually, prefer, counter)

Usage
    python3 bargaining_proxy.py
"""
from __future__ import annotations

import json
import re
import glob
import collections
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent / "data" / "full"
PAIRS = [("canonical", "reskin"), ("shifted", "reskin_shifted"),
         ("strong_control", "strong_novel")]
HAGGLE = re.compile(r"\b(but|instead|how about|what if|rather|actually|"
                    r"prefer|counter|propose|negotiat)\w*\b", re.I)
NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
THINK = re.compile(r"<think>.*?</think>", re.S)


def markers(rec: dict) -> dict:
    msgs = rec.get("transcript") or []
    text = " ".join(THINK.sub("", str(m.get("message", ""))) for m in msgs)
    nums = {n for n in NUM.findall(text)}
    return {"n_amounts": len(nums), "has_number": float(bool(nums)),
            "chars": len(text), "haggle": float(bool(HAGGLE.search(text)))}


def main() -> None:
    print("Recognizable wording against its de-recognized counterpart, "
          "communication treatment.")
    print("A NEGATIVE difference means the recognizable wording bargains LESS, "
          "which is the\nreported direction.\n")
    print(f"{'model':<24}{'Δ n_amounts':>13}{'Δ chars':>10}{'Δ haggle':>10}"
          f"{'  p(chars)':>11}{'  direction':>12}")
    print("-" * 82)
    agree = collections.Counter()
    for p in sorted(DATA.glob("full_*.jsonl")):
        n = p.name
        if any(x in n for x in ("attrition", "purged", "quarantine", "bak",
                                "prepurge", "__")):
            continue
        model = n[len("full_"):-len(".jsonl")]
        by = collections.defaultdict(list)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["condition"] == "communication" and r.get("transcript")
                    and r["investor_parse_clean"] and r["trustee_parse_clean"]):
                by[r["surface"]].append(markers(r))
        rec_side, plain_side = [], []
        for a, b in PAIRS:
            if len(by[a]) >= 20 and len(by[b]) >= 20:
                rec_side += by[a]
                plain_side += by[b]
        if len(rec_side) < 60 or len(plain_side) < 60:
            continue
        def d(k):
            return (np.mean([x[k] for x in rec_side])
                    - np.mean([x[k] for x in plain_side]))
        _, pv = stats.mannwhitneyu([x["chars"] for x in rec_side],
                                   [x["chars"] for x in plain_side],
                                   alternative="two-sided")
        dn, dc, dh = d("n_amounts"), d("chars"), d("haggle")
        # the reported direction is less explicit bargaining under recognition
        votes = sum(1 for v in (dn, dc, dh) if v < 0)
        direction = "reported" if votes >= 2 else "opposite"
        agree[direction] += 1
        print(f"{model:<24}{dn:>13.2f}{dc:>10.0f}{dh:>10.3f}{pv:>11.2g}"
              f"{direction:>12}")
    print("-" * 82)
    print(f"\n{agree['reported']} configurations move in the reported direction, "
          f"{agree['opposite']} against it.")
    print("\nThese markers are a screen and not the codebook. The judge run is what\n"
          "settles the claim, and this says how likely that run is to confirm it.")


if __name__ == "__main__":
    main()
