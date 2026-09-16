#!/usr/bin/env python3
"""
score_kappa.py — score the human coding against the automated judge.

Run this only after coding_form.csv is complete. It reads the human codes, the
judge codes for the same transcripts, and reports Cohen's kappa per categorical
variable together with raw agreement and the disagreements themselves, since the
disagreements are what a revision would have to act on.

A kappa is undefined when one coder used a single category throughout, which
happens on a rare variable at this sample size. That case is reported as such
and not as a zero, because the two carry opposite meanings.

Usage
    python3 score_kappa.py              # the newest round on disk
    python3 score_kappa.py --round 2    # an earlier one
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGE = HERE / "data" / "coding" / "judge_comm.jsonl"


def round_dir(n: int | None = None) -> Path:
    """Which validation round to score. Defaults to the newest one on disk."""
    if n is not None:
        return HERE / "data" / "coding" / f"kappa_round{n}"
    dirs = sorted((HERE / "data" / "coding").glob("kappa_round*"),
                  key=lambda p: int(p.name.replace("kappa_round", "") or 0))
    if not dirs:
        sys.exit("no kappa_round* directory, run make_kappa_sample.py first")
    return dirs[-1]


ROUND = round_dir()

CATEGORICAL = ["explicit_agreement", "agreement_type", "equality_framing",
               "promise_investor", "promise_trustee"]
NUMERIC = ["agreed_send", "agreed_return"]


def norm(v, var=None):
    """Fold the spellings a human and a judge use for the same value.

    `none` is a real category of agreement_type in the codebook and a missing
    marker everywhere else, so the variable has to be known before folding it.
    Treating it as missing would silently drop every transcript without an
    agreement, which is a third of the sample and the easiest third to agree on.
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if var == "agreement_type" and s in ("none", "no", "no agreement"):
        return "none"
    if s in ("", "na", "n/a", "none", "null", "-"):
        return None
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return s


def kappa(a, b):
    """Cohen's kappa on paired labels, ignoring pairs where either is missing."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0.0, 0
    cats = sorted({c for p in pairs for c in p}, key=str)
    n = len(pairs)
    obs = sum(1 for x, y in pairs if x == y) / n
    if len(cats) == 1:
        return None, obs, n          # both coders used one category throughout
    exp = sum((sum(1 for x, _ in pairs if x == c) / n) *
              (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    if exp >= 1.0:
        return None, obs, n
    return (obs - exp) / (1 - exp), obs, n


def main() -> None:
    global ROUND
    if "--round" in sys.argv:
        ROUND = round_dir(int(sys.argv[sys.argv.index("--round") + 1]))
    print(f"scoring {ROUND.name}\n")
    form = ROUND / "coding_form.csv"
    if not form.exists():
        sys.exit(f"no coding form at {form}")
    # a form round-tripped through a spreadsheet comes back semicolon separated
    # in a European locale, so the delimiter is detected instead of assumed
    raw = form.read_text(encoding="utf-8-sig")
    try:
        delim = csv.Sniffer().sniff(raw[:4000], delimiters=",;\t").delimiter
    except csv.Error:
        delim = ","
    human = {}
    with form.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=delim):
            if not row.get("item"):
                continue
            if any(str(row.get(v) or "").strip() for v in CATEGORICAL):
                human[int(row["item"])] = row
    if not human:
        sys.exit("the coding form is still empty")

    key = {json.loads(l)["item"]: json.loads(l)
           for l in (ROUND / "ANSWERS_DO_NOT_OPEN_FIRST.jsonl").read_text().splitlines()
           if l.strip()}
    judged = {}
    if JUDGE.exists():
        for l in JUDGE.read_text().splitlines():
            if l.strip():
                j = json.loads(l)
                judged[(j.get("model_key"), j["dyad_id"])] = j

    items = sorted(human)
    missing = [i for i in items
               if (key[i]["model_key"], key[i]["dyad_id"]) not in judged]
    if missing:
        print(f"NOTE: {len(missing)} of {len(items)} items carry no judge code yet.")
        print("      Run the judge over these transcripts before scoring, with")
        print("      MODELS_JUDGE=gpt-oss-120b-nebius python3 run_judge.py\n")
        items = [i for i in items if i not in missing]
    if not items:
        sys.exit("nothing to score until the judge has coded the same transcripts")

    print(f"{len(items)} items coded by both\n")
    print(f"{'variable':<22}{'kappa':>9}{'raw agree':>12}{'n':>6}   reading")
    print("-" * 72)
    for v in CATEGORICAL:
        h = [norm(human[i].get(v), v) for i in items]
        j = [norm(judged[(key[i]["model_key"], key[i]["dyad_id"])].get(v), v) for i in items]
        k, obs, n = kappa(h, j)
        if k is None:
            print(f"{v:<22}{'undefined':>9}{obs:>11.1%}{n:>6}   one category only")
        else:
            read = ("substantial" if k >= 0.70 else
                    "moderate" if k >= 0.40 else "poor")
            print(f"{v:<22}{k:>9.3f}{obs:>11.1%}{n:>6}   {read}")
    for v in NUMERIC:
        h = [norm(human[i].get(v), v) for i in items]
        j = [norm(judged[(key[i]["model_key"], key[i]["dyad_id"])].get(v), v) for i in items]
        pairs = [(x, y) for x, y in zip(h, j) if x is not None or y is not None]
        same = sum(1 for x, y in pairs if str(x) == str(y))
        print(f"{v:<22}{'--':>9}{(same/len(pairs) if pairs else 0):>11.1%}"
              f"{len(pairs):>6}   numeric, exact match")

    print("\ndisagreements, which are what a revision would act on:")
    shown = 0
    for i in items:
        k_ = key[i]
        jj = judged[(k_["model_key"], k_["dyad_id"])]
        diff = [v for v in CATEGORICAL
                if norm(human[i].get(v), v) != norm(jj.get(v), v)]
        if diff and shown < 15:
            shown += 1
            print(f"  item {i:>3} [{k_['model_key']}/{k_['surface']}]  {', '.join(diff)}")
            for v in diff:
                print(f"        {v:<20} human={norm(human[i].get(v), v)!r:<10} "
                      f"judge={norm(jj.get(v), v)!r}")
    if not shown:
        print("  none")
    print("\nA kappa at or above 0.70 across the categorical variables supports "
          "reporting\nthe automated coding as validated against a human reading.")


if __name__ == "__main__":
    main()
