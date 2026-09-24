from __future__ import annotations

import csv
import json
import sys
import collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECODE = HERE / "data" / "coding" / "promise_recode"
ROUND = HERE / "data" / "coding" / "kappa_round2"
JUDGE = HERE / "data" / "coding" / "judge_comm.jsonl"
VARS = ["promise_investor", "promise_trustee"]

def read_csv(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        delim = csv.Sniffer().sniff(raw[:4000], delimiters=",;\t").delimiter
    except csv.Error:
        delim = ","
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delim))

def norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "na", "n/a", "null", "-"):
        return None
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return None

def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0.0, 0
    n = len(pairs)
    obs = sum(1 for x, y in pairs if x == y) / n
    cats = sorted({c for p in pairs for c in p}, key=str)
    if len(cats) == 1:
        return None, obs, n
    exp = sum((sum(1 for x, _ in pairs if x == c) / n) *
              (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    return ((obs - exp) / (1 - exp) if exp < 1 else None), obs, n

def main() -> None:
    form = RECODE / "promise_form.csv"
    if not form.exists():
        sys.exit(f"no form at {form}")
    new = {int(r["item"]): r for r in read_csv(form)
           if r.get("item") and any(str(r.get(v) or "").strip() for v in VARS)}
    if not new:
        sys.exit("the recode form is still empty")

    mapping = {json.loads(l)["item"]: json.loads(l)
               for l in (RECODE / "MAPPING_DO_NOT_OPEN_FIRST.jsonl").read_text().splitlines()
               if l.strip()}
    first = {int(r["item"]): r for r in read_csv(ROUND / "coding_form.csv")
             if r.get("item")}
    judged = {}
    for l in JUDGE.read_text().splitlines():
        j = json.loads(l)
        if j.get("coder") == "judge:judge-gptoss":
            judged[(j["model_key"], j["dyad_id"])] = j

    items = sorted(new)
    print(f"{len(items)} items recoded\n")
    print(f"{'variable':<20}{'kappa now':>11}{'was':>8}{'raw agree':>12}{'n':>5}")
    print("-" * 58)
    WAS = {"promise_investor": 0.467, "promise_trustee": 0.286}
    for v in VARS:
        h = [norm(new[i].get(v)) for i in items]
        j = [norm(judged[(mapping[i]["model_key"], mapping[i]["dyad_id"])].get(v))
             for i in items]
        k, obs, n = kappa(h, j)
        ks = f"{k:.3f}" if k is not None else "undefined"
        print(f"{v:<20}{ks:>11}{WAS[v]:>8.3f}{obs:>11.1%}{n:>5}")

    print("\nhow far the reading moved from the first pass")
    print(f"{'variable':<20}{'changed':>9}{'True->False':>13}{'False->True':>13}")
    print("-" * 56)
    for v in VARS:
        ch = tf = ft = 0
        for i in items:
            o = mapping[i]["orig_item"]
            a, b = norm(first.get(o, {}).get(v)), norm(new[i].get(v))
            if a is None or b is None or a == b:
                continue
            ch += 1
            tf += a is True and b is False
            ft += a is False and b is True
        print(f"{v:<20}{ch:>9}{tf:>13}{ft:>13}")
    print("\nA move concentrated in True to False is the signature of the third\n"
          "condition being applied where it was missed, which is what the audit\n"
          "of the first pass predicted.")

    print("\nwhich condition carries the Falses")
    for v in VARS:
        col = f"failed_rule_{v.split('_')[1]}"
        c = collections.Counter()
        multi = 0
        for i in items:
            if norm(new[i].get(v)) is not False:
                continue
            parts = [x.strip() for x in str(new[i].get(col) or "").split(",") if x.strip()]
            multi += len(parts) > 1
            for x in parts:
                c[x] += 1
        lab = {"1": "no committing verb", "2": "no specific quantity",
               "3": "hedged"}
        pretty = {lab.get(k, k): n for k, n in c.most_common()}
        n_false = sum(1 for i in items if norm(new[i].get(v)) is False)
        print(f"  {v:<20}{pretty}   over {n_false} Falses, {multi} citing more than one")

    print("\nAt or above 0.70 the corrected coding is usable and the promise based\n"
          "analysis stands. Below it the definition does not carry between readers,\n"
          "which is a result about the instrument and is reported as one.")

if __name__ == "__main__":
    main()
