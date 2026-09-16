from __future__ import annotations

import csv
import json
import glob
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE / "data" / "coding" / "kappa_round2"
OUT = HERE / "data" / "coding" / "promise_recode"
THINK = re.compile(r"<think>.*?</think>", re.S)
SEED = 20260905

CHECKLIST = """\
Code True only if ALL THREE hold for that party. Any one failing makes it False.

  1. COMMITTING VERB, first person, about that party's own future action
     will, I'll, commit, promise, guarantee, honor, plan to, intend to,
     agree to, proceed with

  2. A SPECIFIC QUANTITY or an explicit proportional rule
     a floor such as "at least 6" or "at least half" counts
     "a large amount", "a fair share", "a good chunk" do NOT count

  3. NO HEDGE OF UNCERTAINTY. These disqualify even when a number is present
     considering, thinking of, might, maybe, could, perhaps, leaning toward,
     open to, aim to, aim for, hope to, would be happy to, willing to,
     and hypothetical framing such as "let us say I send 5" or "if I were to send 5"

  A conditional of the form "if you send X, I will return Y" QUALIFIES, because
  the condition is the other party's move and not the speaker's uncertainty.

  Record in failed_rule which condition decided a False, as 1, 2, or 3.
"""

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key = {json.loads(l)["item"]: json.loads(l)
           for l in (ROUND / "ANSWERS_DO_NOT_OPEN_FIRST.jsonl").read_text().splitlines()
           if l.strip()}
    tx = {}
    for f in sorted(glob.glob(str(HERE / "data" / "full" / "full_*.jsonl"))):
        if any(x in f for x in ("_attrition", "_purged", "__")):
            continue
        m = f.split("full_")[-1][:-6]
        for line in open(f):
            r = json.loads(line)
            if r.get("condition") == "communication":
                tx[(m, r["dyad_id"])] = r

    order = sorted(key)
    random.Random(SEED).shuffle(order)

    md = ["# Promise recode", "",
          "Only `promise_investor` and `promise_trustee`. Every other variable from",
          "the first pass stands and is not revisited.", "",
          "The items are in a new order, so an item number here does NOT match the",
          "first pass. Fill `promise_form.csv` by the number shown on each item.", "",
          "```", CHECKLIST, "```", "", "---", ""]

    rows = []
    for n, item in enumerate(order, 1):
        k = key[item]
        rec = tx[(k["model_key"], k["dyad_id"])]
        msgs = [THINK.sub("", str(m.get("message") or "")).strip()
                for m in rec["transcript"]]
        first = [m for i, m in enumerate(msgs) if i % 2 == 0 and m]
        second = [m for i, m in enumerate(msgs) if i % 2 == 1 and m]
        md += [f"## Item {n:02d}", "",
               f"Endowment {rec['endowment']}, multiplier {rec['multiplier']}.", "",
               "**FIRST MOVER** — judge `promise_investor` on these lines only", ""]
        md += [f"> {m}" for m in first] or ["> (no messages)"]
        md += ["", "**SECOND MOVER** — judge `promise_trustee` on these lines only", ""]
        md += [f"> {m}" for m in second] or ["> (no messages)"]
        md += ["", "---", ""]
        rows.append({"item": n, "orig_item": item, "model_key": k["model_key"],
                     "surface": k["surface"], "dyad_id": k["dyad_id"]})

    (OUT / "transcripts.md").write_text("\n".join(md))

    with (OUT / "promise_form.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "promise_investor", "failed_rule_investor",
                    "promise_trustee", "failed_rule_trustee", "notes"])
        for r in rows:
            w.writerow([r["item"], "", "", "", "", ""])

    with (OUT / "MAPPING_DO_NOT_OPEN_FIRST.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    (OUT / "README.md").write_text(f"""# Promise recode, procedure

Sixty transcripts, the same ones as the first pass, reshuffled with seed {SEED}
and stripped down to the two promise variables.

## Why all sixty

Recoding only the disputed items would compare a careful reading against a
hurried one on exactly the cases where they differed, which manufactures
agreement. The kappa from a partial recode would not mean anything.

## What to do

1. Keep the checklist at the top of `transcripts.md` visible while coding.
2. Code in order, filling `promise_form.csv` by the item number shown.
3. For every False, put 1, 2, or 3 in the failed_rule column for which condition
   decided it. That column is what makes the next disagreement diagnosable.
4. Do not open `MAPPING_DO_NOT_OPEN_FIRST.jsonl`, which links these items back to
   the first pass.
5. Run `python3 score_promise_recode.py` when finished.

## The two likely outcomes

A kappa at or above 0.70 confirms the first pass drifted from the written rule
and the coding is usable once corrected. A kappa that stays low says the
definition itself does not carry between readers, which is a finding about the
instrument and belongs in the paper as one.

Roughly 1 to 2 minutes per item, since only two judgements are needed.
""")
    print(f"wrote {len(rows)} items to {OUT}")
    print("  transcripts.md   checklist above every item, parties separated")
    print("  promise_form.csv two variables plus a failed_rule column")
    print("  README.md        why all sixty and not the sixteen")

if __name__ == "__main__":
    main()
