from __future__ import annotations

import json
import csv
import os
import random
import re
import sys
import collections
from pathlib import Path

import _transcripts as T
from run_judge import speaker_of

HERE = Path(__file__).resolve().parent
JUDGED = HERE / "data" / "coding" / "judge_comm.jsonl"
THINK = re.compile(r"<think>.*?</think>", re.S)

ROUND = int(os.environ.get("KAPPA_ROUND", "4"))
EXCLUDE_ARMS = {"qwen3-think"}
ONLY = None
OUT = HERE / "data" / "coding" / f"kappa_round{ROUND}"
SEED = 20260913
TARGET = int(os.environ.get("KAPPA_N", "75"))

VARS = ["explicit_agreement", "agreement_type", "equality_framing",
        "promise_investor", "promise_trustee", "agreed_send", "agreed_return"]

def seen_by_humans() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    coding = HERE / "data" / "coding"
    for f in sorted(coding.glob("coder_b_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            uid = json.loads(line).get("uid", "")
            parts = uid.split("|")
            if len(parts) >= 3:
                out.add((parts[0], "|".join(parts[2:])))
    for f in sorted(coding.glob("kappa_round*/ANSWERS_DO_NOT_OPEN_FIRST.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                out.add((j["model_key"], j["dyad_id"]))
    return out

def clean(msg) -> str:
    return THINK.sub("", str(msg or "")).strip()

def main() -> None:
    global TARGET
    if len(sys.argv) > 1:
        TARGET = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)

    already = seen_by_humans()

    pool = collections.defaultdict(list)
    for model, r in T.communication(exclude=EXCLUDE_ARMS):
        if ONLY and model not in ONLY:
            continue
        if (model, r["dyad_id"]) in already:
            continue
        pool[(model, r["surface"])].append((model, r))
    if not pool:
        raise SystemExit(f"round {ROUND} found no communication dyads a human "
                         f"has not already read")

    rng = random.Random(SEED)
    for rows in pool.values():
        rng.shuffle(rows)

    by_model = collections.defaultdict(list)
    for (m, s_), rows in pool.items():
        if rows:
            by_model[m].append(s_)
    eligible = sorted(by_model)
    per = max(1, TARGET // max(1, len(eligible)))
    all_surf = sorted({s_ for v in by_model.values() for s_ in v})
    sample, used = [], set()
    for i, m in enumerate(eligible):
        surfs = [s_ for s_ in all_surf if s_ in by_model[m]]
        order = surfs[i % len(surfs):] + surfs[:i % len(surfs)]
        taken = 0
        for s_ in order:
            for cand in pool[(m, s_)]:
                key = (cand[0], cand[1]["dyad_id"])
                if key in used:
                    continue
                used.add(key)
                sample.append(cand)
                taken += 1
                break
            if taken >= per:
                break
    if len(sample) < TARGET:
        spare = [c for key in sorted(pool, key=lambda k: -len(pool[k]))
                 for c in pool[key]
                 if (c[0], c[1]["dyad_id"]) not in used]
        rng.shuffle(spare)
        for c in spare[:TARGET - len(sample)]:
            used.add((c[0], c[1]["dyad_id"]))
            sample.append(c)
    rng.shuffle(sample)
    sample = sample[:TARGET]

    md = [f"# Coding sample, round {ROUND}",
          "",
          f"{len(sample)} transcripts, stratified over configuration and surface, "
          "in random order.",
          "Read the codebook in `berg/COMM_CODEBOOK.md` before starting and keep it open.",
          "",
          "Record every judgement in `coding_form.csv`. Do not open the answers file "
          "until the form is complete.",
          "", "---", ""]
    answers = []
    for i, (model, r) in enumerate(sample, 1):
        md += [f"## Item {i:03d}", "",
               f"Endowment {r['endowment']}, multiplier {r['multiplier']}, "
               f"unit as written in the messages.", ""]
        for m in r["transcript"]:
            txt = clean(m.get("message"))
            if txt:
                md.append(f"- **{speaker_of(m)}**: {txt}")
        md += ["", "---", ""]
        answers.append({"item": i, "dyad_id": r["dyad_id"], "model_key": model,
                        "surface": r["surface"], "sent": r["sent"],
                        "returned": r["returned"]})

    (OUT / "transcripts.md").write_text("\n".join(md))

    with (OUT / "coding_form.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item"] + VARS + ["notes"])
        for i in range(1, len(sample) + 1):
            w.writerow([i] + [""] * (len(VARS) + 1))

    with (OUT / "ANSWERS_DO_NOT_OPEN_FIRST.jsonl").open("w") as fh:
        for a in answers:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")

    strata_n = sorted({m for m, _ in sample})
    surf_n = sorted({r["surface"] for _, r in sample})
    scope = (
        "This round replaces the earlier ones rather than adding to them. Rounds one\n"
        "and two were read by a human who saw the label \"party\" on both sides of every\n"
        "exchange, because the renderer looked for a field the engine never wrote. The\n"
        "two variables defined per role were therefore coded by a person inferring who\n"
        "spoke, and those two are the ones that failed, at 0.467 and 0.286, while the\n"
        "three that need no role held between 0.648 and 0.781. The labels are repaired\n"
        "here and the judge reads the same repaired transcripts, so one kappa now covers\n"
        "every configuration in the estimates on material both coders could read."
    )
    (OUT / "README.md").write_text(f"""# Human validation round {ROUND}, procedure

{len(sample)} transcripts over {len(strata_n)} configurations and {len(surf_n)} surfaces, drawn
with seed {SEED}. Every transcript an earlier round put in front of a person is
excluded, so the codebook is tested against material it was not tuned on.

{scope}

## Procedure

1. Open `berg/COMM_CODEBOOK.md` and keep it open. Code to the codebook as
   written, without adjusting it as you go, since a codebook revised mid-sample
   makes the kappa a fit to the sample.
2. Read `transcripts.md` top to bottom and fill one row of `coding_form.csv`
   per item. Leave a cell empty only when the codebook says the variable does
   not apply.
3. Do not open `ANSWERS_DO_NOT_OPEN_FIRST.jsonl`. It carries the identity of
   each item and exists so the scoring can join your rows to the judge's.
4. When the form is complete, run `python3 score_kappa.py --round {ROUND}`.

## What the numbers will mean

`explicit_agreement` carries the compliance results and is the variable the
paper most depends on. `promise_investor` and `promise_trustee` are the open
question of this round. They were withdrawn from every estimate after the
earlier validation, where a high agreement between two machines sat next to a
poor agreement with a person, and the labels those coders were denied are
exactly what the two variables need. If they recover here, the earlier failure
was a rendering fault and the withdrawal can be revisited. If they stay low on
labelled transcripts, the conjunctive definition is what fails and the
withdrawal stands on firmer ground than before.

Code without trying to serve either outcome.
""")

if __name__ == "__main__":
    main()
