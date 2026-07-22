#!/usr/bin/env python3
"""
run_pilot_berg.py — recognition-vs-reaction pilot for the Berg game.

Routing (David, 2026-07-22):
    * DeepSeek (v4-pro, v4-flash) via the DeepSeek API on its free 5M tokens.
    * every other open model via Nebius Token Factory (no throttling).
    * GPT / Claude are NOT in the pilot.

What it runs. For each model key and each surface in the gradient, PILOT_N
no-history dyads (the recognition-vs-reaction core), optionally a communication
block on the canonical surface. Every record is tagged with `surface`,
`model_key`, and `reasoning_class` so power_analysis.py and later analysis can
group by cell. Behavioral parse failures are counted per cell; an aborting
infrastructure error is classified via attrition.classify_exception.

Env knobs (all optional):
    MODELS        comma list of pack keys      (default: deepseek-v4-pro,deepseek-v4-flash)
    SURFACES      comma list of reskin keys    (default: canonical,reskin,shifted,reskin_shifted,strong_novel)
    PILOT_N       dyads per cell               (default: 20)
    COMM          1 to add a communication block on canonical (default: 0)
    COMM_ROUNDS   message rounds when COMM=1   (default: 2)
    LLM_TEMPERATURE  default 1.0

Keys come from the project-root .env (DEEPSEEK_API_KEY, NEBIUS_API_KEY, ...).
Output -> data/pilot/pilot_<timestamp>.jsonl   (+ an attrition summary printed).
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

import berg_trust_game as B
import reskin as R
import providers as P
import attrition as ATT


def tag_and_write(records, fh, *, surface, model_key):
    """Add cell identity to each record and append to the master pilot file."""
    rcls = P.reasoning_class(model_key)
    n_behav = 0
    for rec in records:
        rec["surface"] = surface
        rec["model_key"] = model_key
        rec["reasoning_class"] = rcls
        if not (rec["investor_parse_clean"] and rec["trustee_parse_clean"]):
            n_behav += 1
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    return n_behav


def main() -> None:
    keys = [k.strip() for k in os.environ.get(
        "MODELS", ",".join(P.PILOT_KEYS)).split(",") if k.strip()]
    surfaces = [s.strip() for s in os.environ.get(
        "SURFACES", "canonical,reskin,shifted,reskin_shifted,strong_novel").split(",") if s.strip()]
    n = int(os.environ.get("PILOT_N", "20"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    do_comm = os.environ.get("COMM", "0") == "1"
    comm_rounds = int(os.environ.get("COMM_ROUNDS", "2"))

    for k in keys:
        if k not in P.PACK:
            sys.exit(f"unknown model key {k!r}; options: {', '.join(P.PACK)}")
    for s in surfaces:
        if s not in R.RESKINS:
            sys.exit(f"unknown surface {s!r}; options: {', '.join(R.RESKINS)}")

    missing = P.require_keys(keys)
    if missing:
        print("Missing API keys:\n  " + "\n  ".join(missing))
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parent / "data" / "pilot" / f"pilot_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    att = ATT.AttritionLog()

    print(f"Models: {keys}")
    print(f"Surfaces: {surfaces}")
    print(f"n={n} per cell, comm={'on' if do_comm else 'off'} -> {out_path}\n")

    with out_path.open("a", encoding="utf-8") as fh:
        for key in keys:
            spec = P.PACK[key]
            mcfg = P.model_config(key, temperature=temp)
            chat = B.make_chat_fn(mcfg)
            print(f"=== {key}  ({spec.provider}, {P.reasoning_class(key)}) ===")
            try:
                smoke = B.with_retry(lambda: chat("You output only JSON.",
                                                  'Return {"ok": true} and nothing else.'))
                print("  smoke:", smoke[:80].replace("\n", " "))
            except Exception as e:
                cause = ATT.classify_exception(e)
                print(f"  SMOKE FAILED [{cause}] -> skipping {key}: {e}")
                att.add(cell=f"{key}/smoke", dyad_id="smoke", role="smoke", cause=cause)
                continue

            for surface in surfaces:
                cfg = R.RESKINS[surface]
                cell = f"{key}/{surface}/no_history"
                try:
                    recs = B.run_treatment(chat, cfg, mcfg, "no_history", n, out_path.with_suffix(".raw.jsonl"))
                except Exception as e:
                    cause = ATT.classify_exception(e)
                    print(f"  {surface:16} ABORTED [{cause}]: {e}")
                    att.add(cell=cell, dyad_id="cell", role="run", cause=cause)
                    continue
                n_behav = tag_and_write(recs, fh, surface=surface, model_key=key)
                for _ in range(len(recs) - n_behav):
                    att.add(cell=cell, dyad_id="ok", role="dyad", cause="ok")
                for _ in range(n_behav):
                    att.add(cell=cell, dyad_id="fail", role="dyad", cause="format")
                s = B.summarize(recs, cell)
                mean = s.get("mean_sent")
                print(f"  {surface:16} n={len(recs):>3} clean={len(recs)-n_behav:>3} "
                      f"mean_sent={mean if mean is None else round(mean,2)}")

            if do_comm:
                cfg = R.RESKINS["canonical"]
                cell = f"{key}/canonical/communication"
                try:
                    recs = B.run_treatment(chat, cfg, mcfg, "communication", n,
                                           out_path.with_suffix(".raw.jsonl"),
                                           comm_rounds=comm_rounds)
                    n_behav = tag_and_write(recs, fh, surface="canonical", model_key=key)
                    print(f"  communication   n={len(recs)} clean={len(recs)-n_behav}")
                except Exception as e:
                    cause = ATT.classify_exception(e)
                    print(f"  communication   ABORTED [{cause}]: {e}")
                    att.add(cell=cell, dyad_id="cell", role="run", cause=cause)

    att_path = out_path.with_name(out_path.stem + "_attrition.jsonl")
    att.write(att_path)
    print("\n=== attrition by cell (infra vs behavioral) ===")
    for cell, s in att.summary().items():
        print(f"  {cell:40} attempted={s['n_attempted']:>3} "
              f"infra={s['infrastructure']} behav={s['behavioral']} "
              f"rate={s['attrition_rate']}")
    print(f"\nDone. Pilot data -> {out_path}\n"
          f"Next: python power_analysis.py {out_path} --contrast canonical strong_novel")


if __name__ == "__main__":
    main()
