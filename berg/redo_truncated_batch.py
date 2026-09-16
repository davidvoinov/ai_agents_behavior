#!/usr/bin/env python3
"""
redo_truncated_batch.py — send the dyads the token budget cut off back for a retry.

Claude Sonnet 5 at xhigh effort lost 215 of 2,800 dyads to a 4,000-token output
budget. Thinking counts against that budget as a strict limit, so the completions
where the model thought longest came back empty or cut mid-sentence. Every one of
the 215 failed on the investor call, and they sit in the cells where the decision
takes most thought, which makes the loss non-random and not safe to simply drop.

The runner marks those dyads complete, because a wave that returns a reply has
done its job whether or not the reply parses. This resets the affected dyads to
the investor-decision wave and clears what came back, so a rerun asks again with
the larger budget now set in batch_runner. The communication transcript is
untouched, since the messages parsed fine and re-running them would change the
stimulus the decision was made under.

    python3 redo_truncated_batch.py --arm sonnet-5-think --check
    python3 redo_truncated_batch.py --arm sonnet-5-think
    ./run_batch.sh sonnet-5-think          # then rerun as usual
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch_runner as BR

DECIDE_INVESTOR = BR.WAVES.index(("decide", None, "investor"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="sonnet-5-think")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    d = BR.ROOT / a.arm
    out = d / f"full_{a.arm}.jsonl"
    state_p = d / "state.json"
    if not out.exists():
        sys.exit(f"{out} does not exist, the arm has not finished")

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    bad = {r["dyad_id"] for r in rows
           if not (r["investor_parse_clean"] and r["trustee_parse_clean"])}
    byc = collections.Counter(r["condition"] for r in rows
                              if r["dyad_id"] in bad)
    bys = collections.Counter(r["surface"] for r in rows if r["dyad_id"] in bad)

    print(f"{a.arm}")
    print(f"  {len(rows)} dyads, {len(bad)} unusable ({len(bad)/len(rows):.1%})")
    print(f"  by condition: {dict(sorted(byc.items()))}")
    print(f"  by surface:   {dict(sorted(bys.items()))}")
    print(f"  budget now set for this arm: {BR.budget(a.arm)} tokens")

    if a.check:
        print("\nnothing was changed")
        return
    if not bad:
        print("\nnothing to redo")
        return

    state = json.loads(state_p.read_text())
    n = 0
    for x in state:
        if x["dyad_id"] not in bad:
            continue
        # Back to the investor decision. The transcript stays, because those
        # messages parsed and the decision has to be made under the same ones.
        x["stage"] = DECIDE_INVESTOR
        x["sent"] = None
        x["inv_raw"] = None
        x["tru_raw"] = None
        x["failed"] = None
        keep = len(x["transcript"]) if x["transcript"] else 0
        x["usage"] = x.get("usage", [])[:keep]
        n += 1

    shutil.copy(state_p, state_p.with_suffix(".json.bak_before_redo"))
    shutil.copy(out, out.with_suffix(".jsonl.bak_before_redo"))
    state_p.write_text(json.dumps(state))
    print(f"\n  {n} dyads reset to the investor decision")
    print(f"  backups: {state_p.with_suffix('.json.bak_before_redo').name}, "
          f"{out.with_suffix('.jsonl.bak_before_redo').name}")
    print(f"\nrerun to fill them:\n    ./run_batch.sh {a.arm}")


if __name__ == "__main__":
    main()
