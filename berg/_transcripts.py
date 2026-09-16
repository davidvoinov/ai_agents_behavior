#!/usr/bin/env python3
"""
_transcripts.py — one place that knows where a completed dyad lives.

The eleven open-weight arms were written to data/full/full_<arm>.jsonl. The four
closed frontier arms went through the Batch API and landed in
data/batch/<arm>/full_<arm>.jsonl instead. Three scripts globbed the first
layout only, so the new arms were invisible to the judge, to the kappa sample
and to the toggle check, without any of them reporting a problem.

Anything that needs completed dyads imports from here, so a future layout
change is one edit and not four.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE / "data" / "full"
BATCH = HERE / "data" / "batch"

# archives of failed dyads, shards and backups match the same glob as the
# estimating sample, and coding either would put dyads into the reliability
# figures that no estimate uses
SKIP = ("attrition", "purged", "quarantine", "bak", "prepurge", "__")


def sources() -> list[tuple[str, Path]]:
    """Every (arm, file) pair that holds completed dyads, in a stable order."""
    out: dict[str, Path] = {}
    for p in sorted(FULL.glob("full_*.jsonl")):
        if any(x in p.name for x in SKIP):
            continue
        out[p.name[len("full_"):-len(".jsonl")]] = p
    for p in sorted(BATCH.glob("*/full_*.jsonl")):
        if any(x in p.name for x in SKIP):
            continue
        arm = p.name[len("full_"):-len(".jsonl")]
        out.setdefault(arm, p)
    return sorted(out.items())


def communication(exclude: set[str] | None = None):
    """Yield (arm, record) for every usable communication dyad.

    A dyad enters only when both roles parsed, since compliance compares a
    coded plan against a realized decision and an unparsed decision has none.
    Duplicates across shards are dropped on the configuration and dyad id
    together, because a dyad id is unique only inside a configuration.
    """
    exclude = exclude or set()
    seen = set()
    for arm, p in sources():
        if arm in exclude:
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") != "communication" or not r.get("transcript"):
                continue
            if not (r.get("investor_parse_clean") and r.get("trustee_parse_clean")):
                continue
            key = (arm, r["surface"], r["dyad_id"])
            if key in seen:
                continue
            seen.add(key)
            r.setdefault("model_key", arm)
            yield arm, r


if __name__ == "__main__":
    import collections
    c = collections.Counter(a for a, _ in communication())
    print(f"{'arm':<24}{'communication dyads':>20}")
    for k, v in sorted(c.items()):
        print(f"  {k:<22}{v:>20}")
    print(f"  {'total':<22}{sum(c.values()):>20}   ({len(c)} configurations)")
