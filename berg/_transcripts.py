from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE / "data" / "full"
BATCH = HERE / "data" / "batch"

SKIP = ("attrition", "purged", "quarantine", "bak", "prepurge", "__")

def sources() -> list[tuple[str, Path]]:
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
