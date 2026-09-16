from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "full"

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    files = sorted(p for p in DATA.glob("full_*.jsonl")
                   if "attrition" not in p.name and "quarantine" not in p.name
                   and "purged" not in p.name)
    if args:
        files = [p for p in files if p.stem.replace("full_", "") in args]
    if not files:
        sys.exit("no matching data files")

    total = Counter()
    print(f"{'model':<26}{'kept':>8}{'purged':>9}")
    print("-" * 43)
    for path in files:
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        keep = [r for r in recs if r["investor_parse_clean"] and r["trustee_parse_clean"]]
        drop = [r for r in recs if not (r["investor_parse_clean"] and r["trustee_parse_clean"])]
        print(f"{path.stem.replace('full_',''):<26}{len(keep):>8}{len(drop):>9}")
        total["keep"] += len(keep); total["drop"] += len(drop)
        if apply and drop:
            shutil.copy2(path, path.with_suffix(".jsonl.prepurge"))
            arch = path.with_name(path.stem + "_purged.jsonl")
            with arch.open("a", encoding="utf-8") as fh:
                for r in drop:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            with path.open("w", encoding="utf-8") as fh:
                for r in keep:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("-" * 43)
    print(f"{'TOTAL':<26}{total['keep']:>8}{total['drop']:>9}")
    print("\nAPPLIED — failed replies archived to *_purged.jsonl, .prepurge backup kept"
          if apply else "\nreport only — pass --apply to purge")

if __name__ == "__main__":
    main()
