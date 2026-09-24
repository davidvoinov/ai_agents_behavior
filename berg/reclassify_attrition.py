from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import attrition as ATT

DATA = Path(__file__).resolve().parent / "data" / "full"

def main() -> None:
    apply = "--apply" in sys.argv
    files = sorted(p for p in DATA.glob("full_*.jsonl")
                   if "attrition" not in p.name and "quarantine" not in p.name)
    if not files:
        sys.exit(f"no data files under {DATA}")

    order = ["ok", "empty", "truncated", "degenerate", "format", "refusal", "think_only"]
    head = f"{'model':<24}" + "".join(f"{c[:9]:>10}" for c in order) + f"{'BEHAV%':>9}"
    print(head)
    print("-" * len(head))
    grand = Counter()
    worst = []

    for path in files:
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        counts = Counter()
        per_cell = defaultdict(Counter)
        rows = []
        for rec in recs:
            cell = f"{rec.get('surface')}/{rec.get('condition')}"
            for role in ("investor", "trustee"):
                cause = ATT.classify_completion(
                    rec.get(f"{role}_raw") or "", rec[f"{role}_parse_clean"])
                counts[cause] += 1
                per_cell[cell][cause] += 1
                rows.append({"cell": f"{path.stem}/{cell}", "dyad_id": rec["dyad_id"],
                             "role": role, "cause": cause,
                             "category": ATT.category(cause)})
        n = sum(counts.values())
        behav = sum(v for k, v in counts.items() if ATT.category(k) == ATT.BEHAV)
        name = path.stem.replace("full_", "")
        print(f"{name:<24}" + "".join(f"{counts.get(c,0):>10}" for c in order)
              + f"{100*behav/n:>8.1f}%")
        grand.update(counts)

        for cell, c in per_cell.items():
            tot = sum(c.values())
            bad = tot - c.get("ok", 0)
            if tot and bad / tot > 0.30:
                worst.append((bad / tot, name, cell, bad, tot))

        if apply:
            out = path.with_name(path.stem + "_attrition.jsonl")
            with out.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = sum(grand.values())
    behav = sum(v for k, v in grand.items() if ATT.category(k) == ATT.BEHAV)
    infra = sum(v for k, v in grand.items() if ATT.category(k) == ATT.INFRA)
    print("-" * len(head))
    print(f"{'TOTAL':<24}" + "".join(f"{grand.get(c,0):>10}" for c in order)
          + f"{100*behav/n:>8.1f}%")
    print(f"\n{n} role-decisions: ok={grand.get('ok',0)} "
          f"({100*grand.get('ok',0)/n:.1f}%)  infrastructure={infra} "
          f"({100*infra/n:.1f}%)  behavioral={behav} ({100*behav/n:.1f}%)")

    if worst:
        print(f"\ncells losing over 30% of decisions ({len(worst)}), worst 15:")
        for frac, name, cell, bad, tot in sorted(worst, reverse=True)[:15]:
            print(f"   {frac:6.1%}  {name:<22} {cell:<34} {bad}/{tot}")
    print("\nAPPLIED" if apply else "\nreport only — pass --apply to rewrite the ledgers")

if __name__ == "__main__":
    main()
