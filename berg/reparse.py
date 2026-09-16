from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import berg_trust_game as B

DATA = Path(__file__).resolve().parent / "data" / "full"

def min_return_of(rec: dict) -> int:
    frac = rec.get("min_return_frac") or 0.0
    return int(-(-frac * rec["tripled"] // 1)) if frac else 0

def recompute(rec: dict, sent: int, returned: int) -> None:
    E, m = rec["endowment"], rec["multiplier"]
    tripled = m * sent
    rec.update(
        sent=sent,
        tripled=tripled,
        trustee_received=tripled,
        returned=returned,
        return_ratio=(returned / tripled) if tripled > 0 else None,
        investor_net=returned - sent,
        investor_payoff=E - sent + returned,
        trustee_payoff=E + tripled - returned,
        betrayal=bool(tripled > 0 and returned < sent),
        full_betrayal=bool(tripled > 0 and returned == 0),
        reciprocated=bool(tripled > 0 and returned >= sent),
    )

def main() -> None:
    apply = "--apply" in sys.argv
    files = sorted(p for p in DATA.glob("full_*.jsonl") if "attrition" not in p.name)
    if not files:
        sys.exit(f"no data files under {DATA}")

    grand = defaultdict(int)
    print(f"{'model':<26} {'dirty':>6} {'inv_ok':>7} {'inv_bad':>8} {'tru_ok':>7} {'still':>6}")
    print("-" * 66)

    for path in files:
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        dirty = inv_ok = inv_bad = tru_ok = 0
        quarantine = []

        for rec in recs:
            was_dirty = not (rec["investor_parse_clean"] and rec["trustee_parse_clean"])
            if not was_dirty:
                continue
            dirty += 1
            sent, returned = rec["sent"], rec["returned"]
            contaminated = False

            if not rec["investor_parse_clean"]:
                new_sent, new_reason, clean = B.parse_decision(
                    rec.get("investor_raw") or "", 0, rec["endowment"])
                if clean:
                    if new_sent == rec["sent"]:
                        rec["investor_parse_clean"] = True
                        rec["investor_reasoning"] = new_reason or rec.get("investor_reasoning", "")
                        rec["reparsed_investor"] = True
                        inv_ok += 1
                    else:
                        rec["reparse_conflict"] = {"stored": rec["sent"], "reparsed": new_sent}
                        contaminated = True
                        inv_bad += 1

            if not rec["trustee_parse_clean"] and not contaminated:
                lo = min_return_of(rec)
                new_ret, new_reason, clean = B.parse_decision(
                    rec.get("trustee_raw") or "", lo, rec["tripled"])
                if clean:
                    rec["trustee_parse_clean"] = True
                    rec["trustee_reasoning"] = new_reason or rec.get("trustee_reasoning", "")
                    rec["reparsed_trustee"] = True
                    returned = new_ret
                    tru_ok += 1

            if contaminated:
                quarantine.append(rec["dyad_id"])
            else:
                recompute(rec, sent, returned)

        still = sum(1 for r in recs
                    if not (r["investor_parse_clean"] and r["trustee_parse_clean"]))
        name = path.stem.replace("full_", "")
        print(f"{name:<26} {dirty:>6} {inv_ok:>7} {inv_bad:>8} {tru_ok:>7} {still:>6}")
        grand["dirty"] += dirty; grand["inv_ok"] += inv_ok
        grand["inv_bad"] += inv_bad; grand["tru_ok"] += tru_ok; grand["still"] += still

        if apply:
            shutil.copy2(path, path.with_suffix(".jsonl.bak"))
            with path.open("w", encoding="utf-8") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if quarantine:
                q = path.with_name(path.stem + "_quarantine.txt")
                q.write_text("\n".join(quarantine) + "\n")

    print("-" * 66)
    print(f"{'TOTAL':<26} {grand['dirty']:>6} {grand['inv_ok']:>7} "
          f"{grand['inv_bad']:>8} {grand['tru_ok']:>7} {grand['still']:>6}")
    rec_n = grand["inv_ok"] + grand["tru_ok"]
    print(f"\nrecovered {rec_n} role-level decisions; "
          f"{grand['inv_bad']} dyads contaminated (second mover answered a "
          f"different first move) and must be re-run.")
    print("APPLIED, .bak kept" if apply else "\nreport only — pass --apply to write")

if __name__ == "__main__":
    main()
