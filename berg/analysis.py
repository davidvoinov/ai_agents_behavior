from __future__ import annotations

import json
import glob
import sys
import collections
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent / "data"
LADDER = ("canonical", "social_nonparadigm", "reskin")
PAIRS = [("canonical", "reskin"), ("shifted", "reskin_shifted"),
         ("strong_control", "strong_novel")]
ANCHORS = ("canonical", "shifted", "strong_control")
CONDITIONS = ("no_history", "social_history_pos", "social_history_neg", "communication")
TOGGLES = [("minimax-m3-think", "minimax-m3-nothink"),
           ("glm-5.2-think", "glm-5.2-nothink"),
           ("deepseek-v4-pro-think", "deepseek-v4-pro"),
           ("gpt-5.6-think", "gpt-5.6-nothink"),
           ("sonnet-5-think", "sonnet-5-nothink")]
EXCLUDE = {"qwen3-think"}

BERG_HUMAN_SENT_FRAC = 5.16 / 10.0
BERG_HUMAN_RETURN = 4.66

def load() -> dict[str, list[dict]]:
    out = collections.defaultdict(dict)
    paths = sorted((DATA / "full").glob("full_*.jsonl"))
    paths += sorted((DATA / "batch").glob("*/full_*.jsonl"))
    for p in paths:
        n = p.name
        if any(x in n for x in ("attrition", "purged", "quarantine", "bak", "prepurge")):
            continue
        model = n[len("full_"):-len(".jsonl")].split("__")[0]
        if model in EXCLUDE:
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["investor_parse_clean"] and r["trustee_parse_clean"]:
                out[model].setdefault(r["dyad_id"], r)
    return {m: list(d.values()) for m, d in out.items()}

def frac(r: dict) -> float:
    return r["sent"] / r["endowment"]

def sel(rows, surface=None, condition=None):
    return [r for r in rows
            if (surface is None or r["surface"] == surface)
            and (condition is None or r["condition"] == condition)]

def boot_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    d = [rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
         for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))

def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return (gt - lt) / (len(a) * len(b))

def contrast(a, b, label="") -> dict:
    a, b = list(a), list(b)
    if len(a) < 3 or len(b) < 3:
        return {"label": label, "n1": len(a), "n2": len(b), "delta": float("nan"),
                "p": float("nan"), "d": float("nan"), "cliff": float("nan"),
                "ci": (float("nan"), float("nan"))}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = np.sqrt(((len(a) - 1) * s1**2 + (len(b) - 1) * s2**2) / (len(a) + len(b) - 2))
    return {"label": label, "n1": len(a), "n2": len(b), "m1": m1, "m2": m2,
            "delta": m1 - m2, "p": float(p),
            "d": (m1 - m2) / pooled if pooled > 0 else float("nan"),
            "cliff": cliffs_delta(a, b), "ci": boot_ci(a, b)}

def fmt(c: dict) -> str:
    ci = c["ci"]
    return (f"Δ={c['delta']:+.3f}  d={c['d']:+.2f}  δ={c['cliff']:+.2f}  "
            f"p={c['p']:.2g}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
            f"n={c['n1']}/{c['n2']}")

def sec_coverage(data):
    print("\n" + "=" * 78)
    print("1. COVERAGE")
    print("=" * 78)
    print(f"{'model':<24}{'dyads':>7}{'cells':>7}{'mean frac':>11}{'zeros':>8}{'mean ret':>10}")
    print("-" * 78)
    tot = 0
    for m in sorted(data):
        rows = data[m]
        tot += len(rows)
        cells = len({(r["surface"], r["condition"]) for r in rows})
        f = [frac(r) for r in rows]
        z = sum(1 for r in rows if r["sent"] == 0) / len(rows)
        rr = [r["return_ratio"] for r in rows if r["return_ratio"] is not None]
        print(f"{m:<24}{len(rows):>7}{cells:>7}{np.mean(f):>11.3f}{z:>7.1%}"
              f"{(np.mean(rr) if rr else float('nan')):>10.3f}")
    print("-" * 78)
    print(f"{'TOTAL':<24}{tot:>7}")
    print(f"\nBerg human benchmark, no history: transfer fraction {BERG_HUMAN_SENT_FRAC:.3f}, "
          f"return ${BERG_HUMAN_RETURN:.2f}")

def sec_ladder(data):
    print("\n" + "=" * 78)
    print("2. LABEL LADDER (canonical payoffs, wording varies), baseline treatment")
    print("   social meaning account -> canonical ~ social_nonparadigm > reskin")
    print("   paradigm identity account -> canonical > social_nonparadigm ~ reskin")
    print("=" * 78)
    print(f"{'model':<24}{'canon':>8}{'social':>8}{'reskin':>8}   "
          f"{'canon-reskin':>26}{'  verdict':>10}")
    print("-" * 92)
    res = {}
    for m in sorted(data):
        g = {s: [frac(r) for r in sel(data[m], s, "no_history")] for s in LADDER}
        if min(len(v) for v in g.values()) < 20:
            continue
        c = contrast(g["canonical"], g["reskin"])
        c_soc = contrast(g["canonical"], g["social_nonparadigm"])
        s_res = contrast(g["social_nonparadigm"], g["reskin"])
        verdict = "identity" if (c_soc["p"] < 0.05 and s_res["p"] >= 0.05) else \
                  ("social" if (s_res["p"] < 0.05 and c_soc["p"] >= 0.05) else "mixed")
        res[m] = {"means": {s: float(np.mean(v)) for s, v in g.items()},
                  "canon_vs_reskin": c, "canon_vs_social": c_soc,
                  "social_vs_reskin": s_res, "verdict": verdict}
        print(f"{m:<24}{np.mean(g['canonical']):>8.3f}{np.mean(g['social_nonparadigm']):>8.3f}"
              f"{np.mean(g['reskin']):>8.3f}   Δ={c['delta']:+.3f} p={c['p']:.2g}"
              f"   {verdict:>8}")
    return res

def sec_pairs(data):
    print("\n" + "=" * 78)
    print("3. WORDING EFFECT, structure matched pairs, baseline treatment")
    print("=" * 78)
    res = collections.defaultdict(dict)
    for m in sorted(data):
        line, any_row = [], False
        for a, b in PAIRS:
            ga = [frac(r) for r in sel(data[m], a, "no_history")]
            gb = [frac(r) for r in sel(data[m], b, "no_history")]
            if len(ga) < 20 or len(gb) < 20:
                line.append(f"{a[:9]}/{b[:9]}: --")
                continue
            c = contrast(ga, gb)
            res[m][f"{a}|{b}"] = c
            star = "*" if c["p"] < 0.05 else " "
            line.append(f"{a[:6]}/{b[:6]} Δ={c['delta']:+.3f}{star}")
            any_row = True
        if any_row:
            print(f"  {m:<24}" + "   ".join(line))
    n_pos = sum(1 for m in res for k, c in res[m].items() if c["delta"] > 0)
    n_all = sum(len(v) for v in res.values())
    n_sig = sum(1 for m in res for k, c in res[m].items() if c["p"] < 0.05)
    print(f"\n  recognizable wording draws the higher transfer in {n_pos} of {n_all} "
          f"contrasts, {n_sig} of them at p<0.05")
    return res

def sec_incentive(data):
    print("\n" + "=" * 78)
    print("4. INCENTIVE AXIS, canonical wording, payoffs vary (E,m): "
          "(10,3) -> (12,2) -> (20,4)")
    print("   the optimum is a zero transfer at every anchor, so a flat response")
    print("   here is uninformative on its own and needs the positive control")
    print("=" * 78)
    res = {}
    print(f"{'model':<24}{'canon(10,3)':>12}{'shift(12,2)':>13}{'ctrl(20,4)':>12}"
          f"{'   spread':>10}{'  KW p':>9}")
    print("-" * 82)
    for m in sorted(data):
        g = {s: [frac(r) for r in sel(data[m], s, "no_history")] for s in ANCHORS}
        if min(len(v) for v in g.values()) < 20:
            continue
        h, p = stats.kruskal(*[g[s] for s in ANCHORS])
        mu = {s: float(np.mean(v)) for s, v in g.items()}
        spread = max(mu.values()) - min(mu.values())
        res[m] = {"means": mu, "kruskal_p": float(p), "spread": spread}
        print(f"{m:<24}{mu['canonical']:>12.3f}{mu['shifted']:>13.3f}"
              f"{mu['strong_control']:>12.3f}{spread:>10.3f}{p:>9.2g}")
    return res

def sec_positive_control():
    print("\n" + "=" * 78)
    print("5. POSITIVE CONTROL, enforced minimum return k")
    print("   payoff E + X(km-1) with m=3: optimum is 0 at k=0, indifferent at k=1/3,")
    print("   the whole endowment at k=2/3. An agent that reads incentives MUST move.")
    print("=" * 78)
    res = {}
    for p in sorted((DATA / "incentive").glob("incentive_*.jsonl")):
        if "attrition" in p.name:
            continue
        m = p.stem[len("incentive_"):]
        by = collections.defaultdict(list)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["investor_parse_clean"] and r["trustee_parse_clean"]:
                by[round(r.get("min_return_frac") or 0.0, 3)].append(frac(r))
        if len(by) < 2:
            continue
        ks = sorted(by)
        res[m] = {"means": {str(k): float(np.mean(v)) for k, v in by.items()},
                  "n": {str(k): len(v) for k, v in by.items()}}
        cells = "  ".join(f"k={k:.2f}: {np.mean(by[k]):.3f} (n={len(by[k])})" for k in ks)
        print(f"  {m:<22} {cells}")
        if len(ks) >= 2:
            c = contrast(by[ks[-1]], by[ks[0]])
            res[m]["extreme"] = c
            print(f"  {'':<22} k={ks[-1]:.2f} vs k={ks[0]:.2f}: {fmt(c)}")
    return res

def sec_toggle(data):
    print("\n" + "=" * 78)
    print("6. REASONING TOGGLE, same weights, thinking on vs off")
    print("=" * 78)
    res = {}
    for think, plain in TOGGLES:
        if think not in data or plain not in data:
            continue
        shared = {(r["surface"], r["condition"]) for r in data[think]} & \
                 {(r["surface"], r["condition"]) for r in data[plain]}
        ok = set()
        for cell in shared:
            a = [r for r in data[think] if (r["surface"], r["condition"]) == cell]
            b = [r for r in data[plain] if (r["surface"], r["condition"]) == cell]
            if len(a) >= 50 and len(b) >= 50:
                ok.add(cell)
        ga = [frac(r) for r in data[think] if (r["surface"], r["condition"]) in ok]
        gb = [frac(r) for r in data[plain] if (r["surface"], r["condition"]) in ok]
        if len(ga) < 50 or len(gb) < 50:
            continue
        c = contrast(ga, gb)
        res[think] = {"pair": plain, "cells": len(ok), "contrast": c}
        print(f"  {think:<24} vs {plain:<22} over {len(ok):>2} matched cells")
        print(f"  {'':<24}    {fmt(c)}")
    return res

def sec_history(data):
    print("\n" + "=" * 78)
    print("7. SOCIAL HISTORY VALENCE, positive report vs negative report")
    print("   pooled over surfaces, baseline shown for reference")
    print("=" * 78)
    res = {}
    print(f"{'model':<24}{'base':>8}{'pos':>8}{'neg':>8}{'   pos-neg contrast':>34}")
    print("-" * 84)
    for m in sorted(data):
        b = [frac(r) for r in sel(data[m], None, "no_history")]
        p_ = [frac(r) for r in sel(data[m], None, "social_history_pos")]
        n_ = [frac(r) for r in sel(data[m], None, "social_history_neg")]
        if min(len(b), len(p_), len(n_)) < 50:
            continue
        c = contrast(p_, n_)
        res[m] = {"base": float(np.mean(b)), "pos": float(np.mean(p_)),
                  "neg": float(np.mean(n_)), "contrast": c,
                  "vs_base_pos": contrast(p_, b), "vs_base_neg": contrast(n_, b)}
        print(f"{m:<24}{np.mean(b):>8.3f}{np.mean(p_):>8.3f}{np.mean(n_):>8.3f}"
              f"   Δ={c['delta']:+.3f} d={c['d']:+.2f} p={c['p']:.2g}")
    return res

def sec_comm(data):
    print("\n" + "=" * 78)
    print("8. COMMUNICATION vs BASELINE")
    print("=" * 78)
    res = {}
    print(f"{'model':<24}{'base':>8}{'comm':>8}{'   contrast':>30}{'  comm zeros':>13}")
    print("-" * 86)
    for m in sorted(data):
        b = [frac(r) for r in sel(data[m], None, "no_history")]
        k = [frac(r) for r in sel(data[m], None, "communication")]
        if min(len(b), len(k)) < 50:
            continue
        c = contrast(k, b)
        z = sum(1 for x in k if x == 0) / len(k)
        res[m] = {"base": float(np.mean(b)), "comm": float(np.mean(k)),
                  "contrast": c, "comm_zero_rate": z}
        print(f"{m:<24}{np.mean(b):>8.3f}{np.mean(k):>8.3f}"
              f"   Δ={c['delta']:+.3f} d={c['d']:+.2f} p={c['p']:.2g}{z:>12.1%}")
    return res

def sec_reciprocity(data):
    print("\n" + "=" * 78)
    print("9. SECOND MOVER, return as a fraction of the amount received")
    print("=" * 78)
    res = {}
    print(f"{'model':<24}{'overall':>9}{'base':>8}{'pos':>8}{'neg':>8}{'comm':>8}"
          f"{'  full betrayal':>16}")
    print("-" * 82)
    for m in sorted(data):
        rows = [r for r in data[m] if r["tripled"] > 0]
        if len(rows) < 100:
            continue
        def rr(cond=None):
            v = [r["return_ratio"] for r in rows
                 if (cond is None or r["condition"] == cond) and r["return_ratio"] is not None]
            return float(np.mean(v)) if v else float("nan")
        fb = sum(1 for r in rows if r["full_betrayal"]) / len(rows)
        res[m] = {"overall": rr(), "no_history": rr("no_history"),
                  "pos": rr("social_history_pos"), "neg": rr("social_history_neg"),
                  "comm": rr("communication"), "full_betrayal": fb, "n": len(rows)}
        print(f"{m:<24}{rr():>9.3f}{rr('no_history'):>8.3f}{rr('social_history_pos'):>8.3f}"
              f"{rr('social_history_neg'):>8.3f}{rr('communication'):>8.3f}{fb:>15.1%}")
    return res

def sec_attrition(data):
    import attrition as ATT
    print("\n" + "=" * 78)
    print("10. ATTRITION, and its association with the treatment")
    print("=" * 78)
    res = {}
    for p in sorted((DATA / "full").glob("full_*.jsonl")):
        n = p.name
        if any(x in n for x in ("attrition", "purged", "quarantine", "bak", "prepurge")):
            continue
        if "__" in n:
            continue
        m = n[len("full_"):-len(".jsonl")]
        arch = p.with_name(p.stem + "_purged.jsonl")
        by = collections.defaultdict(lambda: collections.Counter())
        for line in p.read_text().splitlines():
            if line.strip():
                by[json.loads(line)["condition"]]["ok"] += 1
        if arch.exists():
            for line in arch.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                worst = "ok"
                for role in ("investor", "trustee"):
                    if not r[f"{role}_parse_clean"]:
                        cause = ATT.classify_completion(r.get(f"{role}_raw") or "", False)
                        if ATT.category(cause) == ATT.BEHAV:
                            worst = "behav"
                        elif worst != "behav":
                            worst = "infra"
                by[r["condition"]][worst] += 1
        rows = {c: dict(by[c]) for c in CONDITIONS if by[c]}
        if not rows:
            continue
        tab = []
        for c in ("social_history_pos", "social_history_neg"):
            d = by[c]
            tab.append([d["ok"], d["infra"] + d["behav"]])
        assoc = None
        if len(tab) == 2 and all(sum(r) > 0 for r in tab) and sum(r[1] for r in tab) > 0:
            try:
                _, pv = stats.fisher_exact(tab)
                assoc = float(pv)
            except Exception:
                assoc = None
        loss = {c: (by[c]["infra"] + by[c]["behav"]) /
                   max(1, sum(by[c].values())) for c in CONDITIONS if by[c]}
        res[m] = {"by_condition": rows, "loss": loss, "valence_assoc_p": assoc}
        if max(loss.values(), default=0) > 0.05:
            s = "  ".join(f"{c[:9]}={v:.1%}" for c, v in loss.items())
            extra = f"   Fisher p={assoc:.2g}" if assoc is not None else ""
            print(f"  {m:<24} {s}{extra}")
    print("\n  models not listed lose under 5% in every condition")
    return res

def sec_sensitivity(data):
    print("\n" + "=" * 78)
    print("11. SENSITIVITY, worst case imputation for every lost dyad")
    print("=" * 78)
    res = {}
    print(f"{'model':<24}{'observed':>10}{'bounded':>10}{'bounded p':>12}"
          f"{'lost +':>8}{'lost -':>8}")
    print("-" * 72)
    for m in sorted(data):
        p_ = [frac(r) for r in sel(data[m], None, "social_history_pos")]
        n_ = [frac(r) for r in sel(data[m], None, "social_history_neg")]
        if min(len(p_), len(n_)) < 50:
            continue
        lost = collections.Counter()
        arch = DATA / "full" / f"full_{m}_purged.jsonl"
        if arch.exists():
            for line in arch.read_text().splitlines():
                if line.strip():
                    lost[json.loads(line)["condition"]] += 1
        lp, ln = lost["social_history_pos"], lost["social_history_neg"]
        pb, nb = p_ + [0.0] * lp, n_ + [1.0] * ln
        _, pv = stats.mannwhitneyu(pb, nb, alternative="two-sided")
        obs, bnd = float(np.mean(p_) - np.mean(n_)), float(np.mean(pb) - np.mean(nb))
        res[m] = {"observed": obs, "bounded": bnd, "bounded_p": float(pv),
                  "lost_pos": lp, "lost_neg": ln, "survives": bool(pv < 1e-5)}
        flag = "" if pv < 1e-5 else "   DOES NOT SURVIVE"
        print(f"{m:<24}{obs:>10.3f}{bnd:>10.3f}{pv:>12.2g}{lp:>8}{ln:>8}{flag}")
    return res

def main() -> None:
    data = load()
    as_json = "--json" in sys.argv
    if as_json:
        buf = []
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            out = {"coverage": None}
            sec_coverage(data)
            out["ladder"] = sec_ladder(data)
            out["pairs"] = {m: {k: v for k, v in d.items()} for m, d in sec_pairs(data).items()}
            out["incentive"] = sec_incentive(data)
            out["positive_control"] = sec_positive_control()
            out["toggle"] = sec_toggle(data)
            out["history"] = sec_history(data)
            out["comm"] = sec_comm(data)
            out["reciprocity"] = sec_reciprocity(data)
            out["attrition"] = sec_attrition(data)
            out["sensitivity"] = sec_sensitivity(data)
        print(json.dumps(out, indent=1, default=float))
        return
    sec_coverage(data)
    sec_ladder(data)
    sec_pairs(data)
    sec_incentive(data)
    sec_positive_control()
    sec_toggle(data)
    sec_history(data)
    sec_comm(data)
    sec_reciprocity(data)
    sec_attrition(data)
    sec_sensitivity(data)

if __name__ == "__main__":
    main()
