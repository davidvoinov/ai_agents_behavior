from __future__ import annotations

import json
import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A
import lit_reference as L

HERE = Path(__file__).resolve().parent
ORDER = ["gpt-oss-120b", "qwen3-nothink", "deepseek-v4-flash", "deepseek-v4-pro",
         "deepseek-v4-pro-think", "kimi-k3", "minimax-m3-nothink", "minimax-m3-think",
         "glm-5.2-nothink", "glm-5.2-think", "llama-3.3-70b", "gpt-5.6-nothink",
         "gpt-5.6-think", "sonnet-5-nothink", "sonnet-5-think"]


def holm(ps):
    return multipletests(ps, method="holm")[1]


def variance(data):
    rows = [{"configuration": m, "surface": r["surface"], "treatment": r["condition"],
             "y": A.frac(r)} for m in ORDER for r in data[m]]
    d = pd.DataFrame(rows)
    fit = smf.ols("y ~ C(configuration, Sum) * C(surface, Sum) * C(treatment, Sum)",
                  data=d).fit()
    t = anova_lm(fit, typ=3)
    tot = ((d.y - d.y.mean()) ** 2).sum()
    share = (t["sum_sq"] / tot).drop("Intercept")
    print("\nVARIANCE DECOMPOSITION, transfer fraction, Type III, sum contrasts")
    print(f"  dyads {len(d)}, R2 {fit.rsquared:.3f}")
    for k, v in share.items():
        print(f"  {k:<85}{100 * v:6.1f}%")
    return share, fit.rsquared, len(d)


def wording(data):
    out = []
    for m in ORDER:
        for a, b in A.PAIRS:
            ga = [A.frac(r) for r in A.sel(data[m], a, "no_history")]
            gb = [A.frac(r) for r in A.sel(data[m], b, "no_history")]
            p = stats.mannwhitneyu(ga, gb, alternative="two-sided").pvalue
            out.append((m, f"{a}|{b}", np.mean(ga) - np.mean(gb), p))
    adj = holm([o[3] for o in out])
    print("\nWORDING, 45 contrasts, Holm over the family")
    pos = sum(o[2] > 0 for o in out)
    print(f"  positive {pos}, raw p<.05 positive {sum(o[2] > 0 and o[3] < .05 for o in out)},"
          f" Holm p<.05 positive {sum(o[2] > 0 and q < .05 for o, q in zip(out, adj))}")
    for pair in [f"{a}|{b}" for a, b in A.PAIRS]:
        sel = [(o, q) for o, q in zip(out, adj) if o[1] == pair]
        print(f"  {pair:<32} positive {sum(o[2] > 0 for o, q in sel)}"
              f"  raw sig {sum(o[2] > 0 and o[3] < .05 for o, q in sel)}"
              f"  Holm sig {sum(o[2] > 0 and q < .05 for o, q in sel)}")
    return out, adj


def family(name, items):
    adj = holm([p for _, _, p in items])
    print(f"\n{name}, Holm over {len(items)}")
    for (m, eff, p), q in zip(items, adj):
        print(f"  {m:<24}{eff:+9.3f}  raw {p:9.2g}  Holm {q:9.2g}{'  *' if q < .05 else ''}")
    print(f"  raw sig {sum(p < .05 for _, _, p in items)}, Holm sig {sum(q < .05 for q in adj)}")
    return adj


def payoff_axis(data):
    items = []
    for m in ORDER:
        g = [[A.frac(r) for r in A.sel(data[m], s, "no_history")] for s in A.ANCHORS]
        mu = [np.mean(x) for x in g]
        items.append((m, max(mu) - min(mu), stats.kruskal(*g).pvalue))
    return family("PAYOFF AXIS, Kruskal-Wallis", items)


def history(data):
    items = []
    for m in ORDER:
        p_ = [A.frac(r) for r in A.sel(data[m], None, "social_history_pos")]
        n_ = [A.frac(r) for r in A.sel(data[m], None, "social_history_neg")]
        items.append((m, np.mean(p_) - np.mean(n_),
                      stats.mannwhitneyu(p_, n_, alternative="two-sided").pvalue))
    return family("SOCIAL HISTORY, favorable against unfavorable", items)


def communication(data):
    items = []
    for m in ORDER:
        k = [A.frac(r) for r in A.sel(data[m], None, "communication")]
        b = [A.frac(r) for r in A.sel(data[m], None, "no_history")]
        items.append((m, np.mean(k) - np.mean(b),
                      stats.mannwhitneyu(k, b, alternative="two-sided").pvalue))
    return family("COMMUNICATION against baseline", items)


def toggles(data):
    mean_items, zero_items = [], []
    for t, p in A.TOGGLES:
        a = [A.frac(r) for r in data[t]]
        b = [A.frac(r) for r in data[p]]
        mean_items.append((t, np.mean(a) - np.mean(b),
                           stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
        za, zb = sum(x == 0 for x in a), sum(x == 0 for x in b)
        zero_items.append((t, za / len(a) - zb / len(b),
                           stats.fisher_exact([[za, len(a) - za], [zb, len(b) - zb]])[1]))
    family("TOGGLE, mean transfer", mean_items)
    family("TOGGLE, share transferring nothing", zero_items)


def references(data):
    items = []
    for m in ORDER:
        rec, der = [], []
        for a, b in A.PAIRS:
            rec += [1 if L.PARADIGM.search(L.reasoning(r)) else 0 for r in A.sel(data[m], a, None)]
            der += [1 if L.PARADIGM.search(L.reasoning(r)) else 0 for r in A.sel(data[m], b, None)]
        tab = [[sum(rec), len(rec) - sum(rec)], [sum(der), len(der) - sum(der)]]
        items.append((m, np.mean(rec) - np.mean(der), stats.fisher_exact(tab)[1]))
    return family("PARADIGM REFERENCES, original against reworded wording", items)


def agreements():
    codes = {}
    for line in (HERE / "data" / "coding" / "judge_comm.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("coded_clean"):
                codes[r["uid"]] = r
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in codes.values():
        by[r["model_key"]][r["surface"]].append(1 if r.get("explicit_agreement") else 0)
    items = []
    for m in ORDER:
        rec = sum((by[m][a] for a, _ in A.PAIRS), [])
        der = sum((by[m][b] for _, b in A.PAIRS), [])
        if not rec or not der:
            continue
        tab = [[sum(rec), len(rec) - sum(rec)], [sum(der), len(der) - sum(der)]]
        items.append((m, np.mean(rec) - np.mean(der), stats.fisher_exact(tab)[1]))
    return family("EXPLICIT AGREEMENT, original against reworded wording", items)


def benchmark(data):
    print("\nHUMAN BENCHMARK 0.516, canonical surface, baseline treatment")
    for m in ORDER:
        c = [A.frac(r) for r in A.sel(data[m], "canonical", "no_history")]
        allb = [A.frac(r) for r in A.sel(data[m], None, "no_history")]
        t = stats.ttest_1samp(c, A.BERG_HUMAN_SENT_FRAC)
        print(f"  {m:<24} canonical {np.mean(c):.3f} (n={len(c)})  all surfaces {np.mean(allb):.3f}"
              f"  t-test vs 0.516 p={t.pvalue:.2g}")


def copying(data):
    import re
    xs = re.compile(r"X =\s*(\d+)")
    avg = re.compile(r"average X passed = ([\d.]+)")
    print("\nCOPYING THE REPORTED TRANSFERS, social history treatments")
    out = []
    for m in ORDER:
        n = eq_mean = within1 = in_set = 0
        corr_pairs = []
        prompts = {}
        for f in sorted((HERE / "data" / "batch" / m).glob("decide_investor_attempt*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    q = json.loads(line)
                    prompts[q["custom_id"]] = " ".join(x["content"] for x in q["body"]["messages"])
        for r in data[m]:
            if not r["condition"].startswith("social_history"):
                continue
            pr = r.get("investor_prompt") or prompts.get(r["dyad_id"], "")
            vals = [int(v) for v in xs.findall(pr)]
            a = avg.search(pr)
            if not vals or not a:
                continue
            mu = float(a.group(1))
            n += 1
            eq_mean += r["sent"] == round(mu)
            within1 += abs(r["sent"] - mu) <= 1
            in_set += r["sent"] in set(vals)
            corr_pairs.append((r["condition"], mu / r["endowment"], A.frac(r)))
        rho = []
        for cond in ("social_history_pos", "social_history_neg"):
            cp = [(a, b) for c, a, b in corr_pairs if c == cond]
            if len({a for a, _ in cp}) > 2:
                rho.append(stats.spearmanr([a for a, _ in cp], [b for _, b in cp])[0])
        if not n:
            print(f"  {m:<24} report text not stored")
            continue
        out.append((m, n, eq_mean / n, within1 / n, in_set / n, rho))
        print(f"  {m:<24} n {n:5d}  = rounded report mean {eq_mean / n:6.1%}  within 1 {within1 / n:6.1%}"
              f"  equals some reported X {in_set / n:6.1%}  rho(variant mean, transfer) pos {rho[0]:+.2f} neg {rho[1]:+.2f}")
    return out


def main():
    data = A.load()
    variance(data)
    wording(data)
    payoff_axis(data)
    history(data)
    communication(data)
    toggles(data)
    references(data)
    agreements()
    benchmark(data)
    copying(data)


if __name__ == "__main__":
    main()
