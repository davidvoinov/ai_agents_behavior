#!/usr/bin/env python3
"""
lit_reference.py — the paradigm reference rate, which is half of H1.

What it counts
--------------
H1 claims that decision transcripts under the canonical wording carry
spontaneous references to the experimental literature at a rate above their
de-recognized counterparts. The measure is a fixed vocabulary of paradigm terms,
namely the names of the paradigm and of its original authors.

The role labels of the canonical surface are EXCLUDED on purpose. A pilot found
that counting them lifted the rate to every canonical dyad while separating no
surfaces, because an agent told it is the Investor writes the word Investor. The
restricted count tracks genuine references to a published study.

The comparison is inside the three structure matched pairs, so the payoffs are
identical and only the wording differs, and it runs on the stated reasoning of
both roles, which is where a reference would appear.

    python3 lit_reference.py
"""
from __future__ import annotations

import re
import sys
import collections
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A

# Names of the paradigm and of the authors of the original study. Nothing here
# appears in any prompt, so a hit is the model supplying the term itself.
PARADIGM = re.compile(
    r"\b(trust game|investment game|berg(?:,| and|\s+et\s+al)|dickhaut|mccabe|"
    r"gift[- ]exchange game|behavio(?:u)?ral econom\w*|experimental econom\w*|"
    r"lab experiment|laboratory experiment|classic(?:al)? experiment|"
    r"standard (?:trust|investment) (?:game|experiment))\b", re.I)
# Generic game theory talk, counted separately. It is not evidence of recalling
# a particular study, and a pilot showed it does not separate the surfaces.
GENERIC = re.compile(
    r"\b(nash equilibrium|subgame perfect|backward induction|dominant strateg\w*|"
    r"game theor\w*|one[- ]shot game|prisoner'?s dilemma)\b", re.I)

LABEL = {"gpt-oss-120b": "gpt-oss-120b", "qwen3-nothink": "Qwen3-32B",
         "gpt-5.6-think": "GPT-5.6 (r)", "gpt-5.6-nothink": "GPT-5.6",
         "sonnet-5-think": "Sonnet-5 (r)", "sonnet-5-nothink": "Sonnet-5",
         "deepseek-v4-flash": "DeepSeek-V4-Flash",
         "deepseek-v4-pro-think": "DeepSeek-V4-Pro (r)",
         "deepseek-v4-pro": "DeepSeek-V4-Pro", "kimi-k3": "Kimi-K3",
         "minimax-m3-think": "MiniMax-M3 (r)", "minimax-m3-nothink": "MiniMax-M3",
         "glm-5.2-think": "GLM-5.2 (r)", "glm-5.2-nothink": "GLM-5.2",
         "llama-3.3-70b": "Llama-3.3-70B"}
ORDER = ["gpt-oss-120b", "deepseek-v4-flash", "deepseek-v4-pro-think", "kimi-k3",
         "deepseek-v4-pro", "minimax-m3-think", "qwen3-nothink",
         "minimax-m3-nothink", "glm-5.2-nothink", "glm-5.2-think", "llama-3.3-70b",
         "gpt-5.6-think", "gpt-5.6-nothink", "sonnet-5-think", "sonnet-5-nothink"]


def reasoning(r: dict) -> str:
    return f"{r.get('investor_reasoning') or ''} {r.get('trustee_reasoning') or ''}"


def main() -> None:
    data = A.load()
    print("Paradigm reference rate, share of dyads whose stated reasoning names\n"
          "the paradigm or its authors. Structure matched pairs, all treatments.\n")
    print(f"{'configuration':<22}{'recognizable':>13}{'de-recognized':>14}"
          f"{'diff pp':>9}{'p':>10}{'generic':>9}")
    print("-" * 78)
    tot = collections.Counter()
    rows = []
    for m in [k for k in ORDER if k in data]:
        rec, der, gen = [], [], []
        for a, b in A.PAIRS:
            for surf, tgt in ((a, rec), (b, der)):
                for r in A.sel(data[m], surf, None):
                    t = reasoning(r)
                    tgt.append(1 if PARADIGM.search(t) else 0)
                    gen.append(1 if GENERIC.search(t) else 0)
        if len(rec) < 100 or len(der) < 100:
            continue
        tab = [[sum(rec), len(rec) - sum(rec)], [sum(der), len(der) - sum(der)]]
        _, p = stats.fisher_exact(tab)
        d = 100 * (np.mean(rec) - np.mean(der))
        rows.append((LABEL[m], np.mean(rec), np.mean(der), d, p, np.mean(gen)))
        tot["n"] += 1
        tot["pos"] += d > 0
        tot["sig"] += p < 0.05 and d > 0
        print(f"{LABEL[m]:<22}{np.mean(rec):>12.1%}{np.mean(der):>14.1%}"
              f"{d:>9.2f}{p:>10.2g}{np.mean(gen):>9.1%}")
    print("-" * 78)
    print(f"\nthe direction H1 predicts, more references under the recognizable "
          f"wording,\nholds in {tot['pos']} of {tot['n']} configurations and "
          f"reaches significance in {tot['sig']}")
    allrec = float(np.mean([r[1] for r in rows]))
    allder = float(np.mean([r[2] for r in rows]))
    print(f"pooled rate {allrec:.2%} under the recognizable wording against "
          f"{allder:.2%} under its counterpart")
    print(f"generic game theory vocabulary appears in "
          f"{np.mean([r[5] for r in rows]):.1%} of dyads on average, and it is "
          f"reported\nseparately because naming a concept is not recalling a study")


if __name__ == "__main__":
    main()
