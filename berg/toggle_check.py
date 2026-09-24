from __future__ import annotations

import json
import re
import collections
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent / "data" / "full"
BATCH = Path(__file__).resolve().parent / "data" / "batch"
PAIRS = [("glm-5.2-think", "glm-5.2-nothink"),
         ("deepseek-v4-pro-think", "deepseek-v4-pro"),
         ("minimax-m3-think", "minimax-m3-nothink"),
         ("gpt-5.6-think", "gpt-5.6-nothink"),
         ("sonnet-5-think", "sonnet-5-nothink")]
SEED = 20260904

EV = re.compile(r"\b(expected value|expected payoff|rational|self-interest\w*|"
                r"equilibrium|dominant strateg\w*|backward\w*|maximi[sz]\w*|"
                r"one-shot|no future|no reason to return)\b", re.I)
SOC = re.compile(r"\b(fair\w*|trust\w*|reciprocat\w*|cooperat\w*|goodwill|"
                 r"mutual\w*|generous\w*|split)\b", re.I)

def load(model: str) -> list[dict]:
    p = DATA / f"full_{model}.jsonl"
    if not p.exists():
        p = BATCH / model / f"full_{model}.jsonl"
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["investor_parse_clean"] and r["trustee_parse_clean"]:
            out[r["dyad_id"]] = r
    return list(out.values())

def features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = ["transfer_frac", "return_ratio", "is_zero", "is_all",
             "inv_reason_chars", "tru_reason_chars",
             "inv_raw_chars", "tru_raw_chars",
             "inv_ev_terms", "inv_social_terms", "inv_digits"]
    X = []
    for r in rows:
        ir = r.get("investor_reasoning") or ""
        tr = r.get("trustee_reasoning") or ""
        X.append([
            r["sent"] / r["endowment"],
            r["return_ratio"] if r["return_ratio"] is not None else 0.0,
            float(r["sent"] == 0),
            float(r["sent"] == r["endowment"]),
            len(ir), len(tr),
            len(r.get("investor_raw") or ""), len(r.get("trustee_raw") or ""),
            len(EV.findall(ir)), len(SOC.findall(ir)),
            len(re.findall(r"\d", ir)),
        ])
    return np.asarray(X, float), names

def auc(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))

def main() -> None:
    rng = np.random.default_rng(SEED)
    print("Cross-validated AUC for telling the two arms apart.")
    print("0.50 is indistinguishable. The placebo column is what a toggle that")
    print("did nothing would produce at this sample size.\n")
    print(f"{'family':<20}{'observed':>10}{'placebo':>10}{'permuted':>10}"
          f"{'  n':>8}   reading")
    print("-" * 74)

    detail = {}
    for a, b in PAIRS:
        ra, rb = load(a), load(b)
        Xa, names = features(ra)
        Xb, _ = features(rb)
        X = np.vstack([Xa, Xb])
        y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
        obs = auc(X, y)

        idx = rng.permutation(len(Xb))
        half = len(idx) // 2
        Xp = np.vstack([Xb[idx[:half]], Xb[idx[half:2 * half]]])
        yp = np.r_[np.ones(half), np.zeros(half)]
        plac = auc(Xp, yp)

        perm = auc(X, rng.permutation(y))

        margin = obs - plac
        reading = ("separates" if margin > 0.10 else
                   "weak" if margin > 0.03 else "NOT SEPARABLE")
        print(f"{a.split('-think')[0][:19]:<20}{obs:>10.3f}{plac:>10.3f}"
              f"{perm:>10.3f}{len(X):>8}   {reading}")
        detail[a] = (X, y, names, obs, plac)

    print("\nfeature by feature, standardized mean difference (reasoning minus plain):")
    print(f"{'feature':<20}" + "".join(f"{a.split('-')[0][:11]:>12}" for a, _ in PAIRS))
    print("-" * 60)
    names = detail[PAIRS[0][0]][2]
    for k, nm in enumerate(names):
        row = []
        for a, _ in PAIRS:
            X, y, _, _, _ = detail[a]
            v1, v0 = X[y == 1, k], X[y == 0, k]
            sd = np.sqrt((v1.var(ddof=1) + v0.var(ddof=1)) / 2)
            row.append((v1.mean() - v0.mean()) / sd if sd > 0 else 0.0)
        print(f"{nm:<20}" + "".join(f"{v:>12.3f}" for v in row))

    print("\nA family whose arms cannot be told apart above the placebo has no\n"
          "evidence that its parameter reached the model, whatever the p value on\n"
          "any single mean. Confirming it needs a live probe on the reasoning\n"
          "field and the completion token count, which diag_sweep.py performs.")

if __name__ == "__main__":
    main()
