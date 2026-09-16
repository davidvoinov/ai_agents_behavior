#!/usr/bin/env python3
"""
cost_estimate.py — what a configuration costs before it is run.

Token volumes are measured from the completed confirmatory data rather than
guessed. Prompt sizes come from the live prompt builders, so a change to the
stimulus changes the estimate. The one quantity that cannot be measured from
stored rows is hidden reasoning, which GPT-5.6 and Claude Sonnet 5 bill as
output without returning, so it enters as an explicit multiplier the smoke test
is meant to pin down.

    python3 cost_estimate.py                    # default multipliers
    python3 cost_estimate.py --reasoning 6      # 6 hidden tokens per visible one
    python3 cost_estimate.py --dyads 50         # a half-size design

Every price is USD per 1M tokens, batch prices already halved where the
provider offers the discount.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import berg_trust_game as B
import reskin as R

CH = 4.0                      # characters per token, the usual English ratio
SURFACES = 7
TREAT_CELLS = 4               # baseline, two valences of report, communication
COMM_ROUNDS = 2               # messages per side before the private decisions

# (input, output) USD per 1M tokens at list price. Batch halves both.
PRICES = {
    "gpt-5.6-luna":    (0.20, 1.20),
    "claude-sonnet-5": (2.00, 10.00),
}

# Hidden reasoning per visible output token, by arm, MEASURED on full dyads
# rather than assumed. GPT-5.6 at medium billed 1,080 reasoning tokens against
# 402 visible over ten calls. Claude Sonnet 5 at xhigh drew 384 output tokens per
# call against 227 with thinking off.
#
# The Sonnet figure is lower than the 1.19 a single investor call gave, because
# adaptive thinking fired on 6 of the 10 calls in the dyad and on 3 of 3 in the
# isolated sweep. The arm is therefore a mixture, and 0.70 is the mixture's
# average, which is the right number for a budget.
ARMS = {
    "gpt-5.6 think":    ("gpt-5.6-luna",    "reasoning_effort=medium",     2.69),
    "gpt-5.6 plain":    ("gpt-5.6-luna",    "reasoning_effort=none",       0.00),
    "sonnet-5 think":   ("claude-sonnet-5", "thinking adaptive, xhigh",    0.70),
    "sonnet-5 plain":   ("claude-sonnet-5", "thinking disabled, xhigh",    0.00),
}

# Claude Sonnet 5 ships a new tokenizer that turns the same text into about 30%
# more tokens, and the smoke test measured 378 input tokens per call against 256
# for GPT-5.6 on the identical prompt, a ratio of 1.48. Token volumes here are
# counted once in GPT terms and scaled per model, since a count taken against
# one tokenizer cannot be reused against the other.
TOKENIZER = {"gpt-5.6-luna": 1.00, "claude-sonnet-5": 1.48}


def measured() -> dict:
    """Per-dyad visible token volumes, averaged over the completed runs."""
    acc = {"answer": [], "reason": [], "user_plain": [], "user_hist": [],
           "user_comm": [], "transcript": []}
    for f in sorted(Path("data/full").glob("full_*.jsonl")):
        if "attrition" in f.name:
            continue
        n = {"no_history": 0, "social_history": 0, "communication": 0}
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if not (r.get("investor_parse_clean") and r.get("trustee_parse_clean")):
                continue
            t = r.get("treatment")
            if t not in n or n[t] >= 300:
                continue
            n[t] += 1
            L = lambda k: len(r.get(k) or "") / CH                  # noqa: E731
            acc["answer"].append(L("investor_raw") + L("trustee_raw"))
            acc["reason"].append(L("investor_reasoning") + L("trustee_reasoning"))
            u = (L("investor_prompt") + L("trustee_prompt")) / 2
            key = {"no_history": "user_plain", "social_history": "user_hist",
                   "communication": "user_comm"}[t]
            acc[key].append(u)
            if t == "communication":
                tr = r.get("transcript")
                tr = tr if isinstance(tr, str) else (json.dumps(tr) if tr else "")
                acc["transcript"].append(len(tr) / CH)
    return {k: statistics.mean(v) for k, v in acc.items() if v}


def system_tokens() -> tuple[float, float]:
    plain, comm = [], []
    for cfg in R.RESKINS.values():
        plain.append((len(B.investor_system_prompt(cfg, False))
                      + len(B.trustee_system_prompt(cfg, False))) / 2 / CH)
        comm.append((len(B.investor_system_prompt(cfg, True))
                     + len(B.trustee_system_prompt(cfg, True))) / 2 / CH)
    return statistics.mean(plain), statistics.mean(comm)


def volumes(m: dict, dyads: int, reasoning_mult: float) -> tuple[float, float]:
    """Total input and output tokens for one configuration, one arm."""
    sys_plain, sys_comm = system_tokens()
    cells = SURFACES * dyads

    # Baseline and the two report cells: one call per role, each carrying the
    # system prompt and its own user message.
    plain_in = 2 * sys_plain + 2 * m["user_plain"]
    hist_in = 2 * sys_plain + 2 * m["user_hist"]

    # Communication: COMM_ROUNDS messages per side, then the two decisions. Each
    # message call re-sends the system prompt and the transcript so far, so the
    # accumulated transcript is charged once per remaining call.
    calls = 2 * COMM_ROUNDS
    tr = m["transcript"]
    comm_in = ((calls + 2) * sys_comm + (calls + 2) * m["user_comm"]
               + tr * (calls - 1) / 2 + 2 * tr)

    total_in = cells * (plain_in + 2 * hist_in + comm_in)

    visible_out = m["answer"] + m["reason"]
    out_plain = visible_out * (1 + reasoning_mult)
    out_comm = (visible_out + tr) * (1 + reasoning_mult)
    total_out = cells * (3 * out_plain + out_comm)
    return total_in, total_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyads", type=int, default=100,
                    help="dyads per cell, 100 matches the confirmatory design")
    ap.add_argument("--reasoning", type=float, default=None,
                    help="hidden reasoning tokens per visible output token, "
                         "overriding the per-arm defaults in ARMS")
    a = ap.parse_args()

    m = measured()
    sp, sc = system_tokens()
    ndyads = SURFACES * TREAT_CELLS * a.dyads

    print("measured from the completed runs, tokens per dyad")
    print(f"  system prompt      {sp:>7.0f} plain   {sc:>7.0f} with communication")
    print(f"  user message       {m['user_plain']:>7.0f} plain   "
          f"{m['user_hist']:>7.0f} report   {m['user_comm']:>7.0f} communication")
    print(f"  visible answer     {m['answer']:>7.0f}")
    print(f"  visible reasoning  {m['reason']:>7.0f}")
    print(f"  message transcript {m['transcript']:>7.0f}")
    print(f"\none arm is {SURFACES} surfaces x {TREAT_CELLS} cells x {a.dyads} "
          f"dyads = {ndyads} dyads\n")

    print(f"{'arm':<24}{'toggle':<24}{'in M':>7}{'out M':>8}"
          f"{'list $':>10}{'batch $':>10}")
    print("-" * 83)
    total_list = total_batch = 0.0
    for name, (model, toggle, mult) in ARMS.items():
        if a.reasoning is not None and mult > 0:
            mult = a.reasoning                  # a measurement overrides the guess
        ti, to = volumes(m, a.dyads, mult)
        k = TOKENIZER[model]
        ti, to = ti * k, to * k
        pi, po = PRICES[model]
        cost = ti / 1e6 * pi + to / 1e6 * po
        total_list += cost
        total_batch += cost / 2
        print(f"{name:<24}{toggle:<24}{ti/1e6:>7.2f}{to/1e6:>8.2f}"
              f"{cost:>10.2f}{cost/2:>10.2f}")
    print("-" * 83)
    print(f"{'four arms':<48}{'':>15}{total_list:>10.2f}{total_batch:>10.2f}")

    print("\nsensitivity to the hidden reasoning multiplier, batch price, "
          "four arms")
    for r in (1.0, 2.0, 4.0, 8.0):
        s = 0.0
        for name, (model, _t, mult) in ARMS.items():
            mm = 0.0 if mult == 0.0 else r          # the off switch stays off
            ti, to = volumes(m, a.dyads, mm)
            k = TOKENIZER[model]
            pi, po = PRICES[model]
            s += (ti * k / 1e6 * pi + to * k / 1e6 * po) / 2
        print(f"  {r:>3.0f}x hidden reasoning   {s:>8.2f} $")

    sm = 0.0
    for mo, _t, r0 in ARMS.values():
        r0 = a.reasoning if (a.reasoning is not None and r0 > 0) else r0
        si, so = volumes(m, 5, r0)
        k = TOKENIZER[mo] / SURFACES / TREAT_CELLS
        sm += si * k / 1e6 * PRICES[mo][0] + so * k / 1e6 * PRICES[mo][1]
    print(f"\nsmoke test, 5 dyads per arm, one surface, one treatment, list "
          f"price, all four arms\n  about {sm:.2f} $")


if __name__ == "__main__":
    main()
