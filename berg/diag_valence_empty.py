#!/usr/bin/env python3
"""
diag_valence_empty.py — why do positive-history cells lose so many replies?

The finding this probes
-----------------------
Failure to return a parseable decision tracks the VALENCE of the social history
report, not the length of the prompt. Measured within a single hour, so the
state of the provider is held fixed:

    deepseek-v4-flash        positive 52.3%   negative 31.8%
    deepseek-v4-pro-think    positive 35.6%   negative  5.4%

Both reports are the same length to within a character, and the communication
condition has a prompt three times longer with a far lower failure rate. So the
usual suspects are ruled out. Attrition that tracks the manipulated factor is
the kind that biases every conditional estimate, which makes the mechanism worth
one cheap probe before the paper reports the effect of valence at all.

What it separates
-----------------
    content empty + reasoning long   -> the budget went to hidden reasoning,
                                        so raising max_tokens fixes it
    content empty + reasoning empty  -> the provider returned nothing, which is
                                        infrastructure and should be retried
    content long, unparseable        -> the model really does answer differently
                                        under a positive report, which is a
                                        finding and must be reported, not fixed
    finish_reason == 'length'        -> plain truncation

Run it where the API is reachable:
    python3 diag_valence_empty.py [model_key] [n_per_arm]
"""
from __future__ import annotations

import os
import sys
import collections
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

import berg_trust_game as B
import reskin as R
import providers as P
from run_full_cycle import valence_history_report

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-pro-think"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def main() -> None:
    from openai import OpenAI
    mcfg = P.model_config(MODEL, max_tokens=4000)
    client = OpenAI(base_url=mcfg.base_url,
                    api_key=os.environ.get(mcfg.api_key_env, "MISSING"))
    cfg = R.RESKINS["canonical"]
    print(f"{MODEL} via {mcfg.base_url}  n={N} per arm  max_tokens={mcfg.max_tokens}\n")

    for valence in ("positive", "negative"):
        stat = collections.Counter()
        c_len, r_len, comp = [], [], []
        for i in range(N):
            report = valence_history_report(cfg, valence, i % 10)
            sys_p = B.investor_system_prompt(cfg)
            usr_p = B.investor_user_prompt(cfg, history_report=report)
            kw = dict(model=mcfg.model, temperature=mcfg.temperature,
                      max_tokens=mcfg.max_tokens,
                      messages=[{"role": "system", "content": sys_p},
                                {"role": "user", "content": usr_p}])
            if mcfg.extra_body:
                kw["extra_body"] = mcfg.extra_body
            try:
                resp = client.chat.completions.create(**kw)
            except Exception as e:
                stat[f"ERROR {type(e).__name__}"] += 1
                continue
            ch = resp.choices[0]
            content = ch.message.content or ""
            reasoning = (getattr(ch.message, "reasoning", None)
                         or getattr(ch.message, "reasoning_content", None) or "")
            c_len.append(len(content)); r_len.append(len(reasoning))
            comp.append(getattr(resp.usage, "completion_tokens", 0) or 0)
            stat[f"finish={ch.finish_reason}"] += 1
            parsed = B._extract_json(content) is not None
            if not content.strip():
                stat["EMPTY content, reasoning>0" if reasoning else "EMPTY both"] += 1
            elif not parsed:
                stat["unparseable"] += 1
            else:
                stat["ok"] += 1

        n = max(1, len(c_len))
        print(f"--- {valence} ---")
        print(f"   {dict(stat)}")
        print(f"   mean content {sum(c_len)/n:.0f} ch | mean reasoning "
              f"{sum(r_len)/n:.0f} ch | mean completion tokens {sum(comp)/n:.0f}\n")

    print("read the result against the four cases in this file's docstring")


if __name__ == "__main__":
    main()
