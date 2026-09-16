#!/usr/bin/env python3
"""
diag_sonnet_thinking.py — at what effort does adaptive thinking actually engage?

The smoke test found the two Claude Sonnet 5 arms indistinguishable, at 170
output tokens per call with thinking adaptive against 158 with thinking
disabled. The GPT-5.6 pair separated cleanly over the same prompt, at 148
against 40 with 1,080 billed reasoning tokens, so the stimulus is not the
problem.

That smoke test also read zero thinking characters in both arms, which proved
nothing: `display` defaults to "omitted" on this model, so a thinking block
arrives with an empty text field whether or not Claude thought. This diagnostic
asks for "summarized" and counts blocks instead of characters.

The likely reading is that adaptive thinking judged the decision too easy to
think about at medium effort. The documentation is explicit that at lower effort
Claude can skip thinking entirely on simpler problems, and a one-shot transfer
with an integer answer is about as simple as a decision gets. If that is right,
the arms separate once effort is high enough for adaptive to engage, and the
toggle is usable at that setting.

The competing reading is that the parameter never took effect. The two are told
apart by whether output tokens move with effort in the adaptive arm at all. A
parameter that never arrived leaves a flat line across every level.

Results are written to data/diag_sonnet_thinking.jsonl as each cell finishes,
so the run can be inspected afterwards instead of living only in a terminal.

    python3 diag_sonnet_thinking.py
    python3 diag_sonnet_thinking.py --reps 5 --efforts low,medium,high,xhigh
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import berg_trust_game as B
import reskin as R

MODEL = "claude-sonnet-5"
PRICE_IN, PRICE_OUT = 2.00, 10.00
OUT = Path(__file__).resolve().parent / "data" / "diag_sonnet_thinking.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--efforts", default="low,medium,high,xhigh")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    a = ap.parse_args()

    import anthropic_chat as AC

    cfg = R.RESKINS["canonical"]
    system = B.investor_system_prompt(cfg, False)
    # The exact user message the investor sees in the baseline treatment, built
    # by the live prompt function so the diagnostic and the run cannot diverge.
    user = B.investor_user_prompt(cfg, None)

    efforts = [e.strip() for e in a.efforts.split(",") if e.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")
    rows = []
    print(f"{'effort':<10}{'thinking':<12}{'out tok':>9}{'blocks':>8}"
          f"{'answer ch':>11}   sample of the decision")
    print("-" * 88)
    for effort in efforts:
        for mode in ("adaptive", "disabled"):
            chat = AC.make_anthropic_chat_fn(
                model=MODEL, effort=effort, thinking={"type": mode},
                display="summarized", max_tokens=a.max_tokens)
            seen = len(AC.USAGE)
            texts = []
            for _ in range(a.reps):
                try:
                    texts.append(chat(system, user))
                except Exception as exc:                        # noqa: BLE001
                    print(f"{effort:<10}{mode:<12}  FAILED "
                          f"{type(exc).__name__}: {exc}")
                    texts.append("")
            us = AC.USAGE[seen:]
            if not us:
                continue
            out = statistics.mean(u["output_tokens"] or 0 for u in us)
            blk = sum(u.get("thinking_blocks", 0) or 0 for u in us)
            ans = statistics.mean(len(t) for t in texts)
            rows.append((effort, mode, out, blk))
            with OUT.open("a") as fh:           # one cell, written as it lands
                fh.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "effort": effort, "thinking": mode, "reps": len(us),
                    "output_tokens_mean": out,
                    "input_tokens_mean": statistics.mean(
                        u["input_tokens"] or 0 for u in us),
                    "thinking_blocks": blk,
                    "thinking_chars": sum(u["thinking_chars_visible"] or 0
                                          for u in us),
                    "stop_reasons": sorted({u["stop_reason"] for u in us}),
                    "answers": texts,
                }, ensure_ascii=False) + "\n")
            print(f"{effort:<10}{mode:<12}{out:>9.0f}{blk:>8}{ans:>11.0f}   "
                  f"{(texts[0] or '')[:34]!r}")

    print("\nreading")
    for effort in efforts:
        ad = next((r for r in rows if r[0] == effort and r[1] == "adaptive"), None)
        di = next((r for r in rows if r[0] == effort and r[1] == "disabled"), None)
        if not (ad and di):
            continue
        ratio = ad[2] / max(di[2], 1)
        verdict = ("arms separate" if ratio > 1.25 or ad[2] - di[2] > 80
                   else "no separation, adaptive did not think here")
        print(f"  {effort:<8} adaptive {ad[2]:>6.0f} against disabled "
              f"{di[2]:>6.0f}, ratio {ratio:>4.2f}   {verdict}")

    ad = [r for r in rows if r[1] == "adaptive"]
    if len(ad) > 1:
        spread = max(r[2] for r in ad) - min(r[2] for r in ad)
        print(f"\n  output tokens in the adaptive arm move by {spread:.0f} across "
              f"the effort levels tried")
        print("  " + ("effort reaches the model, so the parameter is live"
                      if spread > 60 else
                      "a flat line here would mean the parameter never arrived"))

    tot = AC.usage_total(PRICE_IN, PRICE_OUT)
    print(f"\n{tot['calls']} calls, {tot['usd']:.3f} $")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
