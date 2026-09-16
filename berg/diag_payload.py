#!/usr/bin/env python3
"""Find which thinking-toggle payload (if any) actually works for a model.

Tries several documented formats plus a deliberately invalid control. If the
invalid control also returns 200, the endpoint silently drops unknown fields and
NO toggle can ever work through it — the model must be reached another way.

Usage: python3 diag_payload.py minimax-m3-think glm-5.2-think
"""
import os, sys, time, json
from pathlib import Path
try:
    from dotenv import load_dotenv
    e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(e if e.exists() else None)
except ImportError: pass
import berg_trust_game as B, providers as P

PROMPT = ("Two players. You may pass 0-10 dollars; the amount triples; the other "
          "player may return any part. How much do you pass and why? Answer in one short paragraph.")

VARIANTS = [
    ("baseline (no extra_body)",      None),
    ("thinking disabled",             {"thinking": {"type": "disabled"}}),
    ("thinking enabled",              {"thinking": {"type": "enabled"}}),
    ("chat_template enable_thinking F",{"chat_template_kwargs": {"enable_thinking": False}}),
    ("reasoning_effort low",          {"reasoning_effort": "low"}),
    ("reasoning enabled false",       {"reasoning": {"enabled": False}}),
    ("INVALID CONTROL",               {"zzz_not_a_real_param": {"nonsense": True}}),
]

for key in sys.argv[1:]:
    spec = P.PACK[key]
    print(f"\n=== {key}  ({spec.provider} / {spec.model}) ===")
    lens = {}
    for label, extra in VARIANTS:
        mcfg = P.model_config(key, temperature=0.0, max_tokens=700)
        mcfg.extra_body = extra
        mcfg.hard_timeout = 60.0; mcfg.request_timeout = 60
        try:
            t0 = time.time(); out = B.make_chat_fn(mcfg)("You are a participant.", PROMPT)
            dt = time.time() - t0
            lens[label] = len(out)
            print(f"  {label:34} OK {dt:5.1f}s len={len(out):>5} think={'Y' if '<think>' in out.lower() else 'n'}")
        except Exception as ex:
            print(f"  {label:34} FAIL {type(ex).__name__}: {str(ex)[:90]}")
    if "INVALID CONTROL" in lens:
        print("  -> endpoint ACCEPTS an invalid field, so unknown params are dropped silently.")
        print("     A toggle cannot be verified here by status code, only by output length.")
    base = lens.get("baseline (no extra_body)")
    if base:
        spread = {k: round(v/base, 2) for k, v in lens.items() if k != "baseline (no extra_body)"}
        print(f"  length ratios vs baseline: {spread}")
        print("     a ratio near 1.00 everywhere means no payload changes behaviour")
