#!/usr/bin/env python3
"""Probe a model with and without its toggle payload, on a short timeout.

Usage:  python3 diag_toggle.py glm-5.2-think glm-5.2-nothink minimax-m3-think minimax-m3-nothink
Prints latency, output length, and whether a <think> block appears, so an inert
or rejected toggle is visible immediately instead of after 30 minutes of retries.
"""
import os, sys, time
from pathlib import Path
try:
    from dotenv import load_dotenv
    e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(e if e.exists() else None)
except ImportError: pass
import berg_trust_game as B, providers as P

PROMPT = ("You are a participant. Reply with ONLY this JSON: "
          '{"reasoning":"<one short sentence>","amount":5}')

def probe(key, extra, label):
    mcfg = P.model_config(key, temperature=0.0, max_tokens=800)
    mcfg.extra_body = extra
    mcfg.hard_timeout = 45.0          # fail fast instead of hanging
    mcfg.request_timeout = 45
    chat = B.make_chat_fn(mcfg)
    t0 = time.time()
    try:
        out = chat("You output only JSON.", PROMPT)     # no retry wrapper on purpose
        dt = time.time() - t0
        print(f"  {label:34} OK  {dt:5.1f}s  chars={len(out):>5}  think={'YES' if '<think>' in out.lower() else 'no'}")
        print(f"      {out[:110].replace(chr(10),' ')}")
    except Exception as ex:
        print(f"  {label:34} FAIL {time.time()-t0:5.1f}s  {type(ex).__name__}: {str(ex)[:100]}")

for key in sys.argv[1:]:
    spec = P.PACK[key]
    print(f"\n=== {key}  ({spec.provider} / {spec.model}) ===")
    probe(key, None, "no extra_body")
    if spec.extra_body:
        probe(key, spec.extra_body, f"with {list(spec.extra_body)[0]}")
