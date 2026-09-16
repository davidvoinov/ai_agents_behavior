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
    mcfg.hard_timeout = 45.0
    mcfg.request_timeout = 45
    chat = B.make_chat_fn(mcfg)
    t0 = time.time()
    try:
        out = chat("You output only JSON.", PROMPT)
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
