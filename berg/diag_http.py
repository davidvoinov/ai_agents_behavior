#!/usr/bin/env python3
"""Raw HTTP probe of a Nebius model, bypassing langchain entirely.

Answers three questions with explicit status codes and no retry loops:
  1. is the model id actually served?
  2. does a minimal completion work?
  3. does the toggle payload change the outcome?

Usage:  python3 diag_http.py zai-org/GLM-5.2
"""
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
try:
    from dotenv import load_dotenv
    e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(e if e.exists() else None)
except ImportError: pass

BASE = "https://api.studio.nebius.com/v1"
KEY  = os.environ.get("NEBIUS_API_KEY", "")
if not KEY: sys.exit("NEBIUS_API_KEY missing from .env")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "zai-org/GLM-5.2"

def call(path, payload=None, timeout=40):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, method="POST" if payload else "GET",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time()-t0
    except urllib.error.HTTPError as ex:
        return ex.code, ex.read().decode()[:300], time.time()-t0
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}", time.time()-t0

print(f"key ...{KEY[-6:]}   model {MODEL}\n")
st, body, dt = call("/models")
if st == 200:
    ids = [m["id"] for m in body.get("data", [])]
    print(f"1) /models  {st}  {dt:.1f}s  served={len(ids)}")
    hit = [i for i in ids if MODEL.lower() in i.lower() or i.lower() in MODEL.lower()]
    print(f"   exact match for {MODEL}: {'YES' if MODEL in ids else 'NO'}")
    if hit: print(f"   близкие: {hit[:5]}")
    fam = MODEL.split('/')[-1].split('-')[0].lower()
    sim = [i for i in ids if fam in i.lower()][:8]
    print(f"   same family served: {sim}")
else:
    print(f"1) /models  {st}  {dt:.1f}s  {body}")

msg = [{"role":"user","content":'Return only {"ok": true}'}]
for label, extra in [("minimal, no toggle", {}),
                     ("thinking disabled", {"thinking":{"type":"disabled"}}),
                     ("thinking enabled",  {"thinking":{"type":"enabled"}}),
                     ("reasoning_effort low", {"reasoning_effort":"low"})]:
    st, body, dt = call("/chat/completions",
        {"model": MODEL, "messages": msg, "max_tokens": 60, "temperature": 0, **extra})
    if st == 200:
        ch = body["choices"][0]["message"]
        txt = (ch.get("content") or "")[:60].replace("\n", " ")
        print(f"2) {label:22} {st}  {dt:5.1f}s  out={txt!r}")
    else:
        print(f"2) {label:22} {st}  {dt:5.1f}s  {str(body)[:180]}")
