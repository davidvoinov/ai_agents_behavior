#!/usr/bin/env python3
"""Detect a thinking toggle correctly, by reading the RAW API response.

Models differ in where reasoning goes. Qwen emits inline <think> tags inside
`content`, so a length check finds it. GLM, MiniMax and Kimi are served with a
reasoning parser that moves it into a separate `reasoning_content` field, so
`content` looks identical whether thinking is on or off and a length check is
blind. The decisive signals are `reasoning_content` and completion_tokens.

Usage: python3 diag_reasoning.py MiniMaxAI/MiniMax-M3 zai-org/GLM-5.2 Qwen/Qwen3-32B
"""
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
try:
    from dotenv import load_dotenv
    e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(e if e.exists() else None)
except ImportError: pass

KEY = os.environ.get("NEBIUS_API_KEY", "")
if not KEY: sys.exit("NEBIUS_API_KEY missing")
HOSTS = {"zai-org/GLM-5.2": "https://api.tokenfactory.uk-south1.nebius.com/v1"}
DEFAULT_HOST = "https://api.studio.nebius.com/v1"
PROMPT = ("Two players. You may pass 0-10 dollars; the amount triples; the other player "
          "may return any part. How much do you pass? Think it through, then answer.")

def call(host, model, extra, timeout=120):
    payload = {"model": model, "max_tokens": 1200, "temperature": 0,
               "messages": [{"role": "user", "content": PROMPT}], **(extra or {})}
    req = urllib.request.Request(f"{host}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), time.time()-t0, None
    except urllib.error.HTTPError as ex:
        return None, time.time()-t0, f"HTTP {ex.code}: {ex.read().decode()[:160]}"
    except Exception as ex:
        return None, time.time()-t0, f"{type(ex).__name__}: {ex}"

for model in sys.argv[1:]:
    host = HOSTS.get(model, DEFAULT_HOST)
    print(f"\n=== {model} ===")
    for label, extra in [("thinking ON  (enable_thinking=True)",  {"chat_template_kwargs": {"enable_thinking": True}}),
                         ("thinking OFF (enable_thinking=False)", {"chat_template_kwargs": {"enable_thinking": False}}),
                         ("no kwargs (server default)",           None)]:
        body, dt, err = call(host, model, extra)
        if err: print(f"  {label:38} FAIL {dt:5.1f}s {err}"); continue
        msg = body["choices"][0]["message"]
        content = msg.get("content") or ""
        rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
        u = body.get("usage", {})
        print(f"  {label:38} {dt:5.1f}s content={len(content):>5} reasoning_content={len(rc):>5} "
              f"completion_tokens={u.get('completion_tokens','?'):>5} inline_think={'Y' if '<think>' in content.lower() else 'n'}")
    print("  -> a real toggle shows reasoning_content or completion_tokens collapsing when OFF")
