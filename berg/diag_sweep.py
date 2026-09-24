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
DEFAULT = "https://api.studio.nebius.com/v1"
PROMPT = ("Two players. You may pass 0-10 dollars; it triples; the other may return any part. "
          "How much do you pass? Think it through, then give a number.")

CANDIDATES = [
    ("(none) server default",            {}),
    ("chat_template enable_thinking=F",  {"chat_template_kwargs": {"enable_thinking": False}}),
    ("chat_template thinking=F",         {"chat_template_kwargs": {"thinking": False}}),
    ("chat_template enable_reasoning=F", {"chat_template_kwargs": {"enable_reasoning": False}}),
    ("chat_template reasoning=F",        {"chat_template_kwargs": {"reasoning": False}}),
    ("thinking_mode=disabled",           {"thinking_mode": "disabled"}),
    ("chat_template thinking_mode=dis",   {"chat_template_kwargs": {"thinking_mode": "disabled"}}),
    ("thinking_mode=off",                 {"thinking_mode": "off"}),
    ("thinking type=disabled",           {"thinking": {"type": "disabled"}}),
    ("thinking=false",                   {"thinking": False}),
    ("reasoning enabled=false",          {"reasoning": {"enabled": False}}),
    ("reasoning_effort=none",            {"reasoning_effort": "none"}),
    ("reasoning_effort=minimal",         {"reasoning_effort": "minimal"}),
    ("reasoning_effort=low",             {"reasoning_effort": "low"}),
    ("thinking_budget=0",                {"thinking_budget": 0}),
    ("max_thinking_tokens=0",            {"max_thinking_tokens": 0}),
]

model = sys.argv[1] if len(sys.argv) > 1 else "MiniMaxAI/MiniMax-M3"
host = HOSTS.get(model, DEFAULT)
print(f"=== {model}  @ {host} ===")
print(f"{'payload':36}{'code':>6}{'content':>9}{'reasoning':>11}{'tokens':>8}")
base = None
for label, extra in CANDIDATES:
    payload = {"model": model, "max_tokens": 2000, "temperature": 0,
               "messages": [{"role": "user", "content": PROMPT}], **extra}
    req = urllib.request.Request(f"{host}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            b = json.loads(r.read().decode())
        m = b["choices"][0]["message"]
        rc = len(m.get("reasoning_content") or m.get("reasoning") or "")
        ct = len(m.get("content") or "")
        tk = b.get("usage", {}).get("completion_tokens", 0)
        if base is None: base = rc
        flag = "  <== REASONING OFF" if (base and rc < base * 0.25) or rc == 0 else ""
        print(f"{label:36}{200:>6}{ct:>9}{rc:>11}{tk:>8}{flag}")
    except urllib.error.HTTPError as ex:
        print(f"{label:36}{ex.code:>6}   {ex.read().decode()[:70]}")
    except Exception as ex:
        print(f"{label:36}{'ERR':>6}   {type(ex).__name__}: {str(ex)[:60]}")
