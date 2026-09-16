from __future__ import annotations

import os
import sys
import json
import glob
from pathlib import Path

try:
    from dotenv import load_dotenv
    _e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_e if _e.exists() else None)
except ImportError:
    pass

import providers as P
import code_comm as CC

KEY = sys.argv[1] if len(sys.argv) > 1 else "judge-nemotron"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def transcript_text(rec: dict) -> str:
    tx = rec.get("transcript") or []
    if isinstance(tx, str):
        return tx
    out = []
    for m in tx:
        if isinstance(m, dict):
            out.append(f"{m.get('role','?')}: {m.get('content', m.get('message',''))}")
        else:
            out.append(str(m))
    return "\n".join(out)

def main() -> None:
    from openai import OpenAI
    for mt in (1500,):
        mcfg = P.model_config(KEY, temperature=0.0, max_tokens=mt)
        client = OpenAI(base_url=mcfg.base_url,
                        api_key=os.environ.get(mcfg.api_key_env, "MISSING"))
        print(f"\n=== {mcfg.model} at max_tokens={mt} ===")
        rows = []
        for f in sorted(glob.glob("data/full/full_*.jsonl")):
            if any(x in f for x in ("_attrition", "_purged", "__")):
                continue
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("condition") == "communication" and r.get("transcript"):
                    rows.append(r)
                    break
            if len(rows) >= N:
                break

        for i, rec in enumerate(rows[:N], 1):
            sysp, user = CC.judge_prompt(transcript_text(rec))
            kw = dict(model=mcfg.model, temperature=0.0, max_tokens=mt,
                      messages=[{"role": "system", "content": sysp},
                                {"role": "user", "content": user}],
                      response_format={"type": "json_schema",
                                       "json_schema": CC.JUDGE_SCHEMA})
            if mcfg.extra_body:
                kw["extra_body"] = mcfg.extra_body
            try:
                resp = client.chat.completions.create(**kw)
            except Exception as e:
                print(f"  call {i}: ERROR {type(e).__name__}: {str(e)[:160]}")
                continue
            ch = resp.choices[0]
            content = ch.message.content or ""
            reasoning = (getattr(ch.message, "reasoning", None)
                         or getattr(ch.message, "reasoning_content", None) or "")
            parsed = CC.parse_codes(content)
            print(f"  call {i}: finish={ch.finish_reason} "
                  f"content={len(content)}ch reasoning={len(reasoning)}ch "
                  f"completion_tokens={getattr(resp.usage,'completion_tokens',0)} "
                  f"parsed={parsed.get('coded_clean')}")
            print(f"    prompt was {len(sysp) + len(user)} chars")
            print(f"    content head: {content[:220]!r}")
            if reasoning:
                print(f"    reasoning head: {reasoning[:160]!r}")
        print("\n  read the four cases in this file's docstring against the lines above")

if __name__ == "__main__":
    main()
