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

CANDIDATES = [
    "NousResearch/Hermes-4-405B",
    "zai-org/GLM-5.3-Flash",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/nemotron-3-super-120b-a12b",
    "Qwen/Qwen3.5-397B-A17B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen3-235B-A22B",
    "google/gemma-3-27b-it",
]
STUDY_IDS = {"openai/gpt-oss-120b", "meta-llama/Llama-3.3-70B-Instruct",
             "Qwen/Qwen3-32B", "zai-org/GLM-5.2", "MiniMaxAI/MiniMax-M3",
             "moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro",
             "deepseek-ai/DeepSeek-V4-Flash"}

def sample_transcript() -> str:
    for f in sorted(glob.glob("data/full/full_*.jsonl")):
        if any(x in f for x in ("_attrition", "_purged", "__")):
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            tx = r.get("transcript")
            if r.get("condition") == "communication" and tx:
                return "\n".join(
                    f"{m.get('role','?')}: {m.get('content', m.get('message',''))}"
                    for m in tx if isinstance(m, dict))
    raise SystemExit("no communication transcript found")

def main() -> None:
    from openai import OpenAI
    args = sys.argv[1:]
    prov = "nebius"
    if args and args[0].startswith("provider="):
        prov = args.pop(0).split("=", 1)[1]
    ids = args or CANDIDATES
    base, env, *_ = P.PROVIDERS[prov]
    client = OpenAI(base_url=base, api_key=os.environ.get(env, "MISSING"))
    print(f"screening on {prov} at {base}")
    sysp, user = CC.judge_prompt(sample_transcript())

    print(f"prompt is {len(sysp) + len(user)} chars, cap 1500 tokens, "
          f"schema attached\n")
    print(f"{'model':<40}{'parses':>8}{'finish':>9}{'chars':>8}   verdict")
    print("-" * 78)
    ok = []
    for mid in ids:
        if mid in STUDY_IDS:
            print(f"{mid:<40}{'--':>8}{'--':>9}{'--':>8}   under study, not eligible")
            continue
        try:
            resp = client.chat.completions.create(
                model=mid, temperature=0.0, max_tokens=1500,
                messages=[{"role": "system", "content": sysp},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": CC.JUDGE_SCHEMA})
        except Exception as e:
            msg = str(e)
            short = ("not served" if "does not exist" in msg or "404" in msg
                     else type(e).__name__)
            print(f"{mid:<40}{'--':>8}{'--':>9}{'--':>8}   {short}")
            continue
        ch = resp.choices[0]
        content = ch.message.content or ""
        parsed = CC.parse_codes(content).get("coded_clean", False)
        verdict = ("USABLE" if parsed and ch.finish_reason == "stop" else
                   "ignores the schema" if ch.finish_reason == "length" else
                   "parses but stopped oddly" if parsed else "no object")
        print(f"{mid:<40}{str(parsed):>8}{ch.finish_reason:>9}"
              f"{len(content):>8}   {verdict}")
        if verdict == "USABLE":
            ok.append(mid)

    print("\n" + ("usable judges: " + ", ".join(ok) if ok else
                  "no candidate returned the object"))
    if ok:
        print("\nRegister the one you want in providers.PACK beside judge-nemotron,\n"
              "then run the judge with MODELS_JUDGE set to that key.")
    else:
        print("\nEvery candidate failed, so the independence requirement cannot be met\n"
              "from this catalogue. The fallback is to code with gpt-oss-120b and have\n"
              "a second judge cover only the gpt-oss transcripts, which keeps every\n"
              "configuration read by a model that is not itself.")

if __name__ == "__main__":
    main()
