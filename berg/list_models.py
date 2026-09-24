import os, sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(e if e.exists() else None)
except ImportError:
    pass
import providers as P

prov = sys.argv[1] if len(sys.argv) > 1 else "nebius"
base, env, *_ = P.PROVIDERS[prov]
from openai import OpenAI
client = OpenAI(base_url=base, api_key=os.environ.get(env, "MISSING"))
STUDY = {"openai/gpt-oss-120b", "meta-llama/Llama-3.3-70B-Instruct",
         "Qwen/Qwen3-32B", "zai-org/GLM-5.2", "MiniMaxAI/MiniMax-M3",
         "moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro",
         "deepseek-ai/DeepSeek-V4-Flash"}
ids = sorted(m.id for m in client.models.list().data)
print(f"{len(ids)} models on {prov}\n")
for i in ids:
    print(f"  {i}" + ("   <-- under study, not eligible as judge" if i in STUDY else ""))
print("\nEligible judges are the unmarked rows. Add one to providers.PACK, then")
print("  MODELS_JUDGE=<key> nohup python3 run_judge.py > log_judge.log 2>&1 &")
