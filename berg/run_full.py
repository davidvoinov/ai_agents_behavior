import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

import berg_trust_game as B

PROVIDERS = {
    "groq":       ("openai/gpt-oss-20b",      "https://api.groq.com/openai/v1", "GROQ_API_KEY",       "https://console.groq.com/keys"),
    "openrouter": ("openai/gpt-oss-20b:free", "https://openrouter.ai/api/v1",   "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "cerebras":   ("gpt-oss-120b",            "https://api.cerebras.ai/v1",     "CEREBRAS_API_KEY",   "https://cloud.cerebras.ai"),
    "openai":     ("gpt-4o",                  None,                             "OPENAI_API_KEY",     "https://platform.openai.com/api-keys"),
}

def main() -> None:
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        sys.exit(f"Unknown LLM_PROVIDER={provider!r}. Options: {', '.join(PROVIDERS)}")
    model, base_url, key_env, signup = PROVIDERS[provider]
    model = os.environ.get("MODEL_OVERRIDE", model)

    key = os.environ.get(key_env, "").strip()
    if not key or key.startswith("your-"):
        print(f"\n  No API key for '{provider}'. Get one at {signup}, "
              f"put it in .env as {key_env}=..., then re-run.\n")
        sys.exit(1)

    n_no   = int(os.environ.get("N_NO_HISTORY", "32"))
    n_soc  = int(os.environ.get("N_SOCIAL_HISTORY", "28"))
    n_comm = int(os.environ.get("N_COMM", "32"))
    comm_rounds = int(os.environ.get("COMM_ROUNDS", "2"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    sleep = float(os.environ.get("SLEEP_BETWEEN", "1.0"))
    min_interval = float(os.environ.get("MIN_CALL_INTERVAL", "0"))

    mcfg = B.ModelConfig(model=model, base_url=base_url, api_key_env=key_env,
                         temperature=temp, min_call_interval=min_interval)
    game = B.GameConfig()
    chat = B.make_chat_fn(mcfg)

    total = n_no + n_soc + n_comm
    calls = (n_no + n_soc) * 2 + n_comm * (2 * comm_rounds + 2)
    print(f"Provider={provider}  model={model}  temp={temp}  max_tokens={mcfg.max_tokens}")
    print(f"Plan: no_history={n_no}, social_history={n_soc} (self-generated), "
          f"communication={n_comm} (comm_rounds={comm_rounds})")
    print(f"≈ {total} dyads / ≈ {calls} model calls.")

    print("Smoke test ...", end=" ", flush=True)
    print("got:", chat("You output only JSON.",
                       'Return {"ok": true} and nothing else.')[:120].replace("\n", " "))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parent / "data" / f"run_{provider}_{ts}.jsonl"
    print(f"\nWriting -> {out_path}\n")

    B.run_berg_replication(chat, game, mcfg, out_path,
                           n_no_history=n_no, n_social_history=n_soc,
                           sleep_between=sleep)

    if n_comm > 0:
        print(f"\n--- communication (comm_rounds={comm_rounds}) ---")
        comm = B.run_treatment(chat, game, mcfg, "communication", n_comm, out_path,
                               history_report=None, sleep_between=sleep,
                               comm_rounds=comm_rounds)
        B.summarize(comm, "COMMUNICATION")

    print(f"\nDone. Raw records: {out_path}")

if __name__ == "__main__":
    main()
