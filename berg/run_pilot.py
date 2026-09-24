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
}

def main() -> None:
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        sys.exit(f"Unknown LLM_PROVIDER={provider!r}. Options: {', '.join(PROVIDERS)}")
    model, base_url, key_env, signup = PROVIDERS[provider]

    key = os.environ.get(key_env, "").strip()
    if not key or key.startswith("your-"):
        print(
            f"\n  No API key for provider '{provider}'.\n"
            f"  1. Get a free key: {signup}\n"
            f"  2. Open .env and set {key_env}=<your key>\n"
            f"  3. Re-run: python run_pilot.py\n"
        )
        sys.exit(1)

    n = int(os.environ.get("PILOT_N", "5"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))

    mcfg = B.ModelConfig(model=model, base_url=base_url,
                         api_key_env=key_env, temperature=temp)
    game = B.GameConfig()
    chat = B.make_chat_fn(mcfg)

    print(f"Provider={provider}  model={model}  temperature={temp}")
    print("Smoke test ...", end=" ", flush=True)
    smoke = chat("You output only JSON.", 'Return {"ok": true} and nothing else.')
    print("got:", smoke[:160].replace("\n", " "))

    comm_rounds = int(os.environ.get("COMM_ROUNDS", "2"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parent / "data" / f"pilot_{provider}_{ts}.jsonl"
    print(f"\nRunning pilot: {n} dyads/treatment  ->  {out_path}\n")

    B.run_berg_replication(chat, game, mcfg, out_path,
                           n_no_history=n, n_social_history=n,
                           sleep_between=0.5)

    print(f"\n--- communication treatment (comm_rounds={comm_rounds}) ---")
    comm = B.run_treatment(chat, game, mcfg, "communication", n, out_path,
                           history_report=None, sleep_between=0.5,
                           comm_rounds=comm_rounds)
    B.summarize(comm, "COMMUNICATION")

    print(f"\nDone. Raw records: {out_path}")

if __name__ == "__main__":
    main()
