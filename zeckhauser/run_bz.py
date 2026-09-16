import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

import bz_game as BZ

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

    n = int(os.environ.get("N_PER_CONDITION", "32"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    sleep = float(os.environ.get("SLEEP_BETWEEN", "0.0"))
    min_interval = float(os.environ.get("MIN_CALL_INTERVAL", "0"))
    method = os.environ.get("METHOD", "choicelist").strip().lower()
    grid_env = os.environ.get("BZ_GRID", "").strip()
    grid = [float(x) for x in grid_env.split(",") if x.strip()] or BZ.DEFAULT_GRID

    mcfg = BZ.ModelConfig(model=model, base_url=base_url, api_key_env=key_env,
                          temperature=temp, min_call_interval=min_interval)
    cfg = BZ.BZConfig()
    chat = BZ.make_chat_fn(mcfg)

    if method == "choicelist":
        calls = 3 * n * len(grid) + n
        plan = f"choice-list over {len(grid)} probabilities [{grid[0]:.2f} .. {grid[-1]:.2f}]"
    else:
        calls = 4 * n
        plan = "direct MAP threshold"
    print(f"Provider={provider}  model={model}  temp={temp}  method={method}")
    print(f"Plan: {n} Decision Makers x 3 conditions, {plan} ~= {calls} calls.")
    print(f"Money-neutral MAP = {cfg.money_neutral_map():.3f}")

    print("Smoke test ...", end=" ", flush=True)
    smoke = BZ.with_retry(lambda: chat("You output only JSON.",
                                        'Return {"ok": true} and nothing else.'))
    print("got:", smoke[:120].replace("\n", " "))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parent / "data" / f"bz_{provider}_{ts}.jsonl"
    print(f"\nWriting -> {out_path}\n")

    summaries = {}
    for cond in BZ.CONDITIONS:
        if method == "choicelist":
            recs = BZ.run_choice_list_condition(chat, cfg, mcfg, cond, n, out_path,
                                                grid, sleep_between=sleep)
        else:
            recs = BZ.run_condition(chat, cfg, mcfg, cond, n, out_path,
                                    sleep_between=sleep)
        summaries[cond] = BZ.summarize(recs, cond)
        if method == "choicelist":
            mono = sum(1 for r in recs if r["switch_pattern"] == "single_switch")
            print(f"  monotone single-switch: {mono}/{len(recs)}")

    BZ.betrayal_decomposition(summaries)
    print(f"\nDone. Raw records: {out_path}")

if __name__ == "__main__":
    main()
