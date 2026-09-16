import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
from reskin import RESKINS, recognition_probe_prompt, score_recognition

PROVIDERS = {
    "groq":       ("openai/gpt-oss-20b",      "https://api.groq.com/openai/v1", "GROQ_API_KEY",       "https://console.groq.com/keys"),
    "openrouter": ("openai/gpt-oss-20b:free", "https://openrouter.ai/api/v1",   "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "cerebras":   ("gpt-oss-120b",            "https://api.cerebras.ai/v1",     "CEREBRAS_API_KEY",   "https://cloud.cerebras.ai"),
    "openai":     ("gpt-4o",                  None,                             "OPENAI_API_KEY",     "https://platform.openai.com/api-keys"),
}

def gate_on_tests() -> None:
    here = Path(__file__).resolve().parent
    r = subprocess.run([sys.executable, str(here / "test_reskin.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        sys.exit("test_reskin.py FAILED — fix the harness before spending API quota.")
    print("Offline tests: OK")

def run_full_series(chat, cfg, mcfg, name: str, n_no: int, n_soc: int,
                    n_comm: int, comm_rounds: int, out_path: Path,
                    sleep_between: float = 0.0) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    if n_no > 0:
        no_hist = B.run_treatment(
            chat, cfg, mcfg, f"{name}_no_history", n_no, out_path,
            history_report=None, sleep_between=sleep_between)
        out[f"{name}_no_history"] = no_hist
    else:
        no_hist = []

    if n_soc > 0:
        report = B.build_history_report(no_hist, cfg)
        soc_hist = B.run_treatment(
            chat, cfg, mcfg, f"{name}_social_history", n_soc, out_path,
            history_report=report, sleep_between=sleep_between)
        out[f"{name}_social_history"] = soc_hist

    if n_comm > 0:
        comm = B.run_treatment(
            chat, cfg, mcfg, f"{name}_communication", n_comm, out_path,
            history_report=None, sleep_between=sleep_between,
            comm_rounds=comm_rounds)
        out[f"{name}_communication"] = comm

    return out

def run_recognition(chat, cfg, variant: str, n: int, mcfg,
                    out_path: Path) -> list[dict]:
    rules = B._rules_block(cfg)
    system, user = recognition_probe_prompt(rules)
    records = []
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(n):
            raw = B.with_retry(lambda: chat(system, user))
            rec = score_recognition(raw)
            rec.update({
                "variant": variant, "probe_id": f"{variant}-recog-{i:02d}",
                "model": mcfg.model, "temperature": mcfg.temperature,
                "timestamp": datetime.utcnow().isoformat(),
                "rules_shown": rules,
            })
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
    return records

def summarize_pilot(behavioral: dict[str, dict[str, list[dict]]],
                    recognition: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 88)
    print(f"{'variant/treatment':<32}{'n':>3}{'sent/E':>9}{'ret/recv':>10}"
          f"{'sent>0':>8}{'trustee_recv':>13}")
    print("-" * 88)
    for name, by_treatment in behavioral.items():
        e = RESKINS[name].endowment
        for treatment_name, recs in by_treatment.items():
            clean = [r for r in recs
                     if r["investor_parse_clean"] and r["trustee_parse_clean"]]
            sent_frac = [r["sent"] / e for r in clean]
            ratios = [r["return_ratio"] for r in clean if r["return_ratio"] is not None]
            recv_frac = [r["tripled"] / (e * cfg_mult(name)) for r in clean] if clean else []
            sent_s = f"{statistics.mean(sent_frac):.2f}" if clean else "-"
            ret_s = f"{statistics.mean(ratios):.2f}" if ratios else "-"
            pos_s = f"{sum(s > 0 for s in sent_frac)}/{len(clean)}" if clean else "-"
            trustee_recv = (f"{statistics.mean(recv_frac):.2f}" if recv_frac else "-")
            print(f"{treatment_name:<32}{len(clean):>3}{sent_s:>9}{ret_s:>10}"
                  f"{pos_s:>8}{trustee_recv:>13}")
        rp = recognition.get(name, [])
        if rp:
            recog_s = f"{sum(r['recognized_paradigm'] for r in rp)}/{len(rp)}"
            named_s = f"{sum(r['named_study'] for r in rp)}/{len(rp)}"
            leak_s = f"{sum(r['leaked_benchmark'] for r in rp)}/{len(rp)}"
            print(f"  [{name} recognition] recognized={recog_s} "
                  f"named={named_s} leaked_benchmark={leak_s}")
    print("=" * 88)
    print("Read: sent/E = mean sent as fraction of endowment (Berg humans ≈ 0.52).")
    print("ret/recv = mean return ratio (Y / tripled X); trustee_recv = mean")
    print("tripled amount received by the trustee, as a fraction of the")
    print("trustee's max possible receipt (E * multiplier).")
    print("recognized/named/leaked_benchmark = recognition probes that")
    print("identified the paradigm / named Berg-Dickhaut-McCabe / reproduced")
    print("5.16 or 4.66.")
    print("Hoped-for pattern: sent/E similar across variants (behavior, not")
    print("memorization); recognized high for canonical, low for reskin_shifted.")

def cfg_mult(name: str) -> int:
    return RESKINS[name].multiplier

def main() -> None:
    gate_on_tests()

    provider = os.environ.get("LLM_PROVIDER", "cerebras").strip().lower()
    if provider not in PROVIDERS:
        sys.exit(f"Unknown LLM_PROVIDER={provider!r}. Options: {', '.join(PROVIDERS)}")
    model, base_url, key_env, signup = PROVIDERS[provider]
    model = os.environ.get("MODEL_OVERRIDE", model)

    key = os.environ.get(key_env, "").strip()
    if not key or key.startswith("your-"):
        sys.exit(f"No API key for '{provider}'. Get one at {signup}, "
                 f"put it in .env as {key_env}=..., re-run.")

    n_no = int(os.environ.get("N_NO", "6"))
    n_soc = int(os.environ.get("N_SOC", "6"))
    n_comm = int(os.environ.get("N_COMM", "6"))
    comm_rounds = int(os.environ.get("COMM_ROUNDS", "2"))
    n_recog = int(os.environ.get("RECOG_N", "3"))
    variant_names = [v.strip() for v in
                     os.environ.get("VARIANTS",
                                    ",".join(RESKINS)).split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in RESKINS]
    if unknown:
        sys.exit(f"Unknown VARIANTS {unknown}. Options: {', '.join(RESKINS)}")
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    min_interval = float(os.environ.get("MIN_CALL_INTERVAL", "2"))
    sleep = float(os.environ.get("SLEEP_BETWEEN", "0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "1500"))

    mcfg = B.ModelConfig(model=model, base_url=base_url, api_key_env=key_env,
                         temperature=temp, min_call_interval=min_interval,
                         max_tokens=max_tokens)
    chat = B.make_chat_fn(mcfg)

    calls_per_variant = (
        (n_no + n_soc) * 2 + n_comm * (2 + 2 * comm_rounds) + n_recog
    )
    calls = len(variant_names) * calls_per_variant
    print(f"Provider={provider}  model={model}  temp={temp}")
    print(f"Plan per variant: no_history={n_no}, social_history={n_soc} "
          f"(self-generated), communication={n_comm} (comm_rounds={comm_rounds}), "
          f"recognition={n_recog}.")
    print(f"Plan: {len(variant_names)} variants × {calls_per_variant} calls "
          f"≈ {calls} calls total "
          f"(≥ {calls * min_interval / 60:.0f} min at {min_interval}s/call).")

    print("Smoke test ...", end=" ", flush=True)
    print("got:", chat("You output only JSON.",
                       'Return {"ok": true} and nothing else.')[:120].replace("\n", " "))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(__file__).resolve().parent / "data"
    beh_path = data_dir / f"reskin_pilot_{provider}_{ts}.jsonl"
    rec_path = data_dir / f"recog_{provider}_{ts}.jsonl"
    data_dir.mkdir(exist_ok=True)
    print(f"\nBehavioral -> {beh_path}\nRecognition -> {rec_path}\n")

    behavioral: dict[str, dict[str, list[dict]]] = {}
    recognition: dict[str, list[dict]] = {}
    for name in variant_names:
        cfg = RESKINS[name]
        print(f"--- variant {name} (E={cfg.endowment}, m={cfg.multiplier}, "
              f"labels={cfg.investor_label}/{cfg.trustee_label}) ---")
        behavioral[name] = run_full_series(
            chat, cfg, mcfg, name, n_no, n_soc, n_comm, comm_rounds,
            beh_path, sleep_between=sleep)
        recognition[name] = run_recognition(chat, cfg, name, n_recog, mcfg,
                                            rec_path)
        got = sum(r["recognized_paradigm"] for r in recognition[name])
        print(f"    recognition: {got}/{n_recog} identified the paradigm")

    summarize_pilot(behavioral, recognition)
    print(f"\nDone. Raw: {beh_path} + {rec_path}")

if __name__ == "__main__":
    main()
