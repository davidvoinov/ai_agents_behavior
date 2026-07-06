#!/usr/bin/env python3
"""
run_crossmodel_pilot.py — REAL-AGENT pilot of CROSS-MODEL dyads: canonical
skin, Investor and Trustee played by two different models/providers.

This is a PILOT (exploratory): its job is to answer, cheaply, whether
behavior (mean sent, return ratio, betrayal/reciprocity) shifts when the
Investor and Trustee are different models, vs the same-model self-play
baselines already collected.

Every cell now runs the full original three-treatment series (project
decision: no baseline-only runs), with trustee_chat_fn/trustee_mcfg passed
through every run_treatment call so the Trustee stays the cell's own model:
  1. <cellname>_no_history      — N_NO dyads, no history report.
  2. <cellname>_social_history  — N_SOC dyads, history report built from that
     SAME cell's own clean no_history records (canonical cfg).
  3. <cellname>_communication   — N_COMM dyads, COMM_ROUNDS of pre-play cheap
     talk, no history report.

Safety: refuses to run if test_reskin.py OR test_crossmodel.py fail (never
burn quota on a harness bug).

Usage (locally, keys in .env):
    python run_crossmodel_pilot.py
Env knobs (all optional):
    N_NO                      no_history dyads per cell            (default 6)
    N_SOC                     social_history dyads per cell         (default 6)
    N_COMM                    communication dyads per cell          (default 6)
    COMM_ROUNDS               pre-play message rounds               (default 2)
    LLM_TEMPERATURE           default 1.0
    MAX_TOKENS                default 3000
    MIN_CALL_INTERVAL_CEREBRAS  seconds between Cerebras calls    (default 2)
    MIN_CALL_INTERVAL_GROQ      seconds between Groq calls        (default 25)
    CELLS   comma list overriding the default two cells, e.g.
            "cerebras:gpt-oss-120b>groq:qwen/qwen3-32b,groq:qwen/qwen3-32b>cerebras:gpt-oss-120b"
            ('>' separates investor>trustee; each side is "provider:model")
Outputs:
    data/crossmodel_<ts>.jsonl   behavioral records (one file for all cells)
"""
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
from reskin import RESKINS

# --------------------------------------------------------------------------- #
# Provider map: base_url, key env var                                        #
# --------------------------------------------------------------------------- #
PROVIDER_MAP = {
    "cerebras":   ("https://api.cerebras.ai/v1",   "CEREBRAS_API_KEY"),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# Default cross-model cells: (A) gpt-oss-120b/Cerebras invests, qwen3-32b/Groq
# trustees; (B) the reverse.
DEFAULT_CELLS = [
    "cerebras:gpt-oss-120b>groq:qwen/qwen3-32b",
    "groq:qwen/qwen3-32b>cerebras:gpt-oss-120b",
]

# Same-day self-play reference lines, hardcoded for comparison in the summary.
# These come from 2026-07-02 canonical n≈40 pilots (same-model dyads, no
# history, no communication) — NOT recomputed here, just printed for context.
SELFPLAY_GPT_OSS_120B_MEAN_SENT = 1.88
SELFPLAY_GPT_OSS_120B_FULL_BETRAYAL = "13/17"
SELFPLAY_QWEN3_32B_MEAN_SENT = 0.82
SELFPLAY_QWEN3_32B_RECIPROCATED = "7/7"
SELFPLAY_QWEN3_32B_RECIPROCATED_RATIO = 0.61


def gate_on_tests() -> None:
    """Abort unless BOTH offline harness test suites pass."""
    here = Path(__file__).resolve().parent
    for test_file in ("test_reskin.py", "test_crossmodel.py"):
        r = subprocess.run([sys.executable, str(here / test_file)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            sys.exit(f"{test_file} FAILED — fix the harness before spending API quota.")
    print("Offline tests: OK (test_reskin.py + test_crossmodel.py)")


def parse_side(spec: str) -> tuple[str, str]:
    """'provider:model' -> (provider, model)."""
    if ":" not in spec:
        sys.exit(f"Bad cell side {spec!r}: expected 'provider:model'.")
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDER_MAP:
        sys.exit(f"Unknown provider {provider!r} in {spec!r}. "
                 f"Options: {', '.join(PROVIDER_MAP)}")
    return provider, model


def parse_cells(raw: list[str]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """['prov:model>prov:model', ...] -> [((inv_provider, inv_model), (tru_provider, tru_model)), ...]"""
    cells = []
    for spec in raw:
        if ">" not in spec:
            sys.exit(f"Bad cell {spec!r}: expected 'investor_provider:model>trustee_provider:model'.")
        inv_spec, tru_spec = spec.split(">", 1)
        cells.append((parse_side(inv_spec), parse_side(tru_spec)))
    return cells


def check_key(provider: str) -> str:
    _, key_env = PROVIDER_MAP[provider]
    key = os.environ.get(key_env, "").strip()
    if not key or key.startswith("your-"):
        sys.exit(f"No API key for provider '{provider}'. "
                 f"Put a real key in .env as {key_env}=..., re-run.")
    return key


def sanitize_model(model: str) -> str:
    """Model names can contain '/' (e.g. 'qwen/qwen3-32b') — sanitize for use
    in filenames/treatment names."""
    return model.replace("/", "-")


def build_mcfg(provider: str, model: str, temp: float, max_tokens: int,
              min_interval_cerebras: float, min_interval_groq: float) -> B.ModelConfig:
    base_url, key_env = PROVIDER_MAP[provider]
    if provider == "cerebras":
        min_interval = min_interval_cerebras
    elif provider == "groq":
        min_interval = min_interval_groq
    else:
        min_interval = 0.0
    return B.ModelConfig(
        model=model, base_url=base_url, api_key_env=key_env,
        temperature=temp, max_tokens=max_tokens, min_call_interval=min_interval,
    )


def run_full_series(inv_chat: B.ChatFn, inv_mcfg: B.ModelConfig,
                    tru_chat: B.ChatFn, tru_mcfg: B.ModelConfig,
                    cfg, cellname: str, n_no: int, n_soc: int, n_comm: int,
                    comm_rounds: int, out_path: Path,
                    sleep_between: float = 0.0) -> dict[str, list[dict]]:
    """Run the full three-treatment series for ONE cross-model cell.

    The Investor always uses inv_chat/inv_mcfg and the Trustee always uses
    tru_chat/tru_mcfg (passed through every run_treatment call), so every
    treatment in the series stays a genuine cross-model dyad.

    1. <cellname>_no_history      — n_no dyads, no history report.
    2. <cellname>_social_history  — n_soc dyads, history report built from
       THIS call's own clean no_history records (canonical cfg/labels).
    3. <cellname>_communication   — n_comm dyads, comm_rounds of pre-play
       cheap talk, no history report.

    Returns {treatment_name: records} (a treatment is omitted if its n is 0).
    """
    out: dict[str, list[dict]] = {}

    if n_no > 0:
        no_hist = B.run_treatment(
            inv_chat, cfg, inv_mcfg, f"{cellname}_no_history", n_no, out_path,
            history_report=None, sleep_between=sleep_between,
            trustee_chat_fn=tru_chat, trustee_mcfg=tru_mcfg)
        out[f"{cellname}_no_history"] = no_hist
    else:
        no_hist = []

    if n_soc > 0:
        report = B.build_history_report(no_hist, cfg)
        soc_hist = B.run_treatment(
            inv_chat, cfg, inv_mcfg, f"{cellname}_social_history", n_soc,
            out_path, history_report=report, sleep_between=sleep_between,
            trustee_chat_fn=tru_chat, trustee_mcfg=tru_mcfg)
        out[f"{cellname}_social_history"] = soc_hist

    if n_comm > 0:
        comm = B.run_treatment(
            inv_chat, cfg, inv_mcfg, f"{cellname}_communication", n_comm,
            out_path, history_report=None, sleep_between=sleep_between,
            comm_rounds=comm_rounds,
            trustee_chat_fn=tru_chat, trustee_mcfg=tru_mcfg)
        out[f"{cellname}_communication"] = comm

    return out


def smoke_test(chat_fn: B.ChatFn, label: str) -> None:
    print(f"Smoke test [{label}] ...", end=" ", flush=True)
    # with_retry so a transient provider 429 (e.g. Cerebras queue_exceeded
    # during high traffic) delays the launch instead of crashing it.
    out = B.with_retry(lambda: chat_fn(
        "You output only JSON.", 'Return {"ok": true} and nothing else.'))
    print("got:", out[:120].replace("\n", " "))


def summarize_cell(records: list[dict], label: str) -> dict:
    """n clean, mean sent, SD, zeros count; among sent>0: mean return ratio,
    reciprocated count, full-betrayal count."""
    clean = [r for r in records
             if r["investor_parse_clean"] and r["trustee_parse_clean"]]
    n = len(clean)
    print(f"\n=== {label} (n={n}/{len(records)} clean) ===")
    if n == 0:
        print("  no clean records")
        return {"label": label, "n_clean": 0, "n_total": len(records)}

    sents = [r["sent"] for r in clean]
    mean_sent = statistics.mean(sents)
    sd_sent = statistics.pstdev(sents) if n > 1 else 0.0
    zeros = sum(s == 0 for s in sents)

    print(f"  n clean        : {n}")
    print(f"  mean sent      : {mean_sent:.2f}   (SD {sd_sent:.2f})")
    print(f"  zeros (sent=0) : {zeros}")

    pos = [r for r in clean if r["sent"] > 0]
    out = {
        "label": label, "n_clean": n, "n_total": len(records),
        "mean_sent": mean_sent, "sd_sent": sd_sent, "zeros": zeros,
        "n_trusted": len(pos),
    }
    if pos:
        ratios = [r["return_ratio"] for r in pos if r["return_ratio"] is not None]
        mean_ratio = statistics.mean(ratios) if ratios else None
        reciprocated = sum(r["reciprocated"] for r in pos)
        full_betrayal = sum(r["full_betrayal"] for r in pos)
        out["mean_return_ratio"] = mean_ratio
        out["reciprocated"] = reciprocated
        out["full_betrayal"] = full_betrayal
        print(f"  among sent>0 (n={len(pos)}):")
        print(f"    mean return ratio : "
              f"{(mean_ratio if mean_ratio is not None else 0):.2f}")
        print(f"    reciprocated      : {reciprocated}/{len(pos)}")
        print(f"    full betrayal     : {full_betrayal}/{len(pos)}")
    else:
        print("  among sent>0: none (all investors sent 0)")
    return out


def print_selfplay_reference() -> None:
    print("\n--- Same-day self-play reference (2026-07-02 canonical n≈40 pilots) ---")
    print(f"  gpt-oss-120b self-play : mean sent {SELFPLAY_GPT_OSS_120B_MEAN_SENT:.2f}, "
          f"full betrayal {SELFPLAY_GPT_OSS_120B_FULL_BETRAYAL}")
    print(f"  qwen3-32b self-play    : mean sent {SELFPLAY_QWEN3_32B_MEAN_SENT:.2f}, "
          f"trustees reciprocated {SELFPLAY_QWEN3_32B_RECIPROCATED} "
          f"(ratio {SELFPLAY_QWEN3_32B_RECIPROCATED_RATIO:.2f})")


def main() -> None:
    gate_on_tests()

    n_no = int(os.environ.get("N_NO", "6"))
    n_soc = int(os.environ.get("N_SOC", "6"))
    n_comm = int(os.environ.get("N_COMM", "6"))
    comm_rounds = int(os.environ.get("COMM_ROUNDS", "2"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))
    min_interval_cerebras = float(os.environ.get("MIN_CALL_INTERVAL_CEREBRAS", "2"))
    min_interval_groq = float(os.environ.get("MIN_CALL_INTERVAL_GROQ", "25"))

    cell_specs = [c.strip() for c in
                 os.environ.get("CELLS", ",".join(DEFAULT_CELLS)).split(",") if c.strip()]
    cells = parse_cells(cell_specs)

    # Verify keys up front for every provider referenced.
    providers_needed = set()
    for (inv_provider, _), (tru_provider, _) in cells:
        providers_needed.add(inv_provider)
        providers_needed.add(tru_provider)
    for provider in sorted(providers_needed):
        check_key(provider)

    cfg = RESKINS["canonical"]  # canonical skin for every cell

    # Estimate call count and pacing time before starting. no_history and
    # social_history cost 2 calls/dyad each (1 investor + 1 trustee);
    # communication costs (2 + 2*comm_rounds) calls/dyad (comm_rounds rounds
    # of alternating investor/trustee messages, then 1 decision call each).
    calls_per_cell = (n_no + n_soc) * 2 + n_comm * (2 + 2 * comm_rounds)
    total_calls = calls_per_cell * len(cells)
    # Rough pacing estimate: use the slower of the two per-cell intervals as a
    # conservative floor (calls to each role are paced independently, but a
    # cell's wall-clock is dominated by the slower provider).
    print(f"Plan per cell: no_history={n_no}, social_history={n_soc} "
          f"(self-generated), communication={n_comm} (comm_rounds={comm_rounds}).")
    print(f"Plan: {len(cells)} cell(s) x {calls_per_cell} calls "
          f"= {total_calls} calls total.")
    for idx, ((inv_provider, inv_model), (tru_provider, tru_model)) in enumerate(cells):
        inv_interval = min_interval_cerebras if inv_provider == "cerebras" else (
            min_interval_groq if inv_provider == "groq" else 0.0)
        tru_interval = min_interval_cerebras if tru_provider == "cerebras" else (
            min_interval_groq if tru_provider == "groq" else 0.0)
        est_seconds = (n_no + n_soc + n_comm) * max(inv_interval, tru_interval)
        print(f"  cell {idx}: investor={inv_provider}:{inv_model}  "
              f"trustee={tru_provider}:{tru_model}  "
              f"(~{est_seconds/60:.1f} min at pacing floor)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"crossmodel_{ts}.jsonl"
    print(f"\nOutput -> {out_path}\n")

    # Smoke-test each distinct provider:model pair once before the cells.
    smoke_tested: set[tuple[str, str]] = set()
    chat_fn_cache: dict[tuple[str, str], B.ChatFn] = {}
    mcfg_cache: dict[tuple[str, str], B.ModelConfig] = {}

    def get_chat_and_mcfg(provider: str, model: str) -> tuple[B.ChatFn, B.ModelConfig]:
        key = (provider, model)
        if key not in mcfg_cache:
            mcfg = build_mcfg(provider, model, temp, max_tokens,
                              min_interval_cerebras, min_interval_groq)
            chat = B.make_chat_fn(mcfg)
            mcfg_cache[key] = mcfg
            chat_fn_cache[key] = chat
        if key not in smoke_tested:
            smoke_test(chat_fn_cache[key], f"{provider}:{model}")
            smoke_tested.add(key)
        return chat_fn_cache[key], mcfg_cache[key]

    all_summaries = []
    for (inv_provider, inv_model), (tru_provider, tru_model) in cells:
        inv_chat, inv_mcfg = get_chat_and_mcfg(inv_provider, inv_model)
        tru_chat, tru_mcfg = get_chat_and_mcfg(tru_provider, tru_model)

        cellname = (f"cross_{inv_provider}-{sanitize_model(inv_model)}"
                   f"__{tru_provider}-{sanitize_model(tru_model)}")
        n_total = n_no + n_soc + n_comm
        print(f"\n--- cell: {cellname} ({n_total} dyads: "
              f"{n_no} no_history + {n_soc} social_history + "
              f"{n_comm} communication) ---")

        by_treatment = run_full_series(
            inv_chat, inv_mcfg, tru_chat, tru_mcfg, cfg, cellname,
            n_no, n_soc, n_comm, comm_rounds, out_path)

        for treatment_name, records in by_treatment.items():
            summary = summarize_cell(records, treatment_name)
            all_summaries.append(summary)
        print_selfplay_reference()

    print("\n" + "=" * 72)
    print("FINAL SUMMARY (all cells, all treatments)")
    print("=" * 72)
    for summary in all_summaries:
        label = summary.get("label", "?")
        n = summary.get("n_clean", 0)
        mean_sent = summary.get("mean_sent")
        sd_sent = summary.get("sd_sent")
        zeros = summary.get("zeros")
        mean_ratio = summary.get("mean_return_ratio")
        reciprocated = summary.get("reciprocated")
        full_betrayal = summary.get("full_betrayal")
        n_trusted = summary.get("n_trusted")
        if n == 0:
            print(f"  {label}: no clean records")
            continue
        print(f"  {label}: n={n} mean_sent={mean_sent:.2f} sd={sd_sent:.2f} "
              f"zeros={zeros}"
              + (f" | sent>0(n={n_trusted}): mean_ratio="
                 f"{(mean_ratio if mean_ratio is not None else 0):.2f} "
                 f"reciprocated={reciprocated}/{n_trusted} "
                 f"full_betrayal={full_betrayal}/{n_trusted}"
                 if n_trusted else " | sent>0: none"))
    print_selfplay_reference()
    print(f"\nDone. Raw -> {out_path}")


if __name__ == "__main__":
    main()
