#!/usr/bin/env python3
"""
run_full_cycle.py — the full Berg experiment, resumable.

Crosses  surfaces x conditions x model  in one run:
    surfaces    the six recognition surfaces (reskin.RESKINS)
    conditions  no_history, social_history_pos, social_history_neg, communication
    model       any providers.PACK key (self play, or a cross pair via TRUSTEE_MODEL)

Design decisions it implements (PI directives 2026-07-22):
  * social history VALENCE is a manipulated factor (positive = trust rewarded,
    negative = trust exploited), which is the direct test of H5.
  * every attempted dyad is tagged and its failure cause is split into
    infrastructure vs behavioral by attrition.py.
  * RESUMABLE. Output is a stable file per model. Re-running the same command
    skips the dyads already written and continues, so a crash or a rate wall
    (the Nebius trial running dry) costs nothing but a restart.

Routing note: the provider is decided by providers.PACK[key].provider, so a
Nebius model goes to Nebius and a DeepSeek model to DeepSeek. Override per model
with providers.model_config(key, provider=...). See NEBIUS_GUIDE.md.

Env knobs (all optional):
    MODELS         comma keys                (default: gpt-oss-120b-nebius)
    TRUSTEE_MODEL  key for a cross pair       (default: empty = self play)
    SURFACES       comma reskin keys          (default: all six)
    CONDITIONS     comma condition names      (default: the four above)
    N              dyads per cell             (default: 100)
    COMM_ROUNDS    message rounds for comm    (default: 2)
    MAX_TOKENS     output cap                 (default: 1500; raise for reasoning)
    MIN_CALL_INTERVAL / LLM_TEMPERATURE
    DRY=1          print the plan + a sample history report and exit (no API)

Output -> data/full/full_<model>[__x__<trustee>].jsonl  (+ _attrition.jsonl)
"""
import os
import sys
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

import random

import berg_trust_game as B
import reskin as R
import providers as P
import attrition as ATT

CONDITIONS_DEFAULT = "no_history,social_history_pos,social_history_neg,communication"
ENGINE_TREATMENT = {
    "no_history": "no_history",
    "social_history_pos": "social_history",
    "social_history_neg": "social_history",
    "communication": "communication",
}


N_HISTORY_VARIANTS = 10


def valence_history_report(cfg: B.GameConfig, valence: str, variant: int = 0,
                           n: int = 12) -> str:
    """A prior-session report with a fixed valence, drawn independently per variant.

    The valence is unambiguous in every variant, since a positive report shows
    trust repaid at a profit on average and a negative report shows it exploited.
    Within a report the rows vary, and each report carries one or two rows that
    run against its own valence, so the stimulus is a distribution and not a
    degenerate one. Variants are drawn from independent seeds, so a cell can
    spread several reports across its dyads and the idiosyncrasy of any single
    report stops being confounded with the valence itself.
    """
    rng = random.Random(hash((valence, variant, cfg.endowment, cfg.multiplier)) & 0xFFFFFFFF)
    E, m = cfg.endowment, cfg.multiplier
    rows = []
    # index of the row that runs against the valence, so no report is degenerate
    against = rng.randrange(n)
    for i in range(n):
        x = max(1, round(E * rng.uniform(0.3, 0.8)))
        if valence == "positive":
            ratio = rng.uniform(0.2, 0.7) if i == against else rng.uniform(1.2, 2.2)
        else:
            ratio = rng.uniform(1.2, 1.8) if i == against else rng.uniform(0.0, 0.5)
        y = min(m * x, max(0, round(x * ratio)))
        rows.append({"sent": x, "returned": y,
                     "investor_parse_clean": True, "trustee_parse_clean": True})
    return B.build_history_report(rows, cfg)


def condition_setup(cfg, condition, comm_rounds, variant=0):
    """Return (history_report, comm_rounds, valence_tag) for a condition."""
    if condition == "no_history":
        return None, 0, "none"
    if condition == "social_history_pos":
        return valence_history_report(cfg, "positive", variant), 0, "positive"
    if condition == "social_history_neg":
        return valence_history_report(cfg, "negative", variant), 0, "negative"
    if condition == "communication":
        return None, comm_rounds, "none"
    sys.exit(f"unknown condition {condition!r}")


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["dyad_id"])
                except Exception:
                    pass
    return done


def main() -> None:
    models = [m.strip() for m in os.environ.get("MODELS", "gpt-oss-120b-nebius").split(",") if m.strip()]
    trustee = os.environ.get("TRUSTEE_MODEL", "").strip()
    surfaces = [s.strip() for s in os.environ.get(
        "SURFACES", ",".join(R.RESKINS)).split(",") if s.strip()]
    conditions = [c.strip() for c in os.environ.get("CONDITIONS", CONDITIONS_DEFAULT).split(",") if c.strip()]
    n = int(os.environ.get("N", "100"))
    comm_rounds = int(os.environ.get("COMM_ROUNDS", "2"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "1500"))
    min_iv = float(os.environ.get("MIN_CALL_INTERVAL", "0"))
    max_consec_infra = int(os.environ.get("MAX_CONSEC_INFRA", "8"))
    dry = os.environ.get("DRY", "0") == "1"

    for m in models:
        if m not in P.PACK:
            sys.exit(f"unknown model {m!r}; options: {', '.join(P.PACK)}")
    if trustee and trustee not in P.PACK:
        sys.exit(f"unknown TRUSTEE_MODEL {trustee!r}")
    for s in surfaces:
        if s not in R.RESKINS:
            sys.exit(f"unknown surface {s!r}")
    for c in conditions:
        if c not in ENGINE_TREATMENT:
            sys.exit(f"unknown condition {c!r}; options: {', '.join(ENGINE_TREATMENT)}")

    cells = len(surfaces) * len(conditions)
    print(f"Plan per model: {len(surfaces)} surfaces x {len(conditions)} conditions "
          f"= {cells} cells x {n} = {cells*n} dyads.")
    if dry:
        for val in ("positive", "negative"):
            for v in (0, 1):
                print(f"\n--- {val} history, variant {v} ---")
                print(valence_history_report(R.RESKINS["canonical"], val, v, 6))
        print(f"\nConditions -> engine treatment: {ENGINE_TREATMENT}")
        print("DRY run, no API calls made.")
        return

    if trustee:
        miss = P.require_keys(list(set(models + [trustee])))
    else:
        miss = P.require_keys(models)
    if miss:
        sys.exit("Missing API keys:\n  " + "\n  ".join(miss))

    for model in models:
        mcfg = P.model_config(model, temperature=temp, min_call_interval=min_iv, max_tokens=max_tokens)
        chat = B.make_chat_fn(mcfg)
        tru_chat = tru_mcfg = None
        if trustee:
            tru_mcfg = P.model_config(trustee, temperature=temp, min_call_interval=min_iv, max_tokens=max_tokens)
            tru_chat = B.make_chat_fn(tru_mcfg)
        tag = model + ("__x__" + trustee if trustee else "")
        out = Path(__file__).resolve().parent / "data" / "full" / f"full_{tag}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        att_path = out.with_name(out.stem + "_attrition.jsonl")

        done = load_done(out)
        att = ATT.AttritionLog()
        if att_path.exists():   # keep prior attrition across resumes
            for line in att_path.read_text().splitlines():
                if line.strip():
                    att.records.append(json.loads(line))

        print(f"\n=== {tag}  ({P.PACK[model].provider}, {P.reasoning_class(model)}) ===")
        print(f"    output {out}  (already done: {len(done)} dyads)")
        try:
            smoke = B.with_retry(lambda: chat("You output only JSON.", 'Return {"ok": true} and nothing else.'))
            print("    smoke:", smoke[:70].replace("\n", " "))
        except Exception as e:
            print(f"    SMOKE FAILED [{ATT.classify_exception(e)}] -> skip {model}: {e}")
            continue

        consec_infra = 0
        abort = False
        with out.open("a", encoding="utf-8") as fh:
            for surface in surfaces:
                if abort:
                    break
                cfg = R.RESKINS[surface]
                for condition in conditions:
                    if abort:
                        break
                    _, cr, valence = condition_setup(cfg, condition, comm_rounds)
                    cell = f"{tag}/{surface}/{condition}"
                    ok = fail = skip = 0
                    for i in range(n):
                        did = f"{surface}|{condition}|{i:04d}"
                        if did in done:
                            skip += 1
                            continue
                        variant = i % N_HISTORY_VARIANTS
                        hist, _, _ = condition_setup(cfg, condition, comm_rounds, variant)
                        try:
                            rec = B.run_dyad(chat, cfg, ENGINE_TREATMENT[condition], did, mcfg,
                                             history_report=hist, comm_rounds=cr,
                                             trustee_chat_fn=tru_chat, trustee_mcfg=tru_mcfg)
                        except Exception as e:
                            att.add(cell=cell, dyad_id=did, role="dyad",
                                    cause=ATT.classify_exception(e), snippet=f"{type(e).__name__}: {e}")
                            consec_infra += 1
                            if consec_infra >= max_consec_infra:
                                print(f"    ABORT {tag}: {consec_infra} consecutive infrastructure "
                                      f"failures (provider wall or credit exhausted). "
                                      f"Fix and re-run the same command to resume.")
                                abort = True
                                break
                            continue   # not marked done -> retried on resume
                        consec_infra = 0
                        rec.update(surface=surface, model_key=model,
                                   reasoning_class=P.reasoning_class(model),
                                   condition=condition, valence=valence,
                                   history_variant=(variant if valence != "none" else None))
                        clean = rec["investor_parse_clean"] and rec["trustee_parse_clean"]
                        att.add(cell=cell, dyad_id=did, role="dyad", cause="ok" if clean else "format")
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        ok += clean
                        fail += (0 if clean else 1)
                    print(f"    {surface:16} {condition:20} ok={ok:>3} format_fail={fail:>2} skipped={skip:>3}")
        att.write(att_path)
        s = att.summary()
        infra = sum(c["infrastructure"] for c in s.values())
        behav = sum(c["behavioral"] for c in s.values())
        print(f"    attrition: infrastructure={infra}  behavioral={behav}  -> {att_path}")


if __name__ == "__main__":
    main()
