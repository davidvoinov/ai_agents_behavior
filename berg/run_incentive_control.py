#!/usr/bin/env python3
"""
run_incentive_control.py — positive control for the incentive axis (H3).

Why this exists. In the surface gradient the subgame perfect prediction is a
transfer of zero for every parameterization, so a payoff maximizing agent is
expected to hold the transfer fixed when the endowment and the multiplier move.
A flat response there is therefore uninformative about whether the agent reacts
to incentives at all. This runner supplies the missing positive control by
enforcing a minimum return, which moves the optimum itself.

The mechanics. The second mover must return at least a fraction k of the amount
received. If it returns exactly that floor, the first mover ends with

    E - X + k*m*X  =  E + X*(k*m - 1)

so the first mover's payoff strictly increases in X when k*m > 1 and strictly
decreases when k*m < 1. With the canonical multiplier of 3 and k in
{0, 1/3, 2/3} the payoff maximizing transfer is 0, then indifferent, then the
whole endowment. An agent that responds to incentives MUST move across these
levels. An agent that stays flat is not responding to incentives, which is what
makes the flat response in the surface gradient interpretable.

Usage:
    MODELS="qwen3-nothink,qwen3-think" N=100 MAX_TOKENS=3000 python run_incentive_control.py

Env knobs:
    MODELS       comma pack keys      (default qwen3-nothink)
    K_LEVELS     comma fractions      (default 0,0.3333,0.6667)
    SURFACE      reskin key           (default canonical)
    N            dyads per level      (default 100)
    MAX_TOKENS / LLM_TEMPERATURE / MIN_CALL_INTERVAL / DRY

Output -> data/incentive/incentive_<model>.jsonl   (resumable, same scheme as
the full-cycle runner, so existing cells never need re-running)
"""
import os
import sys
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    _e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_e if _e.exists() else None)
except ImportError:
    pass

import berg_trust_game as B
import reskin as R
import providers as P
import attrition as ATT


def optimum(cfg: B.GameConfig) -> str:
    """The payoff maximizing transfer under the enforced floor, for logging."""
    km = cfg.min_return_frac * cfg.multiplier
    if km > 1.0001:
        return f"send all ({cfg.endowment})"
    if km < 0.9999:
        return "send 0"
    return "indifferent"


def main() -> None:
    models = [m.strip() for m in os.environ.get("MODELS", "qwen3-nothink").split(",") if m.strip()]
    ks = [float(x) for x in os.environ.get("K_LEVELS", "0,0.3333,0.6667").split(",") if x.strip()]
    surface = os.environ.get("SURFACE", "canonical")
    n = int(os.environ.get("N", "100"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "1500"))
    min_iv = float(os.environ.get("MIN_CALL_INTERVAL", "0"))
    dry = os.environ.get("DRY", "0") == "1"
    base = R.RESKINS[surface]

    print(f"Surface {surface} (E={base.endowment}, m={base.multiplier}); {len(ks)} levels x {n} dyads per model.")
    for k in ks:
        cfg = B.GameConfig(**{**base.__dict__, "min_return_frac": k})
        print(f"  k={k:.4f}  k*m={k*base.multiplier:.2f}  payoff-maximizing transfer: {optimum(cfg)}")
    if dry:
        cfg = B.GameConfig(**{**base.__dict__, "min_return_frac": ks[-1]})
        print("\n--- rules text at the top level ---")
        print(B._rules_block(cfg))
        print("\nDRY run, no API calls.")
        return

    miss = P.require_keys(models)
    if miss:
        sys.exit("Missing API keys:\n  " + "\n  ".join(miss))

    for model in models:
        mcfg = P.model_config(model, temperature=temp, min_call_interval=min_iv, max_tokens=max_tokens)
        chat = B.make_chat_fn(mcfg)
        out = Path(__file__).resolve().parent / "data" / "incentive" / f"incentive_{model}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {json.loads(l)["dyad_id"] for l in out.read_text().splitlines() if l.strip()}
        att = ATT.AttritionLog()
        print(f"\n=== {model} ({P.reasoning_class(model)}) -> {out}  (done {len(done)}) ===")
        try:
            B.with_retry(lambda: chat("You output only JSON.", 'Return {"ok": true} and nothing else.'))
        except Exception as e:
            print(f"  SMOKE FAILED [{ATT.classify_exception(e)}]: {e}")
            continue

        with out.open("a", encoding="utf-8") as fh:
            for k in ks:
                cfg = B.GameConfig(**{**base.__dict__, "min_return_frac": k})
                cell = f"{model}/{surface}/k{k:.2f}"
                ok = fail = skip = 0
                for i in range(n):
                    did = f"{surface}|k{k:.4f}|{i:04d}"
                    if did in done:
                        skip += 1
                        continue
                    try:
                        rec = B.run_dyad(chat, cfg, "incentive_control", did, mcfg)
                    except Exception as e:
                        att.add(cell=cell, dyad_id=did, role="dyad",
                                cause=ATT.classify_exception(e), snippet=f"{type(e).__name__}: {e}")
                        continue
                    rec.update(surface=surface, model_key=model, condition="incentive_control",
                               reasoning_class=P.reasoning_class(model), k_level=k,
                               predicted_optimum=optimum(cfg))
                    clean = rec["investor_parse_clean"] and rec["trustee_parse_clean"]
                    att.add(cell=cell, dyad_id=did, role="dyad", cause="ok" if clean else "format")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    ok += clean
                    fail += (0 if clean else 1)
                print(f"    k={k:.4f}  ok={ok:>3} format_fail={fail:>2} skipped={skip:>3}")
        att.write(out.with_name(out.stem + "_attrition.jsonl"))


if __name__ == "__main__":
    main()
