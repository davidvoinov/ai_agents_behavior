from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import berg_trust_game as B
import providers as P
import reskin as R

OUT = Path(__file__).resolve().parent / "data" / "smoke_new_models.jsonl"
BEAT = Path(__file__).resolve().parent / "data" / "smoke_progress.log"

def beat(msg: str) -> None:
    BEAT.parent.mkdir(parents=True, exist_ok=True)
    with BEAT.open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")

ARMS = ["gpt-5.6-think", "gpt-5.6-nothink", "sonnet-5-think", "sonnet-5-nothink"]
SURFACE = "canonical"
TREATMENT = "no_history"
MAX_TOKENS = 4000

CALLS: list[dict] = []

def openai_chat_fn(spec: P.ModelSpec, max_tokens: int):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai --break-system-packages") from exc
    base_url, key_env, _ = P.PROVIDERS[spec.provider]
    key = os.environ.get(key_env, "").strip()
    if not key:
        raise SystemExit(f"set {key_env} before running {spec.key}")
    client = OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))

    def chat_fn(system: str, user: str) -> str:
        kw = dict(model=spec.model,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                  max_completion_tokens=max_tokens)
        kw.update(spec.extra_body or {})
        r = client.chat.completions.create(**kw)
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        CALLS.append({
            "arm": spec.key,
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
            "finish_reason": r.choices[0].finish_reason,
        })
        return r.choices[0].message.content or ""

    return chat_fn

def anthropic_chat_fn(spec: P.ModelSpec, max_tokens: int):
    import anthropic_chat as AC
    eb = spec.extra_body or {}
    inner = AC.make_anthropic_chat_fn(
        model=spec.model, effort=eb.get("effort"), thinking=eb.get("thinking"),
        max_tokens=max_tokens)
    seen = len(AC.USAGE)

    def chat_fn(system: str, user: str) -> str:
        nonlocal seen
        out = inner(system, user)
        for u in AC.USAGE[seen:]:
            CALLS.append({
                "arm": spec.key,
                "input_tokens": u["input_tokens"],
                "output_tokens": u["output_tokens"],
                "reasoning_tokens": None,
                "thinking_blocks": u.get("thinking_blocks", 0),
                "thinking_chars_visible": u["thinking_chars_visible"],
                "finish_reason": u["stop_reason"],
            })
        seen = len(AC.USAGE)
        return out

    return chat_fn

def build(spec: P.ModelSpec, max_tokens: int):
    return (anthropic_chat_fn(spec, max_tokens) if spec.api == "anthropic"
            else openai_chat_fn(spec, max_tokens))

def run_arm(key: str, dyads: int, max_tokens: int) -> dict:
    spec = P.PACK[key]
    chat = build(spec, max_tokens)
    cfg = R.RESKINS[SURFACE]
    mcfg = B.ModelConfig(model=spec.model, temperature=1.0, max_tokens=max_tokens,
                         extra_body=spec.extra_body)
    before = len(CALLS)
    rows, errors = [], []
    for i in range(dyads):
        try:
            rec = B.run_dyad(chat, cfg, TREATMENT, f"smoke-{key}-{i}", mcfg)
            rows.append(rec)
            beat(f"{key} dyad {i+1}/{dyads} ok, sent={rec.get('sent')}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            beat(f"{key} dyad {i+1}/{dyads} FAILED {type(exc).__name__}: {exc}")
        print(".", end="", flush=True)
    calls = CALLS[before:]
    clean = [r for r in rows
             if r.get("investor_parse_clean") and r.get("trustee_parse_clean")]
    rt = [c["reasoning_tokens"] for c in calls if c.get("reasoning_tokens") is not None]
    vis = [c.get("thinking_chars_visible", 0) or 0 for c in calls]
    blk = [c.get("thinking_blocks", 0) or 0 for c in calls]
    out = [c["output_tokens"] or 0 for c in calls]
    return {
        "arm": key, "model": spec.model, "api": spec.api,
        "toggle": spec.extra_body, "attempted": dyads,
        "clean": len(clean), "errors": errors,
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
        "output_tokens": sum(out),
        "reasoning_tokens": sum(rt) if rt else None,
        "thinking_blocks": sum(blk) if any(blk) else None,
        "thinking_chars_visible": sum(vis) if any(vis) else None,
        "mean_transfer": (sum(r["sent"] / r["endowment"] for r in clean) / len(clean)
                          if clean else None),
        "calls_detail": calls,
        "rows": rows,
    }

def report(results: list[dict], price_lookup) -> None:
    print("\n\n" + "=" * 78)
    print(f"{'arm':<18}{'clean':>7}{'calls':>7}{'in tok':>9}{'out tok':>9}"
          f"{'reason tok':>12}{'transfer':>10}")
    print("-" * 78)
    for r in results:
        rt = "--" if r["reasoning_tokens"] is None else f"{r['reasoning_tokens']}"
        tf = "--" if r["mean_transfer"] is None else f"{r['mean_transfer']:.3f}"
        print(f"{r['arm']:<18}{r['clean']:>4}/{r['attempted']:<2}{r['calls']:>7}"
              f"{r['input_tokens']:>9}{r['output_tokens']:>9}{rt:>12}{tf:>10}")
    print("-" * 78)

    for a, b in (("gpt-5.6-think", "gpt-5.6-nothink"),
                 ("sonnet-5-think", "sonnet-5-nothink")):
        ra = next((r for r in results if r["arm"] == a), None)
        rb = next((r for r in results if r["arm"] == b), None)
        if not (ra and rb and ra["calls"] and rb["calls"]):
            continue
        oa, ob = ra["output_tokens"] / ra["calls"], rb["output_tokens"] / rb["calls"]
        print(f"\n{a} against {b}")
        print(f"  output tokens per call   {oa:>8.0f} against {ob:>8.0f}, "
              f"ratio {oa / max(ob, 1):.2f}")
        if ra["reasoning_tokens"] is not None:
            print(f"  reasoning tokens per call {ra['reasoning_tokens']/ra['calls']:>7.0f} "
                  f"against {(rb['reasoning_tokens'] or 0)/rb['calls']:.0f}")
        if ra.get("thinking_blocks") is not None:
            print(f"  thinking blocks returned  {ra['thinking_blocks']:>7} "
                  f"against {rb.get('thinking_blocks') or 0}")
        verdict = ("the two arms separate, the parameter reached the model"
                   if oa / max(ob, 1) > 1.25 or (oa - ob) > 80
                   else "THE TWO ARMS DO NOT SEPARATE, treat the parameter as dropped")
        print(f"  {verdict}")

    print("\nmeasured hidden reasoning, for cost_estimate.py --reasoning")
    pairs = {"gpt-5.6-think": "gpt-5.6-nothink",
             "sonnet-5-think": "sonnet-5-nothink"}
    for r in results:
        if not r["calls"]:
            continue
        if r["reasoning_tokens"] is not None:
            vis = r["output_tokens"] - r["reasoning_tokens"]
            print(f"  {r['arm']:<20}{r['reasoning_tokens']/max(vis,1):>6.2f}x   "
                  f"from the billed reasoning count")
            continue
        other = next((x for x in results if x["arm"] == pairs.get(r["arm"])), None)
        if other and other["calls"]:
            per, ref = r["output_tokens"]/r["calls"], other["output_tokens"]/other["calls"]
            note = ""
            if r.get("thinking_blocks"):
                note = (f", thinking fired on {r['thinking_blocks']} of "
                        f"{r['calls']} calls")
            print(f"  {r['arm']:<20}{(per-ref)/max(ref,1):>6.2f}x   "
                  f"from the gap to its plain arm{note}")
        elif r["arm"] in pairs.values():
            print(f"  {r['arm']:<20}{0.0:>6.2f}x   thinking is off in this arm")

    total = sum(r["input_tokens"] / 1e6 * price_lookup(r["arm"])[0]
                + r["output_tokens"] / 1e6 * price_lookup(r["arm"])[1]
                for r in results)
    print(f"\nthis smoke test cost about {total:.3f} $ at list price")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyads", type=int, default=5)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in arms if x not in P.PACK]
    if unknown:
        sys.exit(f"unknown arm: {', '.join(unknown)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")
    beat(f"start, arms {arms}, {a.dyads} dyads each")

    results = []
    for key in arms:
        print(f"\n{key} ({P.PACK[key].model}, {P.PACK[key].extra_body}) ", end="")
        beat(f"{key} starting")
        try:
            r = run_arm(key, a.dyads, a.max_tokens)
        except SystemExit as exc:
            print(f"\n  skipped: {exc}")
            beat(f"{key} skipped: {exc}")
            continue
        except Exception as exc:
            print(f"\n  FAILED: {type(exc).__name__}: {exc}")
            beat(f"{key} FAILED {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        with OUT.open("a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        beat(f"{key} done, {r['clean']}/{r['attempted']} clean, "
             f"{r['output_tokens']} output tokens")

    report(results, lambda k: (P.PACK[k].price_in, P.PACK[k].price_out))
    print(f"stored in {OUT}")

    bad = [r for r in results if r["errors"]]
    for r in bad:
        print(f"\nerrors in {r['arm']}:")
        for e in r["errors"][:3]:
            print("   ", e)

if __name__ == "__main__":
    main()
