from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import berg_trust_game as B
import reskin as R

OUT = Path(__file__).resolve().parent / "data" / "probe" / "probe_results.jsonl"
TEMPERATURE = 1.0

MODELS = ["gpt-oss-120b",
          "deepseek-v4-pro-think", "deepseek-v4-pro", "kimi-k3",
          "minimax-m3-think", "minimax-m3-nothink", "glm-5.2-think",
          "glm-5.2-nothink",
          "gpt-5.6-nothink", "gpt-5.6-think",
          "sonnet-5-nothink", "sonnet-5-think"]

LABEL = {"gpt-oss-120b": "gpt-oss-120b", "deepseek-v4-pro-think": "DeepSeek-V4-Pro (r)",
         "deepseek-v4-pro": "DeepSeek-V4-Pro", "kimi-k3": "Kimi-K3",
         "minimax-m3-think": "MiniMax-M3 (r)", "minimax-m3-nothink": "MiniMax-M3",
         "glm-5.2-think": "GLM-5.2 (r)", "glm-5.2-nothink": "GLM-5.2",
         "gpt-5.6-nothink": "GPT-5.6", "gpt-5.6-think": "GPT-5.6 (r)",
         "sonnet-5-nothink": "Sonnet-5", "sonnet-5-think": "Sonnet-5 (r)"}

BUDGET = {"sonnet-5-think": 16000, "gpt-5.6-think": 8000,
          "deepseek-v4-pro-think": 4000, "minimax-m3-think": 4000,
          "glm-5.2-think": 4000, "kimi-k3": 4000, "gpt-oss-120b": 4000}
DEFAULT_BUDGET = 1500

SURFACES = ["canonical", "social_nonparadigm", "shifted", "reskin",
            "reskin_shifted", "strong_control", "strong_novel"]

ROUTE = {"kimi-k3": "kimi-k3@euw"}
TRUNCATED = {"length", "max_tokens"}
REQUEST_TIMEOUT = 300
TIMEOUT = {"kimi-k3": 1200}
RETRIES = 2
RETRYABLE = ("Timeout", "RateLimit", "APIConnection", "InternalServer")


def budget(model: str) -> int:
    return BUDGET.get(model, DEFAULT_BUDGET)


def timeout(model: str) -> int:
    return TIMEOUT.get(model, REQUEST_TIMEOUT)


def route(model: str) -> str:
    return ROUTE.get(model, model)


def build_chat(model: str):
    import providers as P
    spec = P.PACK[route(model)]
    if spec.provider in ("openai", "anthropic"):
        import smoke_new_models as SM
        inner = SM.build(spec, budget(model))

        def chat(system: str, user: str):
            n = len(SM.CALLS)
            text = inner(system, user)
            last = SM.CALLS[-1] if len(SM.CALLS) > n else {}
            return text, last.get("finish_reason")
        return chat
    mc = P.model_config(route(model), temperature=TEMPERATURE, max_tokens=budget(model),
                        hard_timeout=timeout(model))
    mc.request_timeout = timeout(model)
    inner = P.make_chat_fn(mc)

    def chat(system: str, user: str):
        return inner(system, user), None
    return chat


def stored(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def done(path: Path) -> set[tuple[str, str, int]]:
    return {(r["model"], r["surface"], r["rep"]) for r in stored(path) if r.get("ok")}


def call_once(chat, model: str, surface: str, rep: int) -> dict:
    cfg = R.RESKINS[surface]
    system, user = R.recognition_probe_prompt(B._rules_block(cfg))
    t0 = time.time()
    for attempt in range(RETRIES + 1):
        try:
            text, finish = chat(system, user)
            break
        except Exception as exc:
            name = type(exc).__name__
            if attempt < RETRIES and any(k in name for k in RETRYABLE):
                time.sleep(5 * (attempt + 1))
                continue
            return {"model": model, "surface": surface, "rep": rep, "ok": False,
                    "error": f"{name}: {exc}"[:500], "attempts": attempt + 1,
                    "temperature": TEMPERATURE, "max_tokens": budget(model)}
    rec = R.score_recognition(text or "")
    rec.update(model=model, surface=surface, rep=rep, route=route(model),
               temperature=TEMPERATURE, max_tokens=budget(model),
               finish_reason=finish, seconds=round(time.time() - t0, 1))
    if not (text or "").strip():
        rec.update(ok=False, error="empty completion")
    elif finish in TRUNCATED:
        rec.update(ok=False, error=f"truncated at {budget(model)} tokens")
    else:
        rec["ok"] = True
    return rec


def check(models: list[str]) -> None:
    import providers as P
    missing = P.require_keys([route(m) for m in models])
    for m in missing:
        print("missing key:", m)
    print(f"\none call per configuration on the canonical surface, temperature {TEMPERATURE}\n")
    bad = 0
    for model in models:
        if any(x.startswith(route(model) + " ") for x in missing):
            print(f"  {LABEL[model]:<22} skipped, no key")
            bad += 1
            continue
        rec = call_once(build_chat(model), model, "canonical", -1)
        state = "ok" if rec.get("ok") else "FAIL"
        bad += state != "ok"
        detail = (f"recognized={rec.get('recognized_paradigm')} "
                  f"finish={rec.get('finish_reason')} {rec.get('seconds')}s"
                  if rec.get("ok") else rec.get("error"))
        print(f"  {LABEL[model]:<22} {state:<5} {detail}")
    print("\nall configurations answer, the full run can start" if not bad
          else f"\n{bad} configuration(s) failed, fix before the full run")


def collect(models: list[str], reps: int, sleep: float) -> None:
    import providers as P
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = done(OUT)
    missing = P.require_keys([route(m) for m in models])
    if missing:
        print("missing API keys, these configurations are skipped:")
        for m in missing:
            print("   ", m)
        models = [m for m in models if not any(x.startswith(route(m) + " ") for x in missing)]
    todo = [(m, s, i) for m in models for s in SURFACES for i in range(reps)
            if (m, s, i) not in seen]
    print(f"{len(todo)} calls to make, {len(seen)} already stored, "
          f"temperature {TEMPERATURE}\n")
    chats = {}
    with OUT.open("a") as fh:
        current = None
        for n, (model, surface, rep) in enumerate(todo, 1):
            if model != current:
                current = model
                print(f"\n{LABEL[model]:<22}", end="", flush=True)
            if model not in chats:
                chats[model] = build_chat(model)
            rec = call_once(chats[model], model, surface, rep)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print("." if rec.get("ok") else "x", end="", flush=True)
            time.sleep(sleep)
    failed = [r for r in stored(OUT) if not r.get("ok")]
    print(f"\n\nstored in {OUT}")
    if failed:
        print(f"{len(failed)} failed calls on record, rerun the same command to retry them")


def report() -> None:
    import numpy as np
    rows = [r for r in stored(OUT) if r.get("ok")]
    latest = {}
    for r in rows:
        latest[(r["model"], r["surface"], r["rep"])] = r
    rows = list(latest.values())
    if not rows:
        sys.exit("no completed probe calls, run the collection first")

    def rate(sel, field):
        v = [r[field] for r in sel]
        return float(np.mean(v)) if v else float("nan")

    print("Recognition probe. Share of calls in which a fresh instance named the "
          "paradigm.\n")
    head = f"{'configuration':<22}" + "".join(f"{s[:9]:>10}" for s in SURFACES)
    print(head)
    print("-" * len(head))
    grand = {s: [] for s in SURFACES}
    for m in MODELS:
        sel = [r for r in rows if r["model"] == m]
        if not sel:
            continue
        cells = []
        for s in SURFACES:
            ss = [r for r in sel if r["surface"] == s]
            v = rate(ss, "recognized_paradigm")
            if ss:
                grand[s].append(v)
            cells.append("        --" if not ss else f"{v:>10.0%}")
        print(f"{LABEL.get(m, m):<22}" + "".join(cells))
    print("-" * len(head))
    print(f"{'all configurations':<22}" +
          "".join(f"{np.mean(grand[s]):>10.0%}" if grand[s] else "        --"
                  for s in SURFACES))
    can = [r for r in rows if r["surface"] == "canonical"]
    nov = [r for r in rows if r["surface"] == "strong_novel"]
    print(f"\ncanonical wording  paradigm named {rate(can, 'recognized_paradigm'):.0%}, "
          f"original study {rate(can, 'named_study'):.0%}, "
          f"a quantitative human result {rate(can, 'approx_benchmark'):.0%}, "
          f"the exact means of the original study {rate(can, 'leaked_benchmark'):.0%}")
    print(f"renarrated wording paradigm named {rate(nov, 'recognized_paradigm'):.0%}, "
          f"original study {rate(nov, 'named_study'):.0%}")
    names = {}
    for r in rows:
        n = (r.get("paradigm_name") or "").strip().lower()
        if n:
            names[n] = names.get(n, 0) + 1
    print("\nmost frequent paradigm names returned")
    for n, c in sorted(names.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {c:>4}  {n[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.check:
        check(models)
        return
    if not a.score_only:
        collect(models, a.reps, a.sleep)
    report()


if __name__ == "__main__":
    main()
