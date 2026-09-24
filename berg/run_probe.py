from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import berg_trust_game as B
import reskin as R

OUT = Path(__file__).resolve().parent / "data" / "probe_results.jsonl"

MODELS = ["gpt-oss-120b", "qwen3-nothink", "deepseek-v4-flash",
          "deepseek-v4-pro-think", "deepseek-v4-pro", "kimi-k3",
          "minimax-m3-think", "minimax-m3-nothink", "glm-5.2-think",
          "glm-5.2-nothink", "llama-3.3-70b"]

LABEL = {"gpt-oss-120b": "gpt-oss-120b", "qwen3-nothink": "Qwen3-32B",
         "deepseek-v4-flash": "DeepSeek-V4-Flash",
         "deepseek-v4-pro-think": "DeepSeek-V4-Pro (r)",
         "deepseek-v4-pro": "DeepSeek-V4-Pro", "kimi-k3": "Kimi-K3",
         "minimax-m3-think": "MiniMax-M3 (r)", "minimax-m3-nothink": "MiniMax-M3",
         "glm-5.2-think": "GLM-5.2 (r)", "glm-5.2-nothink": "GLM-5.2",
         "llama-3.3-70b": "Llama-3.3-70B"}

SURFACES = ["canonical", "social_nonparadigm", "shifted", "reskin",
            "reskin_shifted", "strong_control", "strong_novel"]

def done(path: Path) -> set[tuple[str, str, int]]:
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add((r["model"], r["surface"], r["rep"]))
    return seen

def collect(models: list[str], reps: int, sleep: float) -> None:
    import providers as P

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = done(OUT)
    missing = P.require_keys([m for m in models if m in P.PACK])
    if missing:
        print("missing API keys, these configurations are skipped:")
        for m in missing:
            print("   ", m)
        models = [m for m in models if not any(x.startswith(m + " ") for x in missing)]

    todo = [(m, s, i) for m in models for s in SURFACES for i in range(reps)
            if (m, s, i) not in seen]
    print(f"{len(todo)} calls to make, {len(seen)} already stored\n")

    with OUT.open("a") as fh:
        for n, (model, surface, rep) in enumerate(todo, 1):
            cfg = R.RESKINS[surface]
            system, user = R.recognition_probe_prompt(B._rules_block(cfg))
            try:
                chat = P.chat_for(model, temperature=0.0, max_tokens=600)
                text = chat(system, user)
                rec = R.score_recognition(text)
                rec.update(model=model, surface=surface, rep=rep, ok=True)
            except Exception as exc:
                rec = {"model": model, "surface": surface, "rep": rep,
                       "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            mark = "." if rec.get("ok") else "x"
            print(f"{mark}", end="", flush=True)
            if n % 70 == 0:
                print(f"  {n}/{len(todo)}")
            time.sleep(sleep)
    print("\nstored in", OUT)

def report() -> None:
    import numpy as np

    if not OUT.exists():
        sys.exit("no probe_results.jsonl, run the collection first")
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    if not rows:
        sys.exit("every probe call failed, nothing to score")

    def rate(sel, field):
        v = [r[field] for r in sel]
        return float(np.mean(v)) if v else float("nan")

    print("Recognition probe. Share of probe calls in which a fresh instance "
          "named\nthe paradigm, named the original study, and stated a "
          "quantitative human result.\n")
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
            grand[s].append(v)
            cells.append("     --   " if ss == [] else f"{v:>10.0%}")
        print(f"{LABEL.get(m, m):<22}" + "".join(cells))
    print("-" * len(head))
    print(f"{'all configurations':<22}" +
          "".join(f"{np.mean(grand[s]):>10.0%}" if grand[s] else "     --   "
                  for s in SURFACES))

    can = [r for r in rows if r["surface"] == "canonical"]
    nov = [r for r in rows if r["surface"] == "strong_novel"]
    print(f"\nunder the canonical wording the paradigm is named in "
          f"{rate(can, 'recognized_paradigm'):.0%} of calls, the original study "
          f"in {rate(can, 'named_study'):.0%},\nand a quantitative human result "
          f"is stated in {rate(can, 'approx_benchmark'):.0%}, while the exact "
          f"means of the\noriginal study appear in "
          f"{rate(can, 'leaked_benchmark'):.0%}")
    print(f"under the renarrated wording the paradigm is named in "
          f"{rate(nov, 'recognized_paradigm'):.0%} of calls and the original "
          f"study in {rate(nov, 'named_study'):.0%}")

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
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if not a.score_only:
        collect([m.strip() for m in a.models.split(",") if m.strip()],
                a.reps, a.sleep)
    report()

if __name__ == "__main__":
    main()
