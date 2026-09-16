#!/usr/bin/env python3
"""
run_judge.py — code communication transcripts with an independent LLM judge.

Judge independence: the judge must NOT be a model under study in the dyad. That
was satisfied when only Qwen transcripts were coded and gpt-oss-120b sat outside
the coded set. It is NOT satisfied once every configuration is coded, since
gpt-oss-120b is one of them and would be reading its own transcripts. Pick a
judge that appears nowhere in the study, list what the provider currently serves
with `python3 list_models.py`, and register it in providers.py before running.
Temperature is 0 so the same transcript codes the same way on a re-run.

Where the transcripts come from is _transcripts.py, which reads both the
data/full layout of the open-weight arms and the data/batch layout of the four
closed frontier ones. Globbing data/full alone made those four invisible here
without reporting anything, which is why the lookup now lives in one module.

Usage:
    MODELS_JUDGE=judge-gptoss python3 run_judge.py
    ONLY_MODELS=gpt-5.6-think,gpt-5.6-nothink python3 run_judge.py   # a subset
Output -> data/coding/judge_comm.jsonl   (resumable, one row per coder)
"""
import os, json, glob
from pathlib import Path
try:
    from dotenv import load_dotenv
    _e = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_e if _e.exists() else None)
except ImportError:
    pass
import berg_trust_game as B, providers as P, code_comm as CC, attrition as ATT
import _transcripts as T

def speaker_of(m: dict) -> str:
    """Who said it, in the words the codebook uses.

    The engine writes `speaker` and `label`, never `role`. Reading `role` gave
    every line the placeholder `?`, so the judge saw an unlabelled exchange and
    had to infer the roles from content while being asked to code
    promise_investor and promise_trustee separately. `label` carries the surface
    wording, which names the paradigm on the canonical surface, so `speaker`
    is used and the surface stays invisible to the coder.
    """
    s = str(m.get("speaker") or m.get("role") or "").strip().lower()
    return {"investor": "Investor", "trustee": "Trustee"}.get(s, s.title() or "Party")


def transcript_text(rec):
    tx = rec.get("transcript") or []
    if isinstance(tx, str): return tx
    out=[]
    for m in tx:
        if isinstance(m, dict):
            out.append(f"{speaker_of(m)}: {m.get('message', m.get('content',''))}")
        else: out.append(str(m))
    return "\n".join(out)

def main():
    key = os.environ.get("MODELS_JUDGE", "gpt-oss-120b")
    STUDY = {"deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-pro-think",
             "glm-5.2-nothink", "glm-5.2-think", "gpt-oss-120b", "kimi-k3",
             "llama-3.3-70b", "minimax-m3-nothink", "minimax-m3-think",
             "qwen3-nothink", "qwen3-think",
             # the four closed frontier arms, added 2026-09-13
             "gpt-5.6-nothink", "gpt-5.6-think",
             "sonnet-5-nothink", "sonnet-5-think"}
    # A judge that is also a subject is allowed only when the key says so, which
    # keeps the choice in the record instead of in an environment variable.
    judge_model = P.PACK[key].model if key in P.PACK else key
    study_models = {P.PACK[k].model for k in STUDY if k in P.PACK}
    if judge_model in study_models:
        if not key.startswith("judge-"):
            raise SystemExit(
                f"{key!r} is a model under study, so it would code its own "
                f"transcripts.\nScreen an independent judge with screen_judges.py, "
                f"or register the\ndependence explicitly under a judge- key.")
        print(f"    NOTE {judge_model} is also a configuration in the study, so it "
              f"reads its\n         own transcripts among the rest. Report the "
              f"dependence in the methods.")
    miss = P.require_keys([key])
    if miss: raise SystemExit("Missing key: " + "; ".join(miss))
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))   # v2 brief is long; 1200 truncated 31% of calls
    mcfg = P.model_config(key, temperature=0.0, max_tokens=max_tokens)

    # The judge is called directly instead of through the game engine's chat_fn,
    # because response_format is a top level parameter and the engine has no way
    # to carry it. Constraining the reply to the codebook schema is what stops a
    # judge that thinks out loud from filling the budget before it answers.
    from openai import OpenAI
    _client = OpenAI(base_url=mcfg.base_url,
                     api_key=os.environ.get(mcfg.api_key_env, "MISSING"))

    def chat(system: str, user: str) -> str:
        kw = dict(model=mcfg.model, temperature=0.0, max_tokens=max_tokens,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                  response_format={"type": "json_schema",
                                   "json_schema": CC.JUDGE_SCHEMA})
        if mcfg.extra_body:
            kw["extra_body"] = mcfg.extra_body
        resp = _client.chat.completions.create(**kw)
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            return ""
        return msg.content or ""

    # Only the estimating sample is coded. The archives of failed dyads and the
    # shard files both match the same glob, and coding either would put dyads
    # into the reliability figures that no estimate uses. A dyad also enters only
    # when both roles parsed, since compliance compares a coded plan against a
    # realized decision and an unparsed decision has none.
    only = os.environ.get("ONLY_MODELS", "").strip()
    keep = {m.strip() for m in only.split(",") if m.strip()}
    rows = [r for arm, r in T.communication() if not keep or arm in keep]
    if keep:
        missing = keep - {r["model_key"] for r in rows}
        if missing:
            raise SystemExit("no communication dyads found for: "
                             + ", ".join(sorted(missing)))
    out = Path("data/coding"); out.mkdir(parents=True, exist_ok=True)
    # one stable file, so a change of judge key does not orphan the codes already
    # collected. The coder is recorded per row, so a mixed file stays readable.
    legacy = out / "judge_gpt-oss-120b-nebius.jsonl"
    path = out / "judge_comm.jsonl"
    if legacy.exists() and not path.exists():
        path.write_text(legacy.read_text())
        print(f"carried {len(legacy.read_text().splitlines())} existing codes into {path.name}")
    done = set()
    if path.exists():
        kept = []
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            # Resume is per CODER, not per transcript. A second judge has to code
            # the whole set, otherwise the file mixes two instruments and no
            # estimate drawn from it rests on one coder. Codes from an earlier
            # judge are kept, and they double as the second reading for the
            # machine to machine reliability figure.
            if r.get("coded_clean"):
                kept.append(l)
                done.add((r.get("coder"), r["uid"]))
        path.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"kept {len(kept)} clean codes, failures will be retried")
    mine = sum(1 for c, _ in done if c == f"judge:{key}")
    print(f"{len(rows)} communication transcripts | {mine} already coded by this "
          f"judge, {len(done) - mine} by others -> {path}")

    # One call before the loop. A provider drops an unsupported response_format
    # without raising, so the only way to know the schema took is to read a reply.
    probe = CC.code_transcript(chat, "Investor: I will send 5.\nTrustee: I will return 8.")
    if not probe.get("coded_clean"):
        raise SystemExit(
            f"{key!r} did not return the codebook object on a trivial transcript.\n"
            f"raw={probe.get('raw','')[:300]!r}\n"
            "Screen for a judge that honours the schema with screen_judges.py "
            "before spending a run.")
    print("    schema probe OK, the judge returns the object")

    n_ok = n_fail = 0
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            uid = f"{r['model_key']}|{r['surface']}|{r['dyad_id']}"
            if (f"judge:{key}", uid) in done: continue
            try:
                codes = CC.code_transcript(chat, transcript_text(r))
            except Exception as e:
                print("  fail", uid, ATT.classify_exception(e)); n_fail += 1; continue
            if not codes.get("coded_clean"):
                n_fail += 1
                if n_fail <= 3:
                    print(f"  UNPARSED [{uid}] raw={codes.get('raw','')[:200]!r}")
            rec = {"uid": uid, "model_key": r["model_key"], "surface": r["surface"],
                   "dyad_id": r["dyad_id"], "sent": r["sent"], "returned": r["returned"],
                   "endowment": r["endowment"], "coder": f"judge:{key}", **codes}
            fh.write(json.dumps(rec, ensure_ascii=False)+"\n"); fh.flush()
            n_ok += 1
            if n_ok % 50 == 0: print(f"  coded {n_ok}")
    print(f"done. coded={n_ok} failed={n_fail} -> {path}")

if __name__ == "__main__":
    main()
