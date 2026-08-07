#!/usr/bin/env python3
"""
run_judge.py — code communication transcripts with an independent LLM judge.

Judge independence: the judge must NOT be a model under study in the dyad.
Transcripts here come from Qwen3-32B, so the judge is gpt-oss-120b via Nebius.
Temperature is 0 so the same transcript codes the same way on a re-run.

Usage:
    MODELS_JUDGE=gpt-oss-120b-nebius python run_judge.py
Output -> data/coding/judge_<model>.jsonl   (resumable)
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

def transcript_text(rec):
    tx = rec.get("transcript") or []
    if isinstance(tx, str): return tx
    out=[]
    for m in tx:
        if isinstance(m, dict):
            out.append(f"{m.get('role','?')}: {m.get('content', m.get('message',''))}")
        else: out.append(str(m))
    return "\n".join(out)

def main():
    key = os.environ.get("MODELS_JUDGE", "gpt-oss-120b-nebius")
    miss = P.require_keys([key])
    if miss: raise SystemExit("Missing key: " + "; ".join(miss))
    max_tokens = int(os.environ.get("MAX_TOKENS", "3000"))   # v2 brief is long; 1200 truncated 31% of calls
    mcfg = P.model_config(key, temperature=0.0, max_tokens=max_tokens)
    chat = B.make_chat_fn(mcfg)

    rows=[]
    for f in glob.glob("data/full/full_*.jsonl"):
        if "_attrition" in f: continue
        for l in open(f):
            if not l.strip(): continue
            r=json.loads(l)
            if r.get("condition")=="communication" and r.get("transcript"):
                rows.append(r)
    out = Path("data/coding"); out.mkdir(parents=True, exist_ok=True)
    path = out / f"judge_{key}.jsonl"
    done = set()
    if path.exists():
        kept = []
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("coded_clean"):          # only a clean code counts as done
                kept.append(l); done.add(r["uid"])
        path.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"kept {len(kept)} clean codes, failures will be retried")
    print(f"{len(rows)} communication transcripts | already coded {len(done)} -> {path}")

    n_ok = n_fail = 0
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            uid = f"{r['model_key']}|{r['surface']}|{r['dyad_id']}"
            if uid in done: continue
            try:
                codes = CC.code_transcript(chat, transcript_text(r))
            except Exception as e:
                print("  fail", uid, ATT.classify_exception(e)); n_fail += 1; continue
            if not codes.get("coded_clean"):
                n_fail += 1
            rec = {"uid": uid, "model_key": r["model_key"], "surface": r["surface"],
                   "dyad_id": r["dyad_id"], "sent": r["sent"], "returned": r["returned"],
                   "endowment": r["endowment"], "coder": f"judge:{key}", **codes}
            fh.write(json.dumps(rec, ensure_ascii=False)+"\n"); fh.flush()
            n_ok += 1
            if n_ok % 50 == 0: print(f"  coded {n_ok}")
    print(f"done. coded={n_ok} failed={n_fail} -> {path}")

if __name__ == "__main__":
    main()
