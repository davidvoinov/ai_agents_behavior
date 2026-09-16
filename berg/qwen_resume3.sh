#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
MAIN=data/full/full_qwen3-think.jsonl

read -r -p "run 'pkill -f run_full_cycle' and confirm with pgrep — done? [y/N] " a
[ "$a" = "y" ] || { echo "stopping, nothing changed"; exit 1; }

echo "== 1. merge the seven shards into main, then dedup =="
mkdir -p data/full/shards_done
for f in data/full/full_qwen3-think__*.jsonl; do
  case "$f" in *_attrition.jsonl) continue;; esac
  [ -e "$f" ] || continue
  cat "$f" >> "$MAIN"; mv "$f" data/full/shards_done/
done
mv data/full/full_qwen3-think__*_attrition.jsonl data/full/shards_done/ 2>/dev/null
python3 dedup_runs.py "$MAIN"

echo "== 2. seed three groups, balanced by what each still needs =="
python3 - "$MAIN" <<'PY'
import json, collections
from pathlib import Path
import sys
main = Path(sys.argv[1])
rows = [json.loads(l) for l in main.read_text().splitlines() if l.strip()]
GROUPS = {
    "g1": ["shifted", "reskin_shifted"],
    "g2": ["strong_novel", "strong_control"],
    "g3": ["social_nonparadigm", "reskin", "canonical"],
}
C = ['no_history','social_history_pos','social_history_neg','communication']
cells = collections.Counter(f"{r['surface']}/{r['condition']}" for r in rows)
for g, surfs in GROUPS.items():
    keep = [r for r in rows if r["surface"] in surfs]
    need = sum(max(0, 100 - cells.get(f"{s}/{c}", 0)) for s in surfs for c in C)
    out = main.with_name(main.stem + f"__{g}.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    print(f"   {g}  {'+'.join(surfs):<38} seed {len(keep):>4}  needs {need:>4}")
PY

echo "== 3. launch =="
for g in g1 g2 g3; do
  case $g in
    g1) S="shifted,reskin_shifted";;
    g2) S="strong_novel,strong_control";;
    g3) S="social_nonparadigm,reskin,canonical";;
  esac
  MODELS="qwen3-think" N=100 MAX_TOKENS=3000 HARD_TIMEOUT=600 MIN_CALL_INTERVAL=0.2 \
    MAX_CONSEC_INFRA=25 SURFACES="$S" OUT_TAG="$g" PYTHONUNBUFFERED=1 \
    nohup python3 run_full_cycle.py > "log7_qwen_$g.log" 2>&1 &
  printf '   %-4s pid %-8s %s\n' "$g" "$!" "$S"
done

cat <<'EOF'

1554 dyads at the measured 1.36/min is about 19 hours. Check after an hour, not
after ten minutes -- the early rate always overstates, which is the mistake that
put seven shards on the machine.

PROGRESS (new dyads only, seed excluded):
  python3 -c "
import json,glob
from datetime import datetime,timezone
def P(s):
    s=s.replace('Z','+00:00'); d=datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
seed={json.loads(l)['dyad_id'] for l in open('data/full/full_qwen3-think.jsonl')}
ts=[]
for f in sorted(glob.glob('data/full/full_qwen3-think__g?.jsonl')):
    rows=[json.loads(l) for l in open(f)]
    new=[r for r in rows if r['dyad_id'] not in seed]; ts+=[r['timestamp'] for r in new]
    print(f'{f.split(\"__\")[-1][:-6]:<6} {len(new):>4} new')
ts.sort()
if ts:
    span=(P(ts[-1])-P(ts[0])).total_seconds()/60
    print(f'{len(ts)} new in {span:.0f} min = {len(ts)/max(span,1):.2f}/min -> {(1554-len(ts))/max(len(ts)/max(span,1),0.01)/60:.1f} h left')
"

IF IT ABORTS AGAIN on connection/timeout, drop to two shards or one. A run that
finishes slowly beats a run that dies, and every abort costs the partial cell.

WHEN DONE:
  cat data/full/full_qwen3-think__g?.jsonl >> data/full/full_qwen3-think.jsonl
  python3 dedup_runs.py data/full/full_qwen3-think.jsonl
  mv data/full/full_qwen3-think__g?.jsonl data/full/shards_done/
  python3 reclassify_attrition.py --apply
EOF
