#!/usr/bin/env bash
# finish_qwen.sh — consolidate what is finished, then shard the one model left.
#
# State this repairs
# ------------------
# kimi-k3 is DONE but split. The main run was never killed when the shards were
# launched, so both worked the same cells and 493 of the 500 shard dyads are also
# in the main file. The union is exactly 2800 over all 28 cells with nothing above
# 100, so merging on dyad_id completes the model. Either copy of a collided dyad
# is a valid independent draw from the same cell, and dedup_runs keeps the first
# written, which is a rule independent of the outcome and so does not bias.
#
# qwen3-think has the same collision, still live: two processes are appending to
# one file and 53 duplicate ids have accumulated, the newest 30 minutes ago.
#
# Why qwen3-think is sharded rather than tuned again. The timeout fix worked, its
# calls now succeed (4 gaps in the last 40 ids, against a solid run of failures
# before). It is simply slow in an honest way: about 3000 tokens of inline
# reasoning per call at roughly 23 tok/s is about 130s of generation, and no
# setting removes that. At 0.16 dyad/min the remaining 1837 dyads need 190 hours
# in one process. Eight shards bring it near a day.
set -u
cd "$(dirname "$0")"

read -r -p "run 'pkill -f run_full_cycle' and confirm nothing is left — done? [y/N] " a
[ "$a" = "y" ] || { echo "stopping, nothing changed"; exit 1; }

echo "== 1. kimi: merge the shards, dedup, verify =="
cat data/full/full_kimi-k3__s?.jsonl >> data/full/full_kimi-k3.jsonl
python3 dedup_runs.py data/full/full_kimi-k3.jsonl
mkdir -p data/full/shards_done
mv data/full/full_kimi-k3__s?.jsonl data/full/full_kimi-k3__s?_attrition.jsonl data/full/shards_done/ 2>/dev/null

echo "== 2. qwen3-think: drop the duplicate lines the two writers left =="
python3 dedup_runs.py data/full/full_qwen3-think.jsonl

echo "== 3. qwen3-think: relaunch as 7 shards, one per surface =="
q () { MODELS="qwen3-think" N=100 MAX_TOKENS=3000 HARD_TIMEOUT=600 MIN_CALL_INTERVAL=0.2 \
    MAX_CONSEC_INFRA=25 SURFACES="$1" OUT_TAG="$1" PYTHONUNBUFFERED=1 \
    nohup python3 run_full_cycle.py > "log4_qwen_$1.log" 2>&1 &
  printf '   %-20s pid %s\n' "$1" "$!"; }
for s in canonical reskin shifted reskin_shifted strong_novel social_nonparadigm strong_control; do q "$s"; done

cat <<'EOF'

Each shard writes full_qwen3-think__<surface>.jsonl and resumes from its own
file, so a shard does NOT see the 981 dyads already in the main file and will
redo that surface's finished cells. That is intended -- the merge below keeps the
first copy of any dyad_id, so the existing work still wins and the shard only
adds what is missing. It costs some repeated calls on Qwen, the cheapest model
in the pack at $0.10/$0.30, so the waste is a couple of dollars against a week
of wall clock.

WHEN THEY FINISH:
  cat data/full/full_qwen3-think__*.jsonl >> data/full/full_qwen3-think.jsonl
  python3 dedup_runs.py data/full/full_qwen3-think.jsonl
  mv data/full/full_qwen3-think__*.jsonl data/full/shards_done/
  python3 reclassify_attrition.py --apply

WATCH THE PACE:
  python3 -c "
import json,glob
from datetime import datetime,timezone
def P(s):
    s=s.replace('Z','+00:00'); d=datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
tot=0
for f in sorted(glob.glob('data/full/full_qwen3-think*.jsonl')):
    if 'attrition' in f or 'purged' in f: continue
    ts=sorted(json.loads(l)['timestamp'] for l in open(f))
    tot+=len(ts)
    if len(ts)<5: continue
    r=[P(t) for t in ts[-30:]]; span=(r[-1]-r[0]).total_seconds()/60
    print(f'{f.split(\"full_\")[-1]:<40} {len(ts):>5}  {30/max(span,1):.2f}/min')
print('total dyads across shards+main:', tot)
"
EOF
