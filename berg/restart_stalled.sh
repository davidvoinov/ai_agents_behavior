#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

read -r -p "run 'pkill -f run_full_cycle' first — done? [y/N] " a
[ "$a" = "y" ] || { echo "stopping, nothing launched"; exit 1; }

MODELS="qwen3-think" N=100 MAX_TOKENS=3000 HARD_TIMEOUT=600 MIN_CALL_INTERVAL=0.2 \
  MAX_CONSEC_INFRA=25 PYTHONUNBUFFERED=1 \
  nohup python3 run_full_cycle.py > log3_qwenT.log 2>&1 &
echo "  qwen3-think    pid $!  (1981 dyads, max_tokens=3000, timeout=600)"

kimi () {  # tag  surfaces  conditions
  MODELS="kimi-k3" N=100 MAX_TOKENS=4000 HARD_TIMEOUT=600 MIN_CALL_INTERVAL=0.2 \
    MAX_CONSEC_INFRA=25 SURFACES="$2" CONDITIONS="$3" OUT_TAG="$1" PYTHONUNBUFFERED=1 \
    nohup python3 run_full_cycle.py > "log3_kimi_$1.log" 2>&1 &
  echo "  kimi $1  pid $!  ($2 / $3)"; }

kimi s1 strong_control     no_history
kimi s2 strong_control     social_history_pos
kimi s3 strong_control     social_history_neg
kimi s4 strong_control     communication
kimi s5 social_nonparadigm communication

cat <<'EOF'

kimi's 439 dyads split 5 ways should land in roughly 15 hours instead of 73.
Push it further by splitting a shard again if the pace still looks flat.

MERGE THE SHARDS when they finish, before any analysis:
  cat data/full/full_kimi-k3__s*.jsonl >> data/full/full_kimi-k3.jsonl
  rm data/full/full_kimi-k3__s*.jsonl
  python3 dedup_runs.py --apply      # safety net, shards should not overlap

watch the pace, not the log:
  python3 -c "
import json,glob
from datetime import datetime,timezone
def P(s):
    s=s.replace('Z','+00:00'); d=datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
for f in sorted(glob.glob('data/full/full_kimi-k3*.jsonl'))+['data/full/full_qwen3-think.jsonl']:
    if 'attrition' in f or 'purged' in f: continue
    ts=sorted(json.loads(l)['timestamp'] for l in open(f))
    if len(ts)<5: continue
    r=[P(t) for t in ts[-30:]]; span=(r[-1]-r[0]).total_seconds()/60
    print(f'{f.split(\"full_\")[-1]:<34} {len(ts):>5} dyads  {len(r)/max(span,1):.2f}/min')
"
EOF
