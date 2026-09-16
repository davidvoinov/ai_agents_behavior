#!/usr/bin/env bash
# restart_stalled.sh — restart the two stalled models. They stalled for DIFFERENT
# reasons, so they get opposite treatment.
#
# qwen3-think — real failures. It generates near 23 tok/s with the thinking inline
#   in the content, so a 4000-token cap needs about 175s before any queueing and
#   the call crosses the 300s wall clock and is killed as a hang. Its dyad ids are
#   written with gaps (0027, 0039, 0044, 0053), which is what a run of killed calls
#   looks like. Fix: give the call more wall clock and hand back the 3000 cap that
#   its existing 821 dyads were collected under. That cap is visible in the data
#   for this family, since Qwen puts its reasoning in the content, so it should not
#   be moved for speed.
#
# kimi-k3 — NOT failing. The ledger for the stalled run reads 4666 ok against 15
#   timeouts. At 120 tok/s its 408 billed output tokens are 3.4s of generation, yet
#   a call takes about 200s, so 197 of those seconds are queueing at the provider.
#   Neither the cap nor the timeout touches that. The only lever is running the
#   queue waits concurrently, so kimi is sharded across processes instead.
#   Each shard writes its own file (OUT_TAG) because two writers on one file
#   interleave lines and both resume from the same list.
set -u
cd "$(dirname "$0")"

read -r -p "run 'pkill -f run_full_cycle' first — done? [y/N] " a
[ "$a" = "y" ] || { echo "stopping, nothing launched"; exit 1; }

# --- qwen3-think: one process, more wall clock, its own cap back -------------
MODELS="qwen3-think" N=100 MAX_TOKENS=3000 HARD_TIMEOUT=600 MIN_CALL_INTERVAL=0.2 \
  MAX_CONSEC_INFRA=25 PYTHONUNBUFFERED=1 \
  nohup python3 run_full_cycle.py > log3_qwenT.log 2>&1 &
echo "  qwen3-think    pid $!  (1981 dyads, max_tokens=3000, timeout=600)"

# --- kimi-k3: 5 shards over the cells it still needs -------------------------
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
