#!/usr/bin/env bash
# relaunch_topup.sh — refill the cells emptied by the parser and token-cap faults.
#
# Run this from the berg/ directory on a machine that can reach the providers.
#   cd berg && bash relaunch_topup.sh
#
# Prerequisites, already done offline:
#   reparse.py --apply             recovered 1466 role-decisions from stored replies
#   reclassify_attrition.py --apply rebuilt the ledger from the raw text
#   purge_unusable.py --apply      archived the failed dyads so a resume re-runs them
#
# Token budget, and why it differs by arm
# ---------------------------------------
# The cap never bound in the non-reasoning arms — zero truncations across
# glm-5.2-nothink, minimax-m3-nothink, qwen3-nothink, gpt-oss-120b and
# llama-3.3-70b, with the longest reply well inside 1500 tokens. A limit that
# never binds is not part of the treatment, so those arms keep 1500 and their
# new dyads stay comparable with the ones already collected. The reasoning arms
# and deepseek-v4-flash were cut by the cap, so they get 4000. No dyad in the
# final dataset is truncated either way, which is what removes the bias.
#
# One process per model. Two processes writing one file produce duplicates,
# since both read the same list of finished dyads at startup.
set -u
cd "$(dirname "$0")"

launch () {   # tag  models  max_tokens
  MODELS="$2" N=100 MAX_TOKENS="$3" MIN_CALL_INTERVAL=0.2 MAX_CONSEC_INFRA=15 \
    PYTHONUNBUFFERED=1 nohup python3 run_full_cycle.py > "log_$1.log" 2>&1 &
  echo "$!" > "pid_$1"
  printf '  %-26s pid %s  (max_tokens=%s)\n' "$1" "$!" "$3"
}

echo "launching top-up runs (about 8300 dyads, roughly \$9 at list prices)"

# cap never bound here, so it stays where the existing data was collected
launch cheap  "gpt-oss-120b,qwen3-nothink,minimax-m3-nothink,deepseek-v4-pro,glm-5.2-nothink,llama-3.3-70b" 1500

# cap did bind here
launch flash    "deepseek-v4-flash"       4000
launch kimi     "kimi-k3"                 4000
launch qwenT    "qwen3-think"             4000
launch glmT     "glm-5.2-think"           4000
launch minimaxT "minimax-m3-think"        4000
launch dsT      "deepseek-v4-pro-think"   4000

cat <<'EOF'

watch progress:   tail -f berg/log_*.log
check for stalls: grep -l ABORT berg/log_*.log
count coverage:   python3 -c "
import json,glob,collections
for p in sorted(glob.glob('data/full/full_*.jsonl')):
    if any(x in p for x in ('attrition','quarantine','purged')): continue
    c=collections.Counter()
    for l in open(p):
        r=json.loads(l)
        if r['investor_parse_clean'] and r['trustee_parse_clean']:
            c[f\"{r['surface']}/{r['condition']}\"]+=1
    short=sum(1 for k,v in c.items() if v<100)+ (28-len(c))
    print(f\"{p.split('full_')[-1][:-6]:<24} cells={len(c):>3} usable={sum(c.values()):>5} short={short}\")
"

when it finishes, re-run the ledger so the paper reports the final split:
  python3 reclassify_attrition.py --apply
EOF
