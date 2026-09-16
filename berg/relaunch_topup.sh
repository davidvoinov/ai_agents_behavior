#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

launch () {   # tag  models  max_tokens
  MODELS="$2" N=100 MAX_TOKENS="$3" MIN_CALL_INTERVAL=0.2 MAX_CONSEC_INFRA=15 \
    PYTHONUNBUFFERED=1 nohup python3 run_full_cycle.py > "log_$1.log" 2>&1 &
  echo "$!" > "pid_$1"
  printf '  %-26s pid %s  (max_tokens=%s)\n' "$1" "$!" "$3"
}

echo "launching top-up runs (about 8300 dyads, roughly \$9 at list prices)"

launch cheap  "gpt-oss-120b,qwen3-nothink,minimax-m3-nothink,deepseek-v4-pro,glm-5.2-nothink,llama-3.3-70b" 1500

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
