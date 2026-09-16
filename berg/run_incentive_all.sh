#!/usr/bin/env bash
# run_incentive_all.sh — the positive control on every remaining configuration.
#
# Seven of the eleven are already collected. The five below are what is left,
# and they run in parallel because they sit on independent Nebius endpoints and
# the earlier serial pass showed a single model reaches about 1.6x its
# contended rate. Wall clock is therefore bounded by the slowest, not the sum.
#
# The whole set costs under a dollar. Time is the only real constraint, and the
# two slow ones are DeepSeek-V4-Flash and GLM-5.2-think at roughly a day each.
#
#   ./run_incentive_all.sh            # all five, in parallel
#   ./run_incentive_all.sh fast       # the three that finish inside five hours
#
# Watch with:  tail -f berg/log_inc_*.log
set -u
cd "$(dirname "$0")"

FAST=(minimax-m3-nothink kimi-k3 deepseek-v4-pro-think)
SLOW=(deepseek-v4-flash glm-5.2-think)

if [ "${1:-all}" = "fast" ]; then
    MODELS=("${FAST[@]}")
else
    MODELS=("${FAST[@]}" "${SLOW[@]}")
fi

for m in "${MODELS[@]}"; do
    # Region pinning needs no special case here. GLM-5.2 is served only from
    # uk-south1, and its PACK entry already carries that provider, which
    # providers.model_config resolves on its own.
    log="log_inc_${m}.log"
    echo "launching ${m} -> ${log}"
    MODELS="$m" N=100 K_LEVELS="0,0.3333,0.6667" MAX_TOKENS=3000 \
        nohup python3 run_incentive_control.py > "$log" 2>&1 &
    sleep 5          # stagger the starts so the first requests do not collide
done

echo
echo "launched ${#MODELS[@]} runs. progress:"
echo "  tail -f log_inc_*.log"
echo "  python3 -c \"import subprocess;print(subprocess.run(['ls','-la','data/incentive'],capture_output=True,text=True).stdout)\""
wait
echo "all runs finished"
