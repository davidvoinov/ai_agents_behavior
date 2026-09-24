#!/usr/bin/env bash
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
