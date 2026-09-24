#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
launch () { MODELS="$2" N=100 MAX_TOKENS="$3" MIN_CALL_INTERVAL="${4:-0.2}" \
    MAX_CONSEC_INFRA=40 PYTHONUNBUFFERED=1 \
    nohup python3 run_full_cycle.py > "log2_$1.log" 2>&1 &
  echo "$!" > "pid2_$1"; printf '  %-14s pid %-8s max_tokens=%-6s pace=%s\n' "$1" "$!" "$3" "${4:-0.2}"; }

launch qwenT  "qwen3-think"        4000 1.0     # 1985 dyads, the big hole
launch kimi   "kimi-k3"            4000 1.0     #  468
launch glmN   "glm-5.2-nothink"    1500 0.2     #    1

launch dsT    "deepseek-v4-pro-think" 8000 0.5  #  220 (195 were empty)
launch glmT   "glm-5.2-think"         8000 0.5  #  178 (149 empty)
launch mmT    "minimax-m3-think"      8000 0.5  #   38 ( 24 empty)

launch flash  "deepseek-v4-flash"     4000 0.3  #  471
