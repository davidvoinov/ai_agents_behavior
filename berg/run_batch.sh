#!/usr/bin/env bash
# run_batch.sh — start or resume a batch arm in the background, safely.
#
# Written because `export VAR=x nohup cmd` is a zsh error, not an assignment:
# zsh reads everything after the first word as further names to export and
# refuses on the script name. The export has to be its own statement, which is
# what this file guarantees.
#
# Rerunning is always safe. A live batch is re-attached from batches.jsonl
# rather than resubmitted, so nothing is ever paid for twice.
#
# Arms are locked one process each, so several may run side by side as long as
# each gets its own invocation. Passing a comma list to batch_runner would run
# them one after the other in a single process, which is not the same thing.
#
#   ./run_batch.sh gpt-5.6-nothink
#   ./run_batch.sh gpt-5.6-think sonnet-5-nothink    # side by side
#   tail -f log_batch_<arm>.log
set -u
cd "$(dirname "$0")"

ARMS=("${@:-gpt-5.6-nothink}")
DYADS="${DYADS:-100}"

need() {
    eval "v=\${$1:-}"
    if [ -z "$v" ]; then
        echo "  $1 is not set in this shell."
        echo "  export it on its own line first:"
        echo "      export $1=sk-..."
        exit 1
    fi
}

for ARM in "${ARMS[@]}"; do
    case "$ARM" in
        gpt*)    need OPENAI_API_KEY ;;
        sonnet*) need ANTHROPIC_API_KEY ;;
        all)     need OPENAI_API_KEY; need ANTHROPIC_API_KEY ;;
    esac
done

FAILED=0
for ARM in "${ARMS[@]}"; do
    LOG="log_batch_${ARM}.log"
    nohup python3 batch_runner.py --arms "$ARM" --dyads "$DYADS" > "$LOG" 2>&1 &
    PID=$!
    printf "%-20s pid %-8s -> %s\n" "$ARM" "$PID" "$LOG"
    sleep 4
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "  exited within four seconds:"
        sed 's/^/      /' "$LOG"
        FAILED=1
    fi
done

echo
echo "dyads    $DYADS per cell"
if [ "$FAILED" = 0 ]; then
    echo "all launched. watch with:"
    echo "    tail -f $(pwd)/log_batch_*.log"
    echo "    python3 batch_runner.py --status"
else
    echo "at least one arm did not start, see above"
    exit 1
fi
