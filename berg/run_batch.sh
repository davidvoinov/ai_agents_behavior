#!/usr/bin/env bash
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
