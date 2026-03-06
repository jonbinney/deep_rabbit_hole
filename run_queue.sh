#!/bin/bash
# Simple job queue runner. Reads commands from a file one at a time, executing
# each sequentially. The file is re-read after each job finishes, so you can
# edit it (add, remove, reorder jobs) while a long-running job is in progress.
# The current job is removed from the file before execution, so the file always
# shows what's coming next. A timestamped backup is created at startup.

if [ -z "$1" ]; then
    echo "Usage: $0 <queue_file>"
    exit 1
fi

QUEUE_FILE="$1"

# Back up the job file with a timestamp
BACKUP="${QUEUE_FILE}.$(date +%Y%m%d_%H%M%S).bak"
cp "$QUEUE_FILE" "$BACKUP"
echo "Backed up queue to: $BACKUP"

while true; do
    if [ ! -f "$QUEUE_FILE" ]; then
        echo "Queue file not found: $QUEUE_FILE"
        exit 1
    fi

    # Read the first non-empty line
    CMD=$(sed -n '/\S/p' "$QUEUE_FILE" | head -1)

    if [ -z "$CMD" ]; then
        echo "Queue empty. Done."
        exit 0
    fi

    # Remove the first non-empty line before running
    TEMP=$(mktemp)
    awk 'BEGIN{found=0} /\S/{if(!found){found=1;next}} {print}' "$QUEUE_FILE" > "$TEMP"
    mv "$TEMP" "$QUEUE_FILE"

    echo "=== Running: $CMD ==="
    eval "$CMD"
    STATUS=$?
    echo "=== Finished (exit code $STATUS): $CMD ==="
done
