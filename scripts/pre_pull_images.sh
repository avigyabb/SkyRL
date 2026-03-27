#!/bin/bash
set -e

IMAGE_LIST="${1:-/tmp/r2e_images.txt}"
PARALLEL="${2:-4}"
LOG_FILE="/tmp/pre_pull.log"

if [ ! -f "$IMAGE_LIST" ]; then
    echo "Image list not found: $IMAGE_LIST"
    exit 1
fi

TOTAL=$(wc -l < "$IMAGE_LIST")
echo "Pre-pulling $TOTAL images with $PARALLEL parallel workers"
echo "Log: $LOG_FILE"

pull_image() {
    local img="$1"
    local idx="$2"
    local total="$3"
    if docker image inspect "$img" > /dev/null 2>&1; then
        echo "[$idx/$total] CACHED: $img"
        return 0
    fi
    for attempt in 1 2 3; do
        if docker pull "$img" > /dev/null 2>&1; then
            echo "[$idx/$total] PULLED: $img"
            return 0
        fi
        echo "[$idx/$total] RETRY $attempt: $img (rate limited, waiting 30s)"
        sleep 30
    done
    echo "[$idx/$total] FAILED: $img" | tee -a "$LOG_FILE.failed"
    return 1
}
export -f pull_image

> "$LOG_FILE"
> "$LOG_FILE.failed"

nl -ba "$IMAGE_LIST" | xargs -P "$PARALLEL" -I {} bash -c '
    idx=$(echo "{}" | awk "{print \$1}")
    img=$(echo "{}" | awk "{print \$2}")
    pull_image "$img" "$idx" "'"$TOTAL"'"
' 2>&1 | tee "$LOG_FILE"

FAILED=$(wc -l < "$LOG_FILE.failed")
echo ""
echo "Done. $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed."
if [ "$FAILED" -gt 0 ]; then
    echo "Failed images saved to $LOG_FILE.failed"
fi
