#!/usr/bin/env bash
#
# Mock Training Script for Testing Auto-Retry
#
# Simulates a training run that crashes after a configurable delay.
# Used to test the autoretry wrapper without waiting for real crashes.
#
# Usage:
#   ./mock_crash_training.sh [crash_delay_seconds] [exit_code]
#
# Arguments:
#   crash_delay_seconds: Time to wait before "crashing" (default: 15)
#   exit_code: Exit code to return (default: 1)
#
# Examples:
#   ./mock_crash_training.sh           # Crash after 15s with exit code 137
#   ./mock_crash_training.sh 30        # Crash after 30s
#   ./mock_crash_training.sh 10 137    # Crash after 10s with code 137 (like OOM kill)
#

CRASH_DELAY=${1:-15}
EXIT_CODE=${2:-137}

# Use ray directly from the venv
VENV_BIN=/dfs/scratch1/lansong/venvs/skyrl-agent/bin

echo "=============================================="
echo "Mock Training Script Started"
echo "=============================================="
echo "Crash delay: ${CRASH_DELAY}s"
echo "Exit code: $EXIT_CODE"
echo ""

# Check Ray cluster has 2 nodes before starting
echo "[$(date '+%H:%M:%S')] Checking Ray cluster status..."
RAY_STATUS=$($VENV_BIN/ray status 2>&1)
NODE_COUNT=$(echo "$RAY_STATUS" | grep -c "node_")

if [[ $NODE_COUNT -lt 2 ]]; then
    echo ""
    echo "ERROR: Ray cluster does not have 2 connected nodes!"
    echo "Current node count: $NODE_COUNT"
    echo ""
    echo "Ray status output:"
    echo "$RAY_STATUS"
    echo ""
    echo "Please ensure both head and worker nodes are connected before running training."
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Ray cluster OK: $NODE_COUNT nodes connected"
echo ""

# Simulate some startup work
echo "[$(date '+%H:%M:%S')] Initializing mock training..."
sleep 2

echo "[$(date '+%H:%M:%S')] Loading mock model..."
sleep 2

echo "[$(date '+%H:%M:%S')] Starting mock training loop..."
echo ""

# Countdown to crash
remaining=$CRASH_DELAY
while [[ $remaining -gt 0 ]]; do
    echo "[$(date '+%H:%M:%S')] Training step... (crashing in ${remaining}s)"
    sleep 5
    remaining=$((remaining - 5))
done

echo ""
echo "[$(date '+%H:%M:%S')] SIMULATING CRASH!"
echo "=============================================="

exit $EXIT_CODE
