#!/usr/bin/env bash
#
# Auto-retry wrapper for run_biomni_qwen30ba3b_gspo_tis.sh
#
# Automatically relaunches training when it exits (crash, OOM, etc.)
# The script already uses resume_mode=latest, so it auto-resumes from checkpoints.
#
# Usage:
#   ./run_biomni_qwen30ba3b_gspo_tis_autoretry.sh [max_retries]
#
# Arguments:
#   max_retries: Maximum number of restart attempts (default: unlimited, use -1)
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_SCRIPT="$SCRIPT_DIR/run_biomni_qwen30ba3b_gspo_tis.sh"
LOG_DIR="$SCRIPT_DIR/../../logs"
WRAPPER_LOG="$LOG_DIR/autoretry_wrapper.log"

# Configuration
MAX_RETRIES="${1:--1}"  # -1 means unlimited
RETRY_DELAY_SECONDS=30
MIN_RUN_SECONDS=60      # Detect rapid crash loops

# Counters
ATTEMPT=0
TOTAL_RESTARTS=0

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$WRAPPER_LOG"
}

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    log "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

mkdir -p "$LOG_DIR"

log "=============================================="
log "Auto-retry wrapper started"
log "Training script: $TRAINING_SCRIPT"
log "Max retries: $MAX_RETRIES (-1 = unlimited)"
log "=============================================="

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    
    log "----------------------------------------------"
    log "Attempt #$ATTEMPT (Total restarts: $TOTAL_RESTARTS)"
    log "----------------------------------------------"
    
    start_time=$(date +%s)
    
    # Run training (script uses resume_mode=latest)
    bash "$TRAINING_SCRIPT"
    exit_code=$?
    
    end_time=$(date +%s)
    run_duration=$((end_time - start_time))
    
    log "Training exited with code $exit_code after ${run_duration}s"
    
    # Success - exit cleanly
    if [[ $exit_code -eq 0 ]]; then
        log "Training completed successfully!"
        exit 0
    fi
    
    # Check max retries
    if [[ $MAX_RETRIES -ne -1 ]] && [[ $TOTAL_RESTARTS -ge $MAX_RETRIES ]]; then
        log "ERROR: Max retries ($MAX_RETRIES) reached. Giving up."
        exit 1
    fi
    
    # Rapid crash detection
    if [[ $run_duration -lt $MIN_RUN_SECONDS ]]; then
        log "WARNING: Crashed after only ${run_duration}s - possible config issue"
        sleep $((RETRY_DELAY_SECONDS * 3))
    fi
    
    TOTAL_RESTARTS=$((TOTAL_RESTARTS + 1))
    
    log "Waiting ${RETRY_DELAY_SECONDS}s before retry..."
    sleep $RETRY_DELAY_SECONDS
done
